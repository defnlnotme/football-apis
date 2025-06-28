# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import sys
import pathlib
import argparse
import json
import re
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.logger import set_log_level
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from owl.utils import DocumentProcessingToolkit
from owl.utils.document_toolkit import DocumentProcessingToolkit, ScrapeOptions
from camel.utils import retry_on_error
import time
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Any
import shutil
from site_urls import SITE_URLS
from urllib.parse import urljoin
import datetime
import os
import pprint
from prompts import *
import mimetypes
import requests
from urllib.parse import urlparse
from agent import create_scraping_agent, WebScrapingAgent
import asyncio

# Import APIKeyManager from agent.py
from agent import APIKeyManager, api_key_manager

base_dir = pathlib.Path(__file__).parent
env_path = base_dir / ".envrc"
load_dotenv(dotenv_path=str(env_path))

# Global logging level configuration
GLOBAL_LOG_LEVEL = logging.INFO

def set_global_log_level(level: int):
    """Set the global logging level for the entire application."""
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = level
    # Update the root logger if it exists
    root_logger = logging.getLogger()
    root_logger.setLevel(GLOBAL_LOG_LEVEL)
    # Update all handlers
    for handler in root_logger.handlers:
        handler.setLevel(GLOBAL_LOG_LEVEL)

set_log_level(level=GLOBAL_LOG_LEVEL)

class Models:
    gemini_flash = "gemini-2.5-flash"
    gemini_flash_lite = "gemini-2.5-flash-lite-preview-06-17"

    flash = gemini_flash
    flash_lite = gemini_flash_lite

# Use the root logger for this script
logger = logging.getLogger()

@retry_on_error()
def patched_extract_webpage_content(self, url: str) -> str:
    api_key = os.getenv("FIRECRAWL_API_KEY")
    from firecrawl import FirecrawlApp

    # Initialize the FirecrawlApp with your API key
    app = FirecrawlApp(api_key=api_key)

    resp = app.crawl_url(
        url, limit=1, scrape_options=ScrapeOptions(formats=["html"])
    )
    data = resp.data
    logger.debug(f"Extracted data from {url}: {len(data)} items")
    if len(data) == 0:
        if resp.success:
            return "No content found on the webpage."
        else:
            return "Error while crawling the webpage."

    return str(data[0].html)

def patched_is_webpage(self, url: str) -> bool:
    try:
        parsed_url = urlparse(url)
        is_url = all([parsed_url.scheme, parsed_url.netloc])
        if not is_url:
            return False
        path = parsed_url.path
        file_type, _ = mimetypes.guess_type(path)
        if file_type is not None and "text/html" in file_type:
            return True
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MyBot/1.0; +https://example.com/bot)"}
        response = requests.head(url, allow_redirects=True, timeout=10, headers=headers)
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error while checking the URL: {e}")
        return False
    except TypeError:
        return True

TEAM_DATA_CACHE_TTL_SECONDS = 3600  # 1 hour


def get_team_data_cache_path(data_dir: str, filename: str, category: str) -> str:
    return os.path.join(data_dir, f"{filename}_{category}.json")

def is_team_data_cache_valid(cache_path: str) -> bool:
    if not os.path.exists(cache_path):
        return False
    file_age = datetime.datetime.now().timestamp() - os.path.getmtime(cache_path)
    return file_age < TEAM_DATA_CACHE_TTL_SECONDS

def load_team_data_cache(cache_path: str) -> dict:
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_team_data_cache(cache_path: str, data: dict):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_cache_dir(site_name: str) -> pathlib.Path:
    """Get the cache directory for a given site, ensure it exists."""
    safe_name = re.sub(r'[^\w\-_.]', '_', site_name)
    cache_dir = base_dir / "cache" / safe_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def get_cache_file_path(site_name: str) -> pathlib.Path:
    """Get the cache file path for a given site (HTML)."""
    cache_dir = get_cache_dir(site_name)
    return cache_dir / "page.html"

def get_competitions_cache_file_path(site_name: str, group: Optional[str] = None, path: Optional[str] = None) -> pathlib.Path:
    """Get the competitions cache file path for a given site (JSON), optionally keyed by group and path."""
    cache_dir = get_cache_dir(site_name)
    parts = ["competitions"]
    if group:
        parts.append(re.sub(r'[^\w\-_.]', '_', group))
    if path:
        parts.append(re.sub(r'[^\w\-_.]', '_', path))
    filename = "_".join(parts) + ".json"
    return cache_dir / filename

def is_cache_valid(site_name: str, cache_days: int) -> bool:
    """Check if cached content is still valid.

    Args:
        site_name (str): The name of the site
        cache_days (int): Number of days to cache (0 = no caching)

    Returns:
        bool: True if cache is valid, False otherwise
    """
    if cache_days == 0:
        return False

    cache_file = get_cache_file_path(site_name)
    if not cache_file.exists():
        return False

    # Check if cache file is older than cache_days
    file_age = time.time() - cache_file.stat().st_mtime
    max_age_seconds = cache_days * 24 * 60 * 60

    return file_age < max_age_seconds

def is_cache_valid_group(site_name: str, cache_days_obj: dict, group: str = "default") -> bool:
    cache_days = cache_days_obj.get(group, cache_days_obj["default"])
    return is_cache_valid(site_name, cache_days)

def load_cached_content(site_name: str) -> Optional[str]:
    """Load cached content for a site.

    Args:
        site_name (str): The name of the site

    Returns:
        Optional[str]: Cached content if available, None otherwise
    """
    try:
        cache_file = get_cache_file_path(site_name)
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded cached content for {site_name}: {len(content)} characters")
            return content
    except Exception as e:
        logger.error(f"Failed to load cached content for {site_name}: {str(e)}")

    return None

def save_cached_content(site_name: str, content: str) -> bool:
    """Save content to cache for a site.

    Args:
        site_name (str): The name of the site
        content (str): The content to cache

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cache_file = get_cache_file_path(site_name)
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Cached content for {site_name}: {len(content)} characters")
        return True
    except Exception as e:
        logger.error(f"Failed to cache content for {site_name}: {str(e)}")
        return False

def get_cache_age_days(site_name: str) -> Optional[float]:
    """Get the age of cached content in days.

    Args:
        site_name (str): The name of the site

    Returns:
        Optional[float]: Age in days if cache exists, None otherwise
    """
    cache_file = get_cache_file_path(site_name)
    if cache_file.exists():
        file_age_seconds = time.time() - cache_file.stat().st_mtime
        return file_age_seconds / (24 * 60 * 60)
    return None

def clear_cache(site_name: Optional[str] = None) -> int:
    """Clear cache files (now in cache/<site>/ subfolders)."""
    cleared_count = 0
    cache_root = base_dir / "cache"
    if site_name:
        safe_name = re.sub(r'[^\w\-_.]', '_', site_name)
        site_dir = cache_root / safe_name
        if site_dir.exists() and site_dir.is_dir():
            try:
                shutil.rmtree(site_dir)
                logger.info(f"Cleared cache for {site_name}")
                print(f"\033[92m✓ Cleared cache for {site_name}\033[0m")
                cleared_count = 1
            except Exception as e:
                logger.error(f"Failed to clear cache for {site_name}: {str(e)}")
                print(f"\033[91m✗ Failed to clear cache for {site_name}: {str(e)}\033[0m")
    else:
        if cache_root.exists() and cache_root.is_dir():
            for site_dir in cache_root.iterdir():
                if site_dir.is_dir():
                    try:
                        shutil.rmtree(site_dir)
                        logger.info(f"Cleared cache for {site_dir.name}")
                        cleared_count += 1
                    except Exception as e:
                        logger.error(f"Failed to clear cache for {site_dir.name}: {str(e)}")
            if cleared_count > 0:
                print(f"\033[92m✓ Cleared cache for {cleared_count} site{'s' if cleared_count != 1 else ''}\033[0m")
            else:
                print(f"\033[93mNo cache files found to clear\033[0m")
        else:
            print(f"\033[93mNo cache directory found to clear\033[0m")
    return cleared_count

def create_competition_extraction_agent(group: str, model_type=Models.flash_lite) -> ChatAgent:
    """Create a CAMEL agent for extracting competition data from HTML content, with group interpolation.

    Args:
        group (str): The group to extract competitions for.
    Returns:
        ChatAgent: The configured competition extraction agent
    """
    try:
        # Set the correct API key for the platform before creating the model
        platform = api_key_manager._get_platform_from_model(model_type)
        api_keys = api_key_manager._get_api_keys_for_platform(platform)
        if api_keys:
            idx = api_key_manager.current_key_indices.get(platform, 0)
            os.environ[f"{platform.upper()}_API_KEY"] = api_keys[idx]

        # Create a model for the agent
        model = ModelFactory.create(
            model_platform=ModelPlatformType.GEMINI,
            model_type=model_type
        )

        # Interpolate the group into the system prompt
        system_prompt = COMPETITION_EXTRACTION_PROMPT.format(group=group)

        # Create the agent with the competition extraction system prompt
        agent = ChatAgent(
            model=model,
            system_message=system_prompt
        )

        logger.info("Competition extraction agent created successfully")
        return agent

    except Exception as e:
        logger.error(f"Failed to create competition extraction agent: {str(e)}")
        raise e

def extract_competitions_with_llm(html_content: str, site_name: str, group: Optional[str] = None) -> Dict[str, Any]:
    """Extract competition list from HTML content using a CAMEL agent.

    Args:
        html_content (str): The HTML content to analyze
        site_name (str): The name of the site being analyzed
        group (Optional[str]): Group to filter competitions by

    Returns:
        Dict[str, Any]: Extracted competition data in structured format
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Log brief summary instead of full content
            html_length = len(html_content)
            logger.info(f"Starting competition extraction for {site_name} (HTML length: {html_length} chars)")

            # Create the competition extraction agent with group interpolation
            agent = create_competition_extraction_agent(group or "(not specified)")

            # Prepare the analysis prompt
            analysis_prompt = f"""
Please analyze the following HTML content from {site_name} and extract all football competitions, tournaments, and leagues mentioned.

HTML Content:
{html_content}  # Limit content to avoid token limits

Please provide a comprehensive list of all competitions found, organized by type and category.
"""

            # Create the human message using the correct method
            human_message = BaseMessage.make_user_message(
                role_name="Human",
                content=analysis_prompt
            )

            # Get response from the agent
            logger.info("Sending content to competition extraction agent")
            response = agent.step(human_message)

            if not response.msgs:
                logger.error("No response received from competition extraction agent")
                return {
                    "competitions": [],
                    "summary": {
                        "total_competitions": 0,
                        "categories": {
                            "leagues": 0,
                            "tournaments": 0,
                            "cups": 0,
                            "international": 0,
                            "regional": 0,
                            "youth": 0,
                            "womens": 0
                        }
                    },
                    "error": "No response from agent"
                }

            # Extract the response content
            agent_response = response.msgs[0].content
            logger.info(f"Received response from agent: {len(agent_response)} characters")

            # Try to parse JSON from the response
            try:
                # Look for JSON in the response
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', agent_response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON object in the response
                    json_match = re.search(r'\{[\s\S]*\}', agent_response)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        raise ValueError("No JSON found in response")

                # Parse the JSON
                competition_data = json.loads(json_str)
                logger.info(f"Successfully extracted {competition_data.get('summary', {}).get('total_competitions', 0)} competitions")

                return competition_data

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from agent response: {str(e)}")
                logger.debug(f"Raw response length: {len(agent_response)} chars")

                # Return a fallback structure with the raw response
                return {
                    "competitions": [],
                    "summary": {
                        "total_competitions": 0,
                        "categories": {
                            "leagues": 0,
                            "tournaments": 0,
                            "cups": 0,
                            "international": 0,
                            "regional": 0,
                            "youth": 0,
                            "womens": 0
                        }
                    },
                    "error": f"JSON parsing failed: {str(e)}",
                    "raw_response": agent_response[:1000]  # Include first 1000 chars for debugging
                }
                
        except Exception as e:
            error_str = str(e).lower()
            if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str:
                logger.warning(f"Rate limit ({retry_count + 1}/{max_retries}): {e}")
                
                # Record the rate limit error and rotate API key
                api_key_manager.record_rate_limit_error(Models.flash_lite)
                
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying with new API key...")
                    time.sleep(2)  # Brief pause before retry
                    continue
                else:
                    logger.error(f"Max retries reached for rate limit errors")
                    print(f"\033[91m✗ Rate limit exceeded after {max_retries} retries\033[0m")
                    return {
                        "competitions": [],
                        "summary": {
                            "total_competitions": 0,
                            "categories": {
                                "leagues": 0,
                                "tournaments": 0,
                                "cups": 0,
                                "international": 0,
                                "regional": 0,
                                "youth": 0,
                                "womens": 0
                            }
                        },
                        "error": f"Rate limit exceeded after {max_retries} retries"
                    }
            else:
                # Non-rate-limit error, don't retry
                logger.error(f"Exception during competition extraction: {str(e)}")
                logger.debug(traceback.format_exc())
                return {
                    "competitions": [],
                    "summary": {
                        "total_competitions": 0,
                        "categories": {
                            "leagues": 0,
                            "tournaments": 0,
                            "cups": 0,
                            "international": 0,
                            "regional": 0,
                            "youth": 0,
                            "womens": 0
                        }
                    },
                    "error": f"Extraction failed: {str(e)}"
                }
    
    # This should never be reached, but return empty result to satisfy type checker
    return {
        "competitions": [],
        "summary": {
            "total_competitions": 0,
            "categories": {
                "leagues": 0,
                "tournaments": 0,
                "cups": 0,
                "international": 0,
                "regional": 0,
                "youth": 0,
                "womens": 0
            }
        },
        "error": "Unexpected error in competition extraction"
    }

def save_competitions_to_file(competition_data: Dict[str, Any], site_name: str, path: Optional[str] = None) -> str:
    """Save extracted competition data to a JSON file in the cache folder.
    Args:
        competition_data (Dict[str, Any]): The extracted competition data
        site_name (str): The name of the site
        path (Optional[str]): The path key or sub-URL for this extraction (for filename)
    Returns:
        str: The path to the saved file
    """
    try:
        # Create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', site_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if path:
            # Replace '/' with '_' and strip leading/trailing underscores
            slug = path.replace('/', '_').strip('_')
            filename = f"{safe_name}_competitions_{slug}_{timestamp}.json"
        else:
            filename = f"{safe_name}_competitions_{timestamp}.json"
        file_path = get_cache_dir(site_name) / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(competition_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Competition data saved to: {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Failed to save competition data to file: {str(e)}")
        return f"Error saving file: {str(e)}"

def list_available_sites() -> None:
    """Display all available sites with their names and descriptions."""
    print("\n\033[94m=== Available Sites ===\033[0m")
    for name, info in SITE_URLS.items():
        print(f"\033[92m{name}\033[0m: {info['description']}")
        print(f"  URL: {info['url']}")
        cache_days_obj = info.get('cache_days', {"default": 1})
        cache_days = cache_days_obj["default"]
        if cache_days == 0:
            print(f"  Cache: Disabled")
        else:
            print(f"  Cache: {cache_days} day{'s' if cache_days != 1 else ''}")

            # Show cache status if available
            cache_age = get_cache_age_days(name)
            if cache_age is not None:
                if is_cache_valid(name, cache_days):
                    print(f"  Cache Status: Valid ({cache_age:.1f} days old)")
                else:
                    print(f"  Cache Status: Expired ({cache_age:.1f} days old)")
            else:
                print(f"  Cache Status: Not cached")
        print()

def get_site_info(site_name: str) -> Tuple[bool, str, str, dict]:
    """Get site information by name.

    Args:
        site_name (str): The name of the site to look up.

    Returns:
        Tuple[bool, str, str, dict]: (found, url, description, cache_days)
    """
    site_name_lower = site_name.lower()

    # Exact match
    if site_name_lower in SITE_URLS:
        info = SITE_URLS[site_name_lower]
        return True, info["url"], info["description"], info["cache_days"]

    # Partial match
    for name, info in SITE_URLS.items():
        if site_name_lower in name or name in site_name_lower:
            return True, info["url"], info["description"], info["cache_days"]

    return False, "", "", {"default": 1}


def extract_html_from_url(url: str) -> str:
    r"""Extract HTML content from a URL using DocumentProcessingToolkit.

    Args:
        url (str): The URL to extract HTML from.

    Returns:
        str: The extracted HTML content.
    """
    try:
        # Create a model for the toolkit (using a simple model)
        model = ModelFactory.create(
            model_platform=ModelPlatformType.GEMINI,
            model_type=Models.flash,
            model_config_dict={"temperature": 0},
        )

        # Initialize the DocumentProcessingToolkit
        doc_toolkit = DocumentProcessingToolkit(model=model)
        # Patch the method after instantiation
        doc_toolkit._extract_webpage_content = patched_extract_webpage_content.__get__(doc_toolkit, DocumentProcessingToolkit)
        doc_toolkit._is_webpage = patched_is_webpage.__get__(doc_toolkit, DocumentProcessingToolkit)

        logger.info(f"Extracting HTML from URL: {url}")

        # Extract the document content
        success, content = doc_toolkit.extract_document_content(url)

        if success:
            logger.info("HTML extraction successful")
            return content
        else:
            logger.error(f"Failed to extract HTML: length={len(content)} preview={content[:200]!r}")
            return f"Error: {content}"

    except Exception as e:
        logger.error(f"Exception during HTML extraction: {str(e)}")
        logger.debug(traceback.format_exc())
        return f"Exception: {str(e)}"

def save_html_to_file(html_content: str, site_name: str, path: Optional[str] = None) -> str:
    r"""Save HTML content to a file in the cache/<site>/ folder.

    Args:
        html_content (str): The HTML content to save.
        site_name (str): The name of the site for the filename.
        path (Optional[str]): The path key or sub-URL for this extraction (for filename)

    Returns:
        str: The path to the saved file.
    """
    try:
        # Create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', site_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if path:
            slug = path.replace('/', '_').strip('_')
            filename = f"{safe_name}_{slug}_{timestamp}.html"
        else:
            filename = f"{safe_name}_{timestamp}.html"
        file_path = get_cache_dir(site_name) / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML saved to: {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Failed to save HTML to file: {str(e)}")
        return f"Error saving file: {str(e)}"

def display_competitions(competition_data: Dict[str, Any], site_name: str) -> None:
    """Display extracted competitions in a nice format, including the competition URL."""
    competitions = competition_data.get("competitions", [])
    summary = competition_data.get("summary", {})

    if not competitions:
        print(f"\033[93mNo competitions found on {site_name}\033[0m")
        return

    print(f"\n\033[94m=== Competitions Found on {site_name} ===\033[0m")
    print(f"\033[92mTotal: {summary.get('total_competitions', len(competitions))}\033[0m")

    # Group competitions by type
    competitions_by_type = {}
    for comp in competitions:
        comp_type = comp.get("type", "unknown")
        if comp_type not in competitions_by_type:
            competitions_by_type[comp_type] = []
        competitions_by_type[comp_type].append(comp)

    # Display competitions by type
    for comp_type, comps in competitions_by_type.items():
        print(f"\n\033[95m{comp_type.upper()} ({len(comps)}):\033[0m")
        for comp in comps:
            name = comp.get("name", "Unknown")
            group = comp.get("group", "")
            season = comp.get("season", "")
            description = comp.get("description", "")
            url = comp.get("url") or comp.get("html_url")

            print(f"  • {name}")
            if url:
                print(f"    URL: {url}")
            if group:
                print(f"    Group: {group}")
            if season:
                print(f"    Season: {season}")
            if description:
                print(f"    Description: {description}")
            print()

def is_competitions_cache_valid(site_name: str, cache_days_obj: dict, group: Optional[str] = None, path: Optional[str] = None) -> bool:
    cache_days = cache_days_obj.get("competition", cache_days_obj["default"])
    cache_file = get_competitions_cache_file_path(site_name, group, path)
    if cache_days == 0:
        return False
    if not cache_file.exists():
        return False
    file_age = time.time() - cache_file.stat().st_mtime
    max_age_seconds = cache_days * 24 * 60 * 60
    return file_age < max_age_seconds

def load_competitions_cache(site_name: str, group: Optional[str] = None, path: Optional[str] = None) -> Optional[dict]:
    try:
        cache_file = get_competitions_cache_file_path(site_name, group, path)
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load competitions cache for {site_name} (group={group}, path={path}): {str(e)}")
    return None

def save_competitions_cache(site_name: str, data: dict, group: Optional[str] = None, path: Optional[str] = None) -> bool:
    try:
        cache_file = get_competitions_cache_file_path(site_name, group, path)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached competitions for {site_name} (group={group}, path={path})")
        return True
    except Exception as e:
        logger.error(f"Failed to cache competitions for {site_name} (group={group}, path={path}): {str(e)}")
        return False

def find_cached_files_by_regex(site_name: str, pattern: str) -> list:
    """
    Find cached files for a site whose filenames match a regex pattern.
    Args:
        site_name (str): The name of the site.
        pattern (str): The regex pattern to match filenames.
    Returns:
        List[pathlib.Path]: List of matching cached file paths.
    """
    import re
    cache_dir = get_cache_dir(site_name)
    if not cache_dir.exists() or not cache_dir.is_dir():
        return []
    regex = re.compile(pattern)
    return [f for f in cache_dir.iterdir() if f.is_file() and regex.search(f.name)]

def extract_all_competitions(site_name: str, site_info: dict, group: Optional[str] = None, path: Optional[str] = None, force_fetch: bool = False) -> dict:
    import os
    cache_file = get_competitions_cache_file_path(site_name, group, path)
    if cache_file.exists() and not force_fetch:
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load competitions.json for {site_name}: {str(e)}. Refetching...")
    base_url = site_info["url"]
    paths_info = site_info.get("paths", {})
    competitions_entry = paths_info.get("competitions", base_url)
    # Handle both list and string types
    if isinstance(competitions_entry, list):
        comp_urls = competitions_entry
    elif isinstance(competitions_entry, str):
        comp_urls = [competitions_entry]
    else:
        comp_urls = [base_url]
    if group and path:
        try:
            regex = re.compile(path)
            matched_urls = [u for u in comp_urls if regex.search(u)]
            if not matched_urls:
                logger.warning(f"No paths matched the regex: {path}")
            comp_urls = matched_urls
        except re.error as e:
            logger.error(f"Invalid regex for path: {path} ({e})")
            comp_urls = []
    all_competitions = []
    summary = {
        "total_competitions": 0,
        "categories": {
            "leagues": 0,
            "tournaments": 0,
            "cups": 0,
            "international": 0,
            "regional": 0,
            "youth": 0,
            "womens": 0
        }
    }
    seen = set()
    errors = []
    for comp_url in comp_urls:
        # Format comp_url if it contains placeholders
        if '{' in comp_url and '}' in comp_url:
            try:
                comp_url = fill_url_pattern(comp_url, argparse.Namespace(**{
                    'group': group,
                    'path': path
                }))
            except ValueError as e:
                print(f"\033[91mError: {e}\033[0m")
                return {"competitions": [], "summary": summary, "error": str(e)}
        # Resolve relative URLs
        if not comp_url.startswith("http://") and not comp_url.startswith("https://"):
            url = urljoin(base_url, comp_url)
        else:
            url = comp_url
        # Try to find a cached HTML file for this path
        slug = comp_url.replace('/', '_').strip('_')
        pattern = rf"{re.escape(slug)}.*\.html$"
        cached_files = find_cached_files_by_regex(site_name, pattern)
        if cached_files:
            # Use the most recent cached file (by modification time)
            cached_file = max(cached_files, key=lambda f: f.stat().st_mtime)
            with open(cached_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            logger.info(f"Loaded HTML from cache: {cached_file}")
        else:
            html_content = extract_html_from_url(url)
            # Save HTML for each path
            save_html_to_file(html_content, site_name, path=comp_url)
        if html_content.startswith("Error:") or html_content.startswith("Exception:"):
            errors.append(f"{url}: {html_content}")
            continue
        comp_data = extract_competitions_with_llm(html_content, site_name, group=group)
        # Add html_url to each competition
        for c in comp_data.get("competitions", []):
            c["html_url"] = url
        # Save each path's competitions file
        save_competitions_to_file(comp_data, site_name, path=comp_url)
        comps = comp_data.get("competitions", [])
        for c in comps:
            key = (c.get("name"), c.get("type"), c.get("group"), c.get("season"))
            if key not in seen:
                all_competitions.append(c)
                seen.add(key)
        # Merge category counts
        cats = comp_data.get("summary", {}).get("categories", {})
        for k, v in cats.items():
            summary["categories"][k] += v
    summary["total_competitions"] = len(all_competitions)
    result = {
        "competitions": all_competitions,
        "summary": summary
    }
    if errors:
        result["errors"] = errors
    return result

def extract_json_from_response(agent_response: str) -> dict:
    """
    Extracts a JSON object from a string, which may be wrapped in markdown code blocks.

    Args:
        agent_response: The string potentially containing the JSON data.

    Returns:
        A dictionary parsed from the JSON data.

    Raises:
        ValueError: If no valid JSON is found in the response.
    """
    # Corrected regex: Use `\s` instead of `\\s` in the raw string.
    # This correctly searches for whitespace characters.
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', agent_response)

    if json_match:
        # Group 1 captures the content between the markdown fences.
        json_str = json_match.group(1)
        logger.info("Successfully extracted JSON string from markdown block.")
    else:
        # Fallback 1: If no markdown block is found, search for the outer curly braces.
        # This is a broader search that finds the first '{' and the last '}'
        logger.warning("Could not find markdown JSON block. Trying fallback search for '{}'.")
        print(agent_response)
        json_match = re.search(r'\{[\s\S]*\}', agent_response)
        if json_match:
            # Group 0 captures the entire match, including the braces.
            json_str = json_match.group(0)
            logger.info("Successfully extracted JSON string using fallback curly brace search.")
        else:
            # Fallback 2: Try parsing the entire response string directly.
            logger.warning("Fallback search failed. Attempting to parse the entire response.")
            try:
                return json.loads(agent_response.strip())
            except json.JSONDecodeError:
                # If all methods fail, raise an error.
                raise ValueError("No valid JSON found in the response.")

    # Now, parse the JSON string obtained from the successful regex match.
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Found a potential JSON string but it could not be parsed: {e}")

def create_team_extraction_agent(competition: str) -> ChatAgent:
    """Create a CAMEL agent for extracting team data from HTML content, with competition interpolation.
    Args:
        competition (str): The competition ID to extract teams for.
    Returns:
        ChatAgent: The configured team extraction agent
    """
    try:
        platform = api_key_manager._get_platform_from_model(Models.flash)
        api_keys = api_key_manager._get_api_keys_for_platform(platform)
        if api_keys:
            idx = api_key_manager.current_key_indices.get(platform, 0)
            os.environ[f"{platform.upper()}_API_KEY"] = api_keys[idx]
        model = ModelFactory.create(
            model_platform=ModelPlatformType.GEMINI,
            model_type=Models.flash,
            model_config_dict={"temperature": 1/3},
        )
        system_prompt = TEAM_EXTRACTION_PROMPT.format(competition=competition)
        agent = ChatAgent(
            model=model,
            system_message=system_prompt
        )
        logger.info("Team extraction agent created successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to create team extraction agent: {str(e)}")
        raise e

def extract_teams_with_llm(html_content: str, site_name: str, competition: str) -> Dict[str, Any]:
    """Extract teams from HTML content using LLM."""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Log brief summary instead of full content
            html_length = len(html_content)
            logger.info(f"Starting team extraction for {site_name} (HTML length: {html_length} chars)")
            
            agent = create_team_extraction_agent(competition)
            analysis_prompt = f"""
Please analyze the following HTML content from {site_name} and extract all football teams and clubs participating in competition: {competition}.

HTML Content:
{html_content}

Please provide a comprehensive list of all teams found, organized by type and category.
"""
            human_message = BaseMessage.make_user_message(
                role_name="Human",
                content=analysis_prompt
            )
            logger.info("Sending content to team extraction agent")
            response = agent.step(human_message)
            if not response.msgs:
                logger.error("No response received from team extraction agent")
                return {
                    "teams": [],
                    "summary": {
                        "total_teams": 0,
                        "categories": {
                            "club": 0,
                            "national": 0,
                            "youth": 0,
                            "women": 0
                        }
                    },
                    "error": "No response from agent"
                }
            agent_response = response.msgs[0].content
            logger.info(f"Received response from agent: {len(agent_response)} characters")
            return extract_json_from_response(agent_response)
            
        except Exception as e:
            error_str = str(e).lower()
            if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str:
                logger.warning(f"Rate limit ({retry_count + 1}/{max_retries}): {e}")
                
                # Record the rate limit error and rotate API key
                api_key_manager.record_rate_limit_error(Models.flash_lite)
                
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying with new API key...")
                    time.sleep(2)  # Brief pause before retry
                    continue
                else:
                    logger.error(f"Max retries reached for rate limit errors")
                    print(f"\033[91m✗ Rate limit exceeded after {max_retries} retries\033[0m")
                    return {
                        "teams": [],
                        "summary": {
                            "total_teams": 0,
                            "categories": {
                                "club": 0,
                                "national": 0,
                                "youth": 0,
                                "women": 0
                            }
                        },
                        "error": f"Rate limit exceeded after {max_retries} retries"
                    }
            else:
                # Non-rate-limit error, don't retry
                logger.error(f"Exception during team extraction: {str(e)}")
                logger.debug(traceback.format_exc())
                return {
                    "teams": [],
                    "summary": {
                        "total_teams": 0,
                        "categories": {
                            "club": 0,
                            "national": 0,
                            "youth": 0,
                            "women": 0
                        }
                    },
                    "error": f"Extraction failed: {str(e)}"
                }
    
    # This should never be reached, but return empty result to satisfy type checker
    return {
        "teams": [],
        "summary": {
            "total_teams": 0,
            "categories": {
                "club": 0,
                "national": 0,
                "youth": 0,
                "women": 0
            }
        },
        "error": "Unexpected error in team extraction"
    }

def display_teams(team_data: Dict[str, Any], site_name: str) -> None:
    """Display extracted teams in a nice format for a specific competition."""
    teams = team_data.get("teams", [])
    summary = team_data.get("summary", {})
    if not teams:
        print(f"\033[93mNo teams found on {site_name}\033[0m")
        return
    print(f"\n\033[94m=== Teams Found on {site_name} ===\033[0m")
    print(f"\033[92mTotal: {summary.get('total_teams', len(teams))}\033[0m")
    # Group teams by type
    teams_by_type = {}
    for team in teams:
        team_type = team.get("type", "unknown")
        if team_type not in teams_by_type:
            teams_by_type[team_type] = []
        teams_by_type[team_type].append(team)
    for team_type, tms in teams_by_type.items():
        print(f"\n\033[95m{team_type.upper()} ({len(tms)}):\033[0m")
        for team in tms:
            name = team.get("name", "Unknown")
            group = team.get("group", "")
            league = team.get("league", "")
            description = team.get("description", "")
            print(f"  • {name}")
            if group:
                print(f"    Group: {group}")
            if league:
                print(f"    League: {league}")
            if description:
                print(f"    Description: {description}")
            print()

def get_teams_cache_file_path(site_name: str, competition: Optional[str] = None) -> pathlib.Path:
    """Get the cache file path for teams for a given site and competition ID."""
    cache_dir = get_cache_dir(site_name)
    if competition:
        safe_comp = re.sub(r'[^\w\-_.]', '_', competition)
        return cache_dir / f"teams_{safe_comp}.json"
    return cache_dir / "teams.json"

def is_teams_cache_valid(site_name: str, cache_days_obj: dict, competition: Optional[str] = None) -> bool:
    """Check if the teams cache is valid for a given site and competition ID."""
    cache_days = cache_days_obj.get("teams", cache_days_obj["default"])
    cache_file = get_teams_cache_file_path(site_name, competition)
    if cache_days == 0:
        return False
    if not cache_file.exists():
        return False
    file_age = time.time() - cache_file.stat().st_mtime
    max_age_seconds = cache_days * 24 * 60 * 60
    return file_age < max_age_seconds

def load_teams_cache(site_name: str, competition: Optional[str] = None) -> Optional[dict]:
    """Load teams cache for a given site and competition ID."""
    try:
        cache_file = get_teams_cache_file_path(site_name, competition)
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load teams cache for {site_name} (competition={competition}): {str(e)}")
    return None

def save_teams_cache(site_name: str, data: dict, competition: Optional[str] = None) -> bool:
    """Save teams cache for a given site and competition ID."""
    try:
        cache_file = get_teams_cache_file_path(site_name, competition)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached teams for {site_name} (competition={competition})")
        return True
    except Exception as e:
        logger.error(f"Failed to cache teams for {site_name} (competition={competition}): {str(e)}")
        return False

def scrape_site(site_name: str, url: str, description: str, cache_days_obj: dict = {"default": 1}, extract_competitions: bool = False, group: Optional[str] = None, path: Optional[str] = None, extract_teams: bool = False, competition: Optional[str] = None, force_fetch: bool = False) -> bool:
    """Scrape a specific site and optionally extract competitions. All other extraction should be done via the --extract flag and main().
    Args:
        site_name (str): The name of the site.
        url (str): The URL to scrape.
        description (str): Description of the site.
        cache_days_obj (dict): Number of days to cache content (0 = no caching).
        extract_competitions (bool): Whether to extract competitions using LLM.
        group (Optional[str]): Group to filter competitions by.
        path (Optional[str]): Path key or sub-URL to scrape (required with --group).
        competition (Optional[str]): Competition ID (slug, URL, or identifier) to extract teams for. (Unused)
        force_fetch (bool): If True, skip all cache and always fetch fresh data.
    Returns:
        bool: True if successful, False otherwise.
    """
    if force_fetch:
        print(f"\033[94mForce fetch: Skipping all cache and fetching fresh data\033[0m")
    cache_days = cache_days_obj["default"]
    logger.info(f"Starting extraction from: {site_name}")
    print(f"\033[94mScraping: {site_name}\033[0m")
    print(f"\033[94mDescription: {description}\033[0m")
    print(f"\033[94mURL: {url}\033[0m")
    if cache_days == 0:
        print(f"\033[94mCache: Disabled\033[0m")
    else:
        print(f"\033[94mCache: {cache_days} day{'s' if cache_days != 1 else ''}\033[0m")
    if extract_competitions:
        print(f"\033[94mCompetition extraction: Enabled\033[0m")
        print(f"\033[94mExtracting competitions from {site_name}...\033[0m")
        competition_data = extract_all_competitions(site_name, SITE_URLS[site_name], group=group, path=path, force_fetch=force_fetch)
        if "error" in competition_data:
            print(f"\033[93m⚠ Competition extraction failed: {competition_data['error']}\033[0m")
        else:
            save_competitions_cache(site_name, competition_data, group, path)
            comp_file_path = save_competitions_to_file(competition_data, site_name)
            if comp_file_path.startswith("Error"):
                print(f"\033[93m⚠ Failed to save competition data: {comp_file_path}\033[0m")
            else:
                total_competitions = competition_data.get('summary', {}).get('total_competitions', 0)
                print(f"\033[92m✓ Competition data extracted and saved to: {comp_file_path}\033[0m")
                print(f"\033[92m✓ Found {total_competitions} competitions\033[0m")
                categories = competition_data.get('summary', {}).get('categories', {})
                if categories:
                    print(f"\033[94mCompetition categories:\033[0m")
                    for category, count in categories.items():
                        if count > 0:
                            print(f"  - {category}: {count}")
                display_competitions(competition_data, site_name)
        return True
    print(f"\033[93mNo extraction type specified. Use --extract for all other extraction types.\033[0m")
    return False

def get_team_html_cache_path(site_name: str, team: str, data_type: str, year: str = "", competition: str = "", vs_team: str = "", group: str = "") -> pathlib.Path:
    """Get the cache path for a team's HTML file for a given data type, now including group if provided."""
    cache_dir = get_cache_dir(site_name)
    fname = f"{team}_{data_type}"
    if group:
        fname += f"_{group}"
    if year:
        fname += f"_{year}"
    if competition:
        fname += f"_{competition}"
    if vs_team:
        fname += f"_vs_{vs_team}"
    fname += ".html"
    return cache_dir / fname

TEAM_HTML_CACHE_TTL_SECONDS = 3600  # 1 hour

def is_team_html_cache_valid(cache_path: pathlib.Path) -> bool:
    if not cache_path.exists():
        return False
    file_age = time.time() - cache_path.stat().st_mtime
    return file_age < TEAM_HTML_CACHE_TTL_SECONDS

# --- Caching for extract_team_historical ---
def extract_team_historical(site_name: str, team: str, year: str, force_fetch: bool = False, args=None) -> str:
    paths = SITE_URLS.get(site_name, {}).get("paths", {})
    pattern = paths.get("historical")
    if not pattern:
        raise ValueError(f"No historical pattern for site {site_name}")
    # Ensure year_prev is set if needed
    if pattern and '{year_prev}' in pattern:
        if args and getattr(args, 'year_prev', None) is None and getattr(args, 'year', None) is not None:
            try:
                setattr(args, 'year_prev', str(int(args.year) - 1))
            except Exception:
                pass
    sub_path = fill_url_pattern(pattern, args)
    url = urljoin(SITE_URLS[site_name]["url"], sub_path)
    cache_path = get_team_html_cache_path(site_name, team, "historical", year=year)
    if not force_fetch and is_team_html_cache_valid(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    html_content = extract_html_from_url(url)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return html_content

# --- Caching for extract_team_news ---
def extract_team_news(site_name: str, team: str, force_fetch: bool = False, args=None) -> str:
    paths = SITE_URLS.get(site_name, {}).get("paths", {})
    pattern = paths.get("news")
    if not pattern:
        raise ValueError(f"No news pattern for site {site_name}")
    sub_path = fill_url_pattern(pattern, args)
    url = urljoin(SITE_URLS[site_name]["url"], sub_path)
    cache_path = get_team_html_cache_path(site_name, team, "news")
    if not force_fetch and is_team_html_cache_valid(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    html_content = extract_html_from_url(url)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return html_content

# --- Caching for extract_team_appearances ---
def extract_team_appearances(site_name: str, team: str, competition: str, year: str, force_fetch: bool = False, args=None) -> str:
    paths = SITE_URLS.get(site_name, {}).get("paths", {})
    pattern = paths.get("appearances")
    if not pattern:
        raise ValueError(f"No appearances pattern for site {site_name}")
    # Ensure year_prev is set if needed
    if pattern and '{year_prev}' in pattern:
        if args and getattr(args, 'year_prev', None) is None and getattr(args, 'year', None) is not None:
            try:
                setattr(args, 'year_prev', str(int(args.year) - 1))
            except Exception:
                pass
    sub_path = fill_url_pattern(pattern, args)
    url = urljoin(SITE_URLS[site_name]["url"], sub_path)
    cache_path = get_team_html_cache_path(site_name, team, "appearances", year=year, competition=competition)
    if not force_fetch and is_team_html_cache_valid(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    html_content = extract_html_from_url(url)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return html_content

# --- Caching for extract_team_squad ---
def extract_team_squad(site_name: str, team: str, year: str, force_fetch: bool = False, args=None) -> str:
    paths = SITE_URLS.get(site_name, {}).get("paths", {})
    pattern = paths.get("squad")
    if not pattern:
        raise ValueError(f"No squad pattern for site {site_name}")
    sub_path = fill_url_pattern(pattern, args)
    url = urljoin(SITE_URLS[site_name]["url"], sub_path)
    cache_path = get_team_html_cache_path(site_name, team, "squad", year=year)
    if not force_fetch and is_team_html_cache_valid(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    html_content = extract_html_from_url(url)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return html_content


def create_team_data_extraction_agent(team: str, model_type: str = Models.flash_lite) -> ChatAgent:
    """Create a CAMEL agent for extracting structured team data from HTML content, formatting the prompt with the team.

    Args:
        team (str): The team ID to extract data for.
        model_type (str): The model type to use for extraction. Defaults to "gemini-2.5-flash".
    Returns:
        ChatAgent: The configured team data extraction agent
    """
    platform = api_key_manager._get_platform_from_model(model_type)
    api_keys = api_key_manager._get_api_keys_for_platform(platform)
    if api_keys:
        idx = api_key_manager.current_key_indices.get(platform, 0)
        os.environ[f"{platform.upper()}_API_KEY"] = api_keys[idx]
    model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=model_type,
        model_config_dict={"temperature": 1/3},
    )
    system_prompt = TEAM_DATA_EXTRACTION_PROMPT.format(team=team)
    agent = ChatAgent(
        model=model,
        system_message=system_prompt
    )
    logger.info("Team data extraction agent created successfully")
    return agent

def extract_team_data_with_llm(team: str, html_by_type: dict, meta: dict, save_dir: str = "", save_prefix: str = "") -> dict:
    """Extract structured team data from HTML content for each requested type using the specialized agent.
    If JSON extraction fails, save the raw response to a .txt file if save_dir and save_prefix are provided.
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Log brief summary instead of full content
            total_html_length = sum(len(html) for html in html_by_type.values())
            html_types = list(html_by_type.keys())
            logger.info(f"Starting team data extraction for {team} (HTML types: {html_types}, total length: {total_html_length} chars)")
            
            agent = create_team_data_extraction_agent(team)
            # Compose the prompt
            prompt = f"""
Extract the following data for team: {team}

Meta information:
{json.dumps(meta, ensure_ascii=False, indent=2)}

HTML content for each type:
"""
            for t, html in html_by_type.items():
                prompt += f"\n--- {t.upper()} HTML ---\n{html}\n"
            prompt += "\nPlease return the extracted data in the required JSON format."
            human_message = BaseMessage.make_user_message(
                role_name="Human",
                content=prompt
            )
            response = agent.step(human_message)
            agent_response = response.msgs[0].content if response.msgs else ""
            
            try:
                result = extract_json_from_response(agent_response)
                # Sort historical matches by date if present
                if "historical" in result and isinstance(result["historical"], list):
                    from datetime import datetime
                    def parse_date_safe(d):
                        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):  # try common formats
                            try:
                                return datetime.strptime(d, fmt)
                            except Exception:
                                continue
                        try:
                            return datetime.fromisoformat(d)
                        except Exception:
                            return d  # fallback: string sort
                    result["historical"] = sorted(
                        result["historical"],
                        key=lambda m: parse_date_safe(m.get("date", ""))
                    )
                return result
            except Exception as e:
                logger.error(f"Failed to extract JSON: {e}")
                if save_dir and save_prefix:
                    raw_path = os.path.join(save_dir, save_prefix + "_raw.txt")
                    with open(raw_path, "w", encoding="utf-8") as f:
                        f.write(agent_response)
                    print(f"\033[93m⚠ Failed to extract JSON, saved raw agent response to {raw_path}\033[0m")
                raise
                
        except Exception as e:
            error_str = str(e).lower()
            if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str:
                logger.warning(f"Rate limit ({retry_count + 1}/{max_retries}): {e}")
                
                # Record the rate limit error and rotate API key
                api_key_manager.record_rate_limit_error(Models.flash_lite)
                
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying with new API key...")
                    time.sleep(2)  # Brief pause before retry
                    continue
                else:
                    logger.error(f"Max retries reached for rate limit errors")
                    raise
            else:
                # Non-rate-limit error, don't retry
                logger.error(f"Error in team data extraction: {e}")
                raise
    
    # This should never be reached, but return empty dict to satisfy type checker
    return {}

def extract_team_h2h(site_name: str, team: str, force_fetch: bool = False, args=None) -> str:
    """Extract the h2h summary page for a team, if the site provides a static link."""
    paths = SITE_URLS.get(site_name, {}).get("paths", {})
    pattern = paths.get("h2h")
    if not pattern:
        raise ValueError(f"No h2h pattern for site {site_name}")
    sub_path = fill_url_pattern(pattern, args)
    url = urljoin(SITE_URLS[site_name]["url"], sub_path)
    cache_path = get_team_html_cache_path(site_name, team, "h2h")
    if not force_fetch and is_team_html_cache_valid(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    html_content = extract_html_from_url(url)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return html_content

# --- Caching for extract_team_h2h_vs ---
def extract_team_h2h_vs(site_name: str, team: str, vs_team: str, force_fetch: bool = False, args=None) -> str:
    paths = SITE_URLS.get(site_name, {}).get("paths", {})
    pattern = paths.get("h2h-vs")
    if not pattern:
        raise ValueError(f"No h2h-vs pattern for site {site_name}")
    sub_path = fill_url_pattern(pattern, args)
    url = urljoin(SITE_URLS[site_name]["url"], sub_path)
    cache_path = get_team_html_cache_path(site_name, team, "h2h-vs", vs_team=vs_team)
    if not force_fetch and is_team_html_cache_valid(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    html_content = extract_html_from_url(url)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return html_content

# Utility to compute aggregate stats for h2h-vs filtered matches

def compute_h2h_aggregate(matches):
    agg = {
        "matches": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0
    }
    for m in matches:
        agg["matches"] += 1
        result = m.get("result", "").lower()
        if result == "win":
            agg["wins"] += 1
        elif result == "draw":
            agg["draws"] += 1
        elif result == "loss":
            agg["losses"] += 1
        # Parse score, e.g. "2:1"
        score = m.get("score", "")
        if ":" in score:
            parts = score.split(":")
            try:
                goals_for = int(parts[0].strip())
                goals_against = int(parts[1].strip())
                agg["goals_for"] += goals_for
                agg["goals_against"] += goals_against
            except Exception:
                pass
    return agg

def extract_competition_data_with_llm(html_content: str, site_name: str, competition: str) -> dict:
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Log brief summary instead of full content
            html_length = len(html_content)
            logger.info(f"Starting competition data extraction for {competition} on {site_name} (HTML length: {html_length} chars)")
            
            agent = ChatAgent(
                model=ModelFactory.create(
                    model_platform=ModelPlatformType.GEMINI,
                    model_type=Models.flash_lite
                ),
                system_message=COMPETITION_DATA_EXTRACTION_PROMPT.format(competition=competition)
            )
            prompt = f"Extract all teams statistics for competition: {competition} from the following HTML:\n{html_content}"
            human_message = BaseMessage.make_user_message(role_name="Human", content=prompt)
            response = agent.step(human_message)
            agent_response = response.msgs[0].content if response.msgs else ""
            return extract_json_from_response(agent_response)
            
        except Exception as e:
            error_str = str(e).lower()
            if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str:
                logger.warning(f"Rate limit ({retry_count + 1}/{max_retries}): {e}")
                
                # Record the rate limit error and rotate API key
                api_key_manager.record_rate_limit_error(Models.flash_lite)
                
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying with new API key...")
                    time.sleep(2)  # Brief pause before retry
                    continue
                else:
                    logger.error(f"Max retries reached for rate limit errors")
                    print(f"\033[91m✗ Rate limit exceeded after {max_retries} retries\033[0m")
                    return {"error": f"Rate limit exceeded after {max_retries} retries"}
            else:
                # Non-rate-limit error, don't retry
                logger.error(f"Error in competition data extraction: {e}")
                return {"error": f"Extraction failed: {str(e)}"}
    
    # This should never be reached, but return error to satisfy type checker
    return {"error": "Unexpected error in competition data extraction"}

def extract_team_stats(site_name: str, team: str, group: str = "", competition: str = "", force_fetch: bool = False, args=None) -> str:
    site_name = site_name or ""
    team = team or ""
    group = group or ""
    competition = competition or ""
    if site_name not in SITE_URLS or not SITE_URLS[site_name].get("url"):
        raise ValueError(f"Invalid or missing site_name '{site_name}' in SITE_URLS")
    base_url = SITE_URLS[site_name]["url"]
    paths = SITE_URLS.get(site_name, {}).get("paths", {})
    pattern = paths.get("team-stats")
    cache_path = get_team_html_cache_path(site_name, team, "team-stats", competition=competition)
    if not force_fetch and is_team_html_cache_valid(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    sub_path = fill_url_pattern(pattern, args)
    url = urljoin(base_url, sub_path)
    html_content = extract_html_from_url(url)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return html_content

# --- Caching for extract_team_outrights ---
def extract_team_outrights(site_name: str, team: str, group: str, competition: str, force_fetch: bool = False, args=None) -> str:
    paths = SITE_URLS.get(site_name, {}).get("paths", {})
    pattern = paths.get("outrights")
    if not pattern:
        raise ValueError(f"No outrights pattern for site {site_name}")
    if args is not None:
        if getattr(args, 'group', None) is None:
            setattr(args, 'group', group)
        if getattr(args, 'competition', None) is None:
            setattr(args, 'competition', competition)
    sub_path = fill_url_pattern(pattern, args)
    url = urljoin(SITE_URLS[site_name]["url"], sub_path)
    cache_path = get_team_html_cache_path(site_name, team, "outrights", competition=competition)
    if not force_fetch and is_team_html_cache_valid(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    html_content = extract_html_from_url(url)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return html_content

# --- Caching for extract_team_odds_upcoming ---
def extract_team_odds_upcoming(site_name: str, team: str, group: str, competition: str, force_fetch: bool = False, args=None) -> str:
    paths = SITE_URLS.get(site_name, {}).get("paths", {})
    pattern = paths.get("odds")
    if not pattern:
        raise ValueError(f"No odds pattern for site {site_name}")
    if args is not None:
        if getattr(args, 'group', None) is None:
            setattr(args, 'group', group)
        if getattr(args, 'competition', None) is None:
            setattr(args, 'competition', competition)
    sub_path = fill_url_pattern(pattern, args)
    url = urljoin(SITE_URLS[site_name]["url"], sub_path)
    cache_path = get_team_html_cache_path(site_name, team, "odds", competition=competition)
    if not force_fetch and is_team_html_cache_valid(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    html_content = extract_html_from_url(url)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return html_content

def extract_team_odds_historical(site_name: str, team: str, group: str, competition: str, year: str, page: str = "1", force_fetch: bool = False, args=None) -> str:
    paths = SITE_URLS.get(site_name, {}).get("paths", {})
    pattern = paths.get("odds-historical")
    if not pattern:
        raise ValueError(f"No odds-historical pattern for site {site_name}")
    # Ensure year_prev is set if needed
    if pattern and '{year_prev}' in pattern:
        if args and getattr(args, 'year_prev', None) is None and getattr(args, 'year', None) is not None:
            try:
                setattr(args, 'year_prev', str(int(args.year) - 1))
            except Exception:
                pass
    if args and getattr(args, 'page', None) is None:
        setattr(args, 'page', page)
    sub_path = fill_url_pattern(pattern, args)
    url = urljoin(SITE_URLS[site_name]["url"], sub_path)
    cache_path = get_team_html_cache_path(site_name, team, "odds-historical", year=year, competition=competition)
    if not force_fetch and is_team_html_cache_valid(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    html_content = extract_html_from_url(url)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return html_content

def fill_url_pattern(pattern, args):
    # Find all {placeholders} in the pattern
    placeholders = re.findall(r"{(.*?)}", pattern)
    # Build a dict of replacements from args
    replacements = {}
    missing = []
    for key in placeholders:
        value = getattr(args, key, None)
        if value is not None:
            replacements[key] = value
        else:
            missing.append(key)
    if missing:
        raise ValueError(f"Missing required argument(s) for pattern: {', '.join(missing)}")
    # Replace only the found placeholders
    for key, value in replacements.items():
        pattern = pattern.replace(f"{{{key}}}", str(value))
    return pattern

def _parse_args():
    parser = argparse.ArgumentParser(
        description="""
Football Web Scraper & Data Extractor
====================================

A unified CLI tool for scraping, extracting, and structuring football data from multiple football-related websites. Supports LLM-powered extraction of competitions, teams, and detailed team data (historical, squad, news, appearances, h2h, h2h-vs). Includes cache management, logging, and flexible batch workflows.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Capabilities:
- List all supported football sites and their scraping capabilities
- Scrape and cache HTML content from football sites
- Extract and structure competition lists for a given group or sub-URL using LLMs
- Extract and structure team lists for a given competition using LLMs
- Extract detailed team data (historical matches, squad, news, appearances, h2h, h2h-vs) for a given team, year, and competition
- List cached competitions and teams for quick lookup
- Clear cache for all or specific sites
- Fine-grained cache control (duration, force fetch, etc.)
- Logging to file or console
- Flexible CLI for advanced workflows (e.g., batch extraction, filtering, etc.)

Examples:
  python scrape.py --list
  python scrape.py --site worldfootball --extract-competitions --group Italy --sub-url ita-serie-a
  python scrape.py --site worldfootball --extract-teams --group Italy --sub-url ita-serie-a --competition ita-serie-a
  python scrape.py --site worldfootball --extract-team-data all --team ac-milan --year 2023 --competition ita-serie-a
  python scrape.py --site worldfootball --extract-team-data historical,news --team ac-milan --year 2023
  python scrape.py --site worldfootball --extract-team-data appearances --team ac-milan --competition ita-serie-a
  python scrape.py --site worldfootball --extract-team-data squad --team ac-milan --year 2023
  python scrape.py --site worldfootball --extract-team-data h2h --team ac-milan
  python scrape.py --site worldfootball --extract-team-data h2h-vs --team ac-milan --vs-team inter --date-from 2021 --date-to 2024
  python scrape.py --clear-cache
  python scrape.py --site worldfootball --clear-cache
  python scrape.py --site worldfootball --enable-file-logging

For more details, see the README or function docstrings.
        """
    )
    parser.add_argument(
        "--site",
        type=str,
        help="Name of the site to scrape (use 'all' for all sites, '--list' to see available sites). Supports all major football data sites configured in SITE_URLS."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available sites with their scraping and extraction capabilities."
    )
    parser.add_argument(
        "--cache-days",
        type=int,
        default=None,
        help="Override cache duration in days (0 = no caching, default = use site setting). Controls how long HTML and extracted data are cached."
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cache files before scraping. Use with --site to clear cache for a specific site."
    )
    parser.add_argument(
        "--enable-file-logging",
        action="store_true",
        help="Enable logging to file (logs/web_scraper.log) in addition to console output. Useful for debugging and batch runs."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO). Controls verbosity of log output."
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Group (nation or continent) to filter competitions or teams by (required with --extract-competitions or --extract-teams)."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path key or sub-URL to scrape (required with --group). Used to target a specific competition, league, or team page. If it does not start with a slash, it is treated as a key in the site's paths group."
    )
    parser.add_argument(
        "--competition",
        type=str,
        default=None,
        help="Competition ID (slug, URL, or identifier) to extract teams for. Required with --extract-teams and for some team data extractions."
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Force fetch fresh data and skip all cache (HTML, competitions, teams, team data). Useful for re-scraping or debugging."
    )
    parser.add_argument(
        "--team",
        type=str,
        default=None,
        help="Team ID or slug for team-specific fetches."
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Year for historical or squad extraction (default: latest year). Used for team data extraction."
    )
    parser.add_argument(
        "--vs-team",
        type=str,
        default=None,
        help="Opponent team ID or slug for h2h-vs extraction."
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD or year) for h2h-vs extraction (optional, used for filtering matches)."
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD or year) for h2h-vs extraction (optional, defaults to today)."
    )
    parser.add_argument(
        "--list-competitions",
        action="store_true",
        help="List all competitions from cache for the given site (or all sites if not specified). Useful for quickly finding competition IDs."
    )
    parser.add_argument(
        "--list-teams",
        action="store_true",
        help="List all teams from cache for the given site and competition (or all if not specified). Useful for quickly finding team IDs."
    )
    parser.add_argument(
        "--list-sites",
        action="store_true",
        help="Print the SITE_URLS dictionary as formatted JSON and exit."
    )
    parser.add_argument(
        "--extract",
        type=str,
        default=None,
        help="Comma-separated list of extraction types: competition-teams, competition-stats, historical, news, appearances, squad, h2h, h2h-vs, team-stats, etc. Chooses the system prompt and extraction logic based on the value(s)."
    )
    return parser.parse_args()


def _setup_logging(enable_file_logging):
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "web_scraper.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(GLOBAL_LOG_LEVEL)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    if enable_file_logging:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(GLOBAL_LOG_LEVEL)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(file_handler)
        logger.info("File logging enabled")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(GLOBAL_LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(console_handler)
    
    # Silence camel-related log messages by default
    # Only enable them if global logging level is DEBUG
    camel_logger = logging.getLogger("camel")
    if GLOBAL_LOG_LEVEL == logging.DEBUG:
        camel_logger.setLevel(logging.DEBUG)
    else:
        camel_logger.setLevel(logging.WARNING)
    
    # Also silence specific camel submodules
    camel_agents_logger = logging.getLogger("camel.agents")
    camel_agents_chat_agent_logger = logging.getLogger("camel.agents.chat_agent")
    if GLOBAL_LOG_LEVEL == logging.DEBUG:
        camel_agents_logger.setLevel(logging.DEBUG)
        camel_agents_chat_agent_logger.setLevel(logging.DEBUG)
    else:
        camel_agents_logger.setLevel(logging.WARNING)
        camel_agents_chat_agent_logger.setLevel(logging.WARNING)


def _handle_list_sites():
    # ... existing code for --list-sites ...
    ICON_SITE = "\u26BD"
    ICON_LABEL = "\U0001F3F7"
    ICON_URL = "\U0001F310"
    ICON_CACHE = "\U0001F4C5"
    COLOR_SITE = "\033[96m"
    COLOR_RESET = "\033[0m"
    COLOR_URL = "\033[94m"
    COLOR_DESC = "\033[92m"
    COLOR_CACHE = "\033[93m"
    COLOR_PATTERN = "\033[95m"
    for name, info in SITE_URLS.items():
        print(f"{COLOR_SITE}{ICON_SITE} {name}{COLOR_RESET}")
        print(f"  {COLOR_DESC}{info.get('description', '')}{COLOR_RESET}")
        print(f"  {COLOR_URL}{ICON_URL} {info.get('url', '')}{COLOR_RESET}")
        cache_days_obj = info.get('cache_days', {'default': 1})
        cache_days = cache_days_obj.get('default', 1)
        print(f"  {COLOR_CACHE}{ICON_CACHE} Cache: {cache_days} day{'s' if cache_days != 1 else ''}{COLOR_RESET}")
        if 'competition' in info:
            print(f"  Competitions:")
            for c in info['competition']:
                print(f"    {COLOR_PATTERN}{ICON_LABEL}  {c}{COLOR_RESET}")
        if 'teams' in info:
            print(f"  Teams:")
            for k, v in info['teams'].items():
                print(f"    {COLOR_PATTERN}{ICON_LABEL}  {k}: {v}{COLOR_RESET}")
        print()
    sys.exit(0)


def _handle_list_competitions(args):
    def print_competitions(data, site_name):
        competitions = data.get("competitions", [])
        if not competitions:
            print(f"\033[93mNo competitions found for {site_name}\033[0m")
            return
        print(f"\033[94mCompetitions for {site_name}:\033[0m")
        for comp in competitions:
            url = comp.get("url", "")
            slug = url.rstrip("/").split("/")[-1] if url else comp.get("name", "")
            name = comp.get("name", slug)
            print(f"{slug} ({name})")
    if args.site and args.site.lower() != "all":
        found, _, _, _ = get_site_info(args.site)
        if not found:
            print(f"\033[91mError: Site '{args.site}' not found.\033[0m")
            print("\033[93mUse --list to see available sites.\033[0m")
            sys.exit(1)
        data = load_competitions_cache(args.site)
        if data:
            print_competitions(data, args.site)
        else:
            print(f"\033[93mNo competitions cache found for {args.site}\033[0m")
    else:
        for site_name in SITE_URLS:
            data = load_competitions_cache(site_name)
            if data:
                print_competitions(data, site_name)
            else:
                print(f"\033[93mNo competitions cache found for {site_name}\033[0m")
    sys.exit(0)


def _handle_list_teams(args):
    def print_teams(data, site_name):
        teams = data.get("teams", [])
        if not teams:
            print(f"\033[93mNo teams found for {site_name}\033[0m")
            return
        print(f"\033[94mTeams for {site_name}:\033[0m")
        for team in teams:
            url = team.get("url", "")
            slug = url.rstrip("/").split("/")[-1] if url else team.get("name", "")
            name = team.get("name", slug)
            print(f"{slug} ({name})")
    if args.site and args.site.lower() != "all":
        found, _, _, _ = get_site_info(args.site)
        if not found:
            print(f"\033[91mError: Site '{args.site}' not found.\033[0m")
            print("\033[93mUse --list to see available sites.\033[0m")
            sys.exit(1)
        if args.competition:
            data = load_teams_cache(args.site, args.competition)
            if data:
                print_teams(data, args.site)
            else:
                print(f"\033[93mNo teams cache found for {args.site} (competition: {args.competition})\033[0m")
        else:
            cache_dir = get_cache_dir(args.site)
            found_any = False
            for f in cache_dir.glob("teams_*.json"):
                comp = f.name[len("teams_"):-len(".json")]
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                print(f"\033[96m[Competition: {comp}]\033[0m")
                print_teams(data, args.site)
                found_any = True
            if not found_any:
                print(f"\033[93mNo teams cache found for {args.site}\033[0m")
    else:
        for site_name in SITE_URLS:
            cache_dir = get_cache_dir(site_name)
            found_any = False
            for f in cache_dir.glob("teams_*.json"):
                comp = f.name[len("teams_"):-len(".json")]
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                print(f"\033[96m[Site: {site_name} | Competition: {comp}]\033[0m")
                print_teams(data, site_name)
                found_any = True
            if not found_any:
                print(f"\033[93mNo teams cache found for {site_name}\033[0m")
    sys.exit(0)


def _handle_clear_cache(args):
    if args.site and args.site.lower() != "all":
        found, _, _, _ = get_site_info(args.site)
        if found:
            clear_cache(args.site)
        else:
            print(f"\033[91mError: Site '{args.site}' not found.\033[0m")
            print("\033[93mUse --list to see available sites.\033[0m")
    else:
        clear_cache()
    return




def handle_extract_competitions(args, data_dir, filename):
    if not args.site:
        print("\033[91mError: --site is required for competition extraction.\033[0m")
        return
    site_info = SITE_URLS[args.site]
    group = args.group
    path = args.path
    force_fetch = args.force_fetch
    competition_data = extract_all_competitions(args.site, site_info, group=group, path=path, force_fetch=force_fetch)
    if "error" in competition_data:
        print(f"\033[93m⚠ Competition extraction failed: {competition_data['error']}\033[0m")
    else:
        save_competitions_cache(args.site, competition_data, group, path)
        comp_file_path = save_competitions_to_file(competition_data, args.site)
        if comp_file_path.startswith("Error"):
            print(f"\033[93m⚠ Failed to save competition data: {comp_file_path}\033[0m")
        else:
            total_competitions = competition_data.get('summary', {}).get('total_competitions', 0)
            print(f"\033[92m✓ Competition data extracted and saved to: {comp_file_path}\033[0m")
            print(f"\033[92m✓ Found {total_competitions} competitions\033[0m")
            categories = competition_data.get('summary', {}).get('categories', {})
            if categories:
                print(f"\033[94mCompetition categories:\033[0m")
                for category, count in categories.items():
                    if count > 0:
                        print(f"  - {category}: {count}")
            display_competitions(competition_data, args.site)
    return


def _handle_extract(args):
    extract_types = [x.strip().lower() for x in args.extract.split(",") if x.strip()] if args.extract else []
    current_year = str(datetime.datetime.now().year)
    today_str = datetime.datetime.now().strftime('%Y%m%d')
    data_dir = os.path.join(base_dir, "data", today_str)
    os.makedirs(data_dir, exist_ok=True)
    fname_parts = [args.site]
    if args.competition:
        fname_parts.append(args.competition)
    if args.team:
        fname_parts.append(args.team)
    if args.year:
        fname_parts.append(args.year)
        args.year_prev = int(args.year) - 1
    filename = "_".join(fname_parts)
    meta = {"site": args.site, "competition": args.competition, "team": args.team, "year": args.year}
    if "competitions" in extract_types:
        handle_extract_competitions(args, data_dir, filename)
    if "competition-teams" in extract_types:
        handle_competition_teams(args, data_dir, filename)
    if "competition-stats" in extract_types:
        handle_competition_stats(args, data_dir, filename)
    team_data_types = {"historical", "news", "appearances", "squad", "h2h", "h2h-vs", "team-stats", "stats", "odds-historical", "odds", "outrights", "odds-match"}
    requested_team_data = [t for t in extract_types if t in team_data_types]
    if requested_team_data:
        handle_team_data(args, data_dir, filename, requested_team_data, meta)
    return


def _handle_scrape_all_sites(args, extract_competitions, extract_teams):
    print(f"\033[94mScraping all {len(SITE_URLS)} available sites...\033[0m")
    if args.cache_days is not None:
        if args.cache_days == 0:
            print(f"\033[94mCache override: Disabled for all sites\033[0m")
        else:
            print(f"\033[94mCache override: {args.cache_days} day{'s' if args.cache_days != 1 else ''} for all sites\033[0m")
    successful_scrapes = 0
    for site_name, info in SITE_URLS.items():
        print(f"\n\033[94m{'='*50}\033[0m")
        cache_days_obj = info["cache_days"].copy()
        if args.cache_days is not None:
            cache_days_obj["default"] = args.cache_days
            cache_days_obj["competition"] = args.cache_days
        success = scrape_site(site_name, info["url"], info["description"], cache_days_obj, extract_competitions=extract_competitions, group=args.group, path=args.path, extract_teams=extract_teams, competition=args.competition, force_fetch=args.force_fetch)
        if success:
            successful_scrapes += 1
        print(f"\033[94m{'='*50}\033[0m")
    print(f"\n\033[92mScraping completed! {successful_scrapes}/{len(SITE_URLS)} sites scraped successfully.\033[0m")
    return


def _handle_scrape_single_site(args, extract_competitions, extract_teams):
    found, url, description, cache_days_obj = get_site_info(args.site)
    if not found:
        print(f"\033[91mError: Site '{args.site}' not found.\033[0m")
        print("\033[93mUse --list to see available sites.\033[0m")
        return
    if args.cache_days is not None:
        cache_days_obj["default"] = args.cache_days
        cache_days_obj["competition"] = args.cache_days
        if args.cache_days == 0:
            print(f"\033[94mCache override: Disabled\033[0m")
        else:
            print(f"\033[94mCache override: {args.cache_days} day{'s' if args.cache_days != 1 else ''}\033[0m")
    success = scrape_site(args.site, url, description, cache_days_obj, extract_competitions=extract_competitions, group=args.group, path=args.path, extract_teams=extract_teams, competition=args.competition, force_fetch=args.force_fetch)
    if success:
        print(f"\n\033[92m✓ Scraping completed successfully!\033[0m")
    else:
        print(f"\n\033[91m✗ Scraping failed.\033[0m")
        sys.exit(1)
    return


def handle_competition_teams(args, data_dir, filename):
    if not args.site or not args.competition:
        print("\033[91mError: --site and --competition are required for --extract competition-teams.\033[0m")
        return
    site_info = SITE_URLS[args.site]
    paths_info = site_info["paths"]
    pattern = paths_info.get("competition-teams")
    if not pattern:
        print(f"\033[91mNo competition-teams pattern for site {args.site}\033[0m")
        return
    sub_path = fill_url_pattern(pattern, args)
    url = urljoin(site_info["url"], sub_path)
    html_content = extract_html_from_url(url)
    teams_data = extract_teams_with_llm(html_content, args.site, args.competition)
    save_teams_cache(args.site, teams_data, args.competition)
    display_teams(teams_data, args.site)
    # Save to data dir
    file_path = os.path.join(data_dir, filename + "_competition-teams.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(teams_data, f, ensure_ascii=False, indent=2)
    print(f"\033[92m✓ Saved extracted competition teams to {file_path}\033[0m")


def handle_competition_stats(args, data_dir, filename):
    if not args.site or not args.competition:
        print("\033[91mError: --site and --competition are required for --extract competition-stats.\033[0m")
        return
    site_info = SITE_URLS[args.site]
    paths_info = site_info["paths"]
    pattern = paths_info.get("competition-stats")
    if not pattern:
        print(f"\033[91mNo competition-stats pattern for site {args.site}\033[0m")
        return
    sub_path = fill_url_pattern(pattern, args)
    url = urljoin(site_info["url"], sub_path)
    # Caching logic for competition stats HTML
    cache_dir = get_cache_dir(args.site)
    safe_competition = re.sub(r'[^\w\-_.]', '_', args.competition)
    cache_file = cache_dir / f"competitionstats_{safe_competition}.html"
    cache_ttl_seconds = 3600  # 1 hour
    use_cache = cache_file.exists() and not args.force_fetch
    if use_cache:
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age < cache_ttl_seconds:
            with open(cache_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            print(f"\033[92m✓ Using cached competition stats HTML: {cache_file}\033[0m")
        else:
            use_cache = False
    if not use_cache:
        html_content = extract_html_from_url(url)
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\033[92m✓ Cached competition stats HTML: {cache_file}\033[0m")
    data = extract_competition_data_with_llm(html_content, args.site, args.competition)
    # Save to persistent JSON file in data/<today>/
    file_path = os.path.join(data_dir, filename + "_competition-stats.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\033[92m✓ Saved extracted competition stats to {file_path}\033[0m")


def handle_team_data(args, data_dir, filename, requested_team_data, meta):
    current_year = str(datetime.datetime.now().year)
    for item in requested_team_data:
        cache_path = get_team_data_cache_path(data_dir, filename, item)
        if not args.force_fetch and is_team_data_cache_valid(cache_path):
            print(f"\033[92m✓ Using cached {item} data: {cache_path}\033[0m")
            try:
                result = load_team_data_cache(cache_path)
                print(json.dumps(result, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"\033[91mError loading cached {item} data: {e}\033[0m")
            continue
        html_by_type = {}
        if item == "historical":
            year = args.year if args.year else current_year
            html = extract_team_historical(args.site, args.team, year, args.force_fetch, args=args)
            html_by_type["historical"] = html
        elif item == "news":
            html = extract_team_news(args.site, args.team, args.force_fetch, args=args)
            html_by_type["news"] = html
        elif item == "appearances":
            if not args.competition:
                print("\033[91mError: --competition is required for appearances extraction.\033[0m")
                continue
            year = args.year if args.year else current_year
            html = extract_team_appearances(args.site, args.team, args.competition, year, args.force_fetch, args=args)
            html_by_type["appearances"] = html
        elif item == "squad":
            year = args.year if args.year else current_year
            html = extract_team_squad(args.site, args.team, year, args.force_fetch, args=args)
            html_by_type["squad"] = html
        elif item == "h2h":
            html = extract_team_h2h(args.site, args.team, args.force_fetch, args=args)
            html_by_type["h2h"] = html
        elif item == "h2h-vs":
            if not args.vs_team:
                print("\033[91mError: --vs-team is required for h2h-vs extraction.\033[0m")
                continue
            html = extract_team_h2h_vs(args.site, args.team, args.vs_team, args.force_fetch, args=args)
            html_by_type["h2h-vs"] = html
        elif item == "team-stats" or item == "stats":
            html = extract_team_stats(
                args.site,
                args.team,
                group=args.group or "",
                competition=args.competition or "",
                force_fetch=args.force_fetch,
                args=args
            )
            html_by_type[item] = html
        elif item == "odds-historical":
            # odds-historical requires group, competition, year, and optionally page
            if not args.group or not args.competition or not args.year:
                print("\033[91mError: --group, --competition, and --year are required for odds-historical extraction.\033[0m")
                continue
            page = getattr(args, 'page', "1")
            html = extract_team_odds_historical(
                args.site,
                args.team,
                args.group,
                args.competition,
                args.year,
                page=page,
                force_fetch=args.force_fetch,
                args=args
            )
            html_by_type["odds-historical"] = html
        elif item == "odds":
            # odds requires group and competition
            if not args.group or not args.competition:
                print("\033[91mError: --group and --competition are required for odds extraction.\033[0m")
                continue
            html = extract_team_odds_upcoming(
                args.site,
                args.team,
                args.group,
                args.competition,
                force_fetch=args.force_fetch,
                args=args
            )
            html_by_type["odds"] = html
        elif item == "outrights":
            # outrights requires group and competition
            if not args.group or not args.competition:
                print("\033[91mError: --group and --competition are required for outrights extraction.\033[0m")
                continue
            html = extract_team_outrights(
                args.site,
                args.team,
                args.group,
                args.competition,
                force_fetch=args.force_fetch,
                args=args
            )
            html_by_type["outrights"] = html
        elif item == "odds-match":
            # odds-match requires group, competition, team, vs_team
            if not args.group or not args.competition or not args.team or not args.vs_team:
                print("\033[91mError: --group, --competition, --team, and --vs-team are required for odds-match extraction.\033[0m")
                continue
            html = extract_team_odds_match(
                args.site,
                args.team,
                args.vs_team,
                args.group,
                args.competition,
                force_fetch=args.force_fetch,
                args=args
            )
            html_by_type["odds-match"] = html
        else:
            print(f"\033[91mUnknown extract value: {item}\033[0m")
            continue
        if html_by_type:
            meta_with_url = dict(meta)
            result = extract_team_data_with_llm(args.team, html_by_type, meta_with_url, save_dir=str(data_dir), save_prefix=f"{filename}_{item}")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            save_team_data_cache(cache_path, result)
            print(f"\033[92m✓ Saved extracted {item} data to {cache_path}\033[0m")

def extract_team_odds_match(site_name: str, team: str, vs_team: str, group: str, competition: str, force_fetch: bool = False, args=None) -> str:
    paths = SITE_URLS.get(site_name, {}).get("paths", {})
    pattern = paths.get("odds-match")
    if not pattern:
        raise ValueError(f"No odds-match pattern for site {site_name}")
    if args is not None:
        if getattr(args, 'group', None) is None:
            setattr(args, 'group', group)
        if getattr(args, 'competition', None) is None:
            setattr(args, 'competition', competition)
        if getattr(args, 'team', None) is None:
            setattr(args, 'team', team)
        if getattr(args, 'vs_team', None) is None:
            setattr(args, 'vs_team', vs_team)
    sub_path = fill_url_pattern(pattern, args)
    url = urljoin(SITE_URLS[site_name]["url"], sub_path)
    cache_path = get_team_html_cache_path(site_name, team, "odds-match", competition=competition, vs_team=vs_team, group=group)
    
    # Check if we have cached market data (JSON format)
    json_cache_path = cache_path.with_suffix('.json')
    if not force_fetch and json_cache_path.exists():
        try:
            with open(json_cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                logger.info(f"Using cached market data: {json_cache_path}")
                return json.dumps(cached_data, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Error loading cached market data: {e}")
    
    # Use the Playwright-based LLM agent for dynamic odds market scraping
    try:
        logger.info(f"Using Playwright agent for dynamic odds market scraping: {url}")
        agent = create_scraping_agent(headless=True, model_type=Models.flash_lite)  # Use headless mode for production
        
        # Run the async scraping function - now returns List[Dict[str, Any]]
        collected_markets = asyncio.run(agent.scrape_odds_markets(url, team, vs_team, competition))
        
        # Save the structured market data as JSON
        market_data = {
            'site': site_name,
            'team': team,
            'vs_team': vs_team,
            'competition': competition,
            'group': group,
            'url': url,
            'scraped_at': time.time(),
            'markets': collected_markets,
            'total_markets': len(collected_markets)
        }
        
        # Save as JSON
        with open(json_cache_path, 'w', encoding='utf-8') as f:
            json.dump(market_data, f, ensure_ascii=False, indent=2)
        
        # Also save as HTML for backward compatibility (convert to HTML representation)
        html_content = f"""
        <html>
        <head><title>Odds Markets - {team} vs {vs_team}</title></head>
        <body>
        <h1>Odds Markets: {team} vs {vs_team}</h1>
        <p>Competition: {competition}</p>
        <p>Site: {site_name}</p>
        <p>Total Markets: {len(collected_markets)}</p>
        <div id="markets">
        """
        
        for i, market in enumerate(collected_markets, 1):
            # Ensure market is a dictionary
            if isinstance(market, dict):
                market_name = market.get('market_name', 'Unknown Market')
                market_type = market.get('market_type', 'unknown')
                market_structure = market.get('structure', 'unknown')
            else:
                # Fallback if market is not a dictionary
                market_name = str(market) if market else 'Unknown Market'
                market_type = 'unknown'
                market_structure = 'unknown'
            
            html_content += f"""
            <div class="market" data-market-id="{i}">
                <h3>Market {i}: {market_name}</h3>
                <p><strong>Type:</strong> {market_type}</p>
                <p><strong>Structure:</strong> {market_structure}</p>
            """
            
            if isinstance(market, dict) and market.get('structure') == 'table':
                headers = market.get('headers', [])
                rows = market.get('rows', [])
                if headers:
                    html_content += "<table border='1'><thead><tr>"
                    for header in headers:
                        html_content += f"<th>{header}</th>"
                    html_content += "</tr></thead><tbody>"
                    
                    for row in rows:
                        html_content += "<tr>"
                        for cell in row:
                            html_content += f"<td>{cell}</td>"
                        html_content += "</tr>"
                    
                    html_content += "</tbody></table>"
            
            elif isinstance(market, dict) and market.get('structure') == 'list':
                odds_list = market.get('odds', [])
                if odds_list:
                    html_content += "<table border='1'><thead><tr><th>Condition</th><th>Odds</th><th>Bookmaker</th></tr></thead><tbody>"
                    for odds_item in odds_list:
                        html_content += f"<tr><td>{odds_item.get('condition', 'Unknown')}</td><td>{odds_item.get('odds', 'Unknown')}</td><td>{odds_item.get('bookmaker', 'Unknown')}</td></tr>"
                    html_content += "</tbody></table>"
            
            elif isinstance(market, dict) and market.get('structure') == 'text':
                content = market.get('content', '')
                html_content += f"<p><strong>Content:</strong> {content}</p>"
            
            html_content += "</div>"
        
        html_content += """
        </div>
        </body>
        </html>
        """
        
        # Save HTML version for backward compatibility
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Dynamic odds market scraping completed for {team} vs {vs_team}. Collected {len(collected_markets)} markets.")
        return json.dumps(market_data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Playwright error in dynamic odds market scraping: {str(e)}")
        print(f"\033[91m✗ Playwright error: {str(e)}\033[0m")
        print(f"\033[91mTerminating script due to Playwright error.\033[0m")
        # Terminate the script as requested
        sys.exit(1)

def main():
    args = _parse_args()
    
    # Set global logging level based on command line argument
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    set_global_log_level(log_level_map[args.log_level])
    
    if getattr(args, "list_sites", False):
        _handle_list_sites()
        return
    if args.list_competitions:
        _handle_list_competitions(args)
        return
    if args.list_teams:
        _handle_list_teams(args)
        return
    _setup_logging(args.enable_file_logging)
    if args.list:
        list_available_sites()
        return
    if args.clear_cache:
        _handle_clear_cache(args)
        return
    if not args.site:
        import argparse
        parser = argparse.ArgumentParser()
        parser.print_help()
        print("\n\033[93mNo site specified. Use --list to see available sites or --site <name> to scrape.\033[0m")
        return
    extract_types = [x.strip().lower() for x in args.extract.split(",") if x.strip()] if args.extract else []
    extract_competitions = 'competitions' in extract_types
    extract_teams = 'teams' in extract_types
    if args.extract:
        _handle_extract(args)
        return
    if args.site.lower() == "all":
        _handle_scrape_all_sites(args, extract_competitions, extract_teams)
        return
    _handle_scrape_single_site(args, extract_competitions, extract_teams)


if __name__ == "__main__":
    main()

