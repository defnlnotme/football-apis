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
import time
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Any
import shutil
from site_urls import SITE_URLS
from urllib.parse import urljoin
import datetime
import os

base_dir = pathlib.Path(__file__).parent
env_path = base_dir / ".envrc"
load_dotenv(dotenv_path=str(env_path))

set_log_level(level="DEBUG")

MODEL_TYPES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
]

# Use the root logger for this script
logger = logging.getLogger()

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

def get_competitions_cache_file_path(site_name: str) -> pathlib.Path:
    """Get the competitions cache file path for a given site (JSON)."""
    cache_dir = get_cache_dir(site_name)
    return cache_dir / "competitions.json"

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

# Competition extraction agent system prompt (with {nation} placeholder)
COMPETITION_EXTRACTION_PROMPT = """
You are a specialized football competition data extraction agent. Your task is to analyze raw HTML content from football websites and extract a comprehensive list of all competitions, tournaments, and leagues mentioned on the page.

Your responsibilities include:
1. Identifying all football competitions, tournaments, and leagues mentioned in the content
2. Extracting competition names, types, and relevant details
3. Organizing competitions by category (domestic leagues, international tournaments, cups, etc.)
4. Providing structured data in JSON format
5. Ensuring accuracy and completeness of the extracted information

IMPORTANT: For each competition, you MUST extract the URL that points to the competition's page. The URL is mandatory. If the URL is not directly visible, you must infer it from the context, links, or any available information. Do NOT omit the URL field. If you cannot find a URL, make a best effort to construct it based on the patterns used on the website, and clearly indicate it is inferred.

IMPORTANT: Only include competitions that are associated with the specified nation: {nation}. Ignore and do not return competitions that do not match the given nation. The nation to match will be provided in the extraction context or prompt.

Return ALL leagues, tournaments, and cups for the specified nation. Do not limit the results to a single competition.

When analyzing content, look for:
- League names and abbreviations
- Tournament names and seasons
- Cup competitions
- International competitions
- Regional competitions
- Youth competitions
- Women's competitions

Return ONLY a valid JSON object as your output, with no extra text or explanation.

Example output:
{{
  "competitions": [
    {{
      "name": "Competition name 1",
      "type": "league|tournament|cup|international|regional|youth|womens",
      "nation": "Nation or region",
      "season": "Season if mentioned",
      "url": "URL to the competition page (MANDATORY)",
      "description": "Brief description if available"
    }},
    {{
      "name": "Competition name 2",
      "type": "league|tournament|cup|international|regional|youth|womens",
      "nation": "Nation or region",
      "season": "Season if mentioned",
      "url": "URL to the competition page (MANDATORY)",
      "description": "Brief description if available"
    }}
    // ... more competitions ...
  ],
  "summary": {{
    "total_competitions": 0,
    "categories": {{
      "leagues": 0,
      "tournaments": 0,
      "cups": 0,
      "international": 0,
      "regional": 0,
      "youth": 0,
      "womens": 0
    }}
  }}
}}

If no competitions are found, return:
{{
  "competitions": [],
  "summary": {{
    "total_competitions": 0,
    "categories": {{
      "leagues": 0,
      "tournaments": 0,
      "cups": 0,
      "international": 0,
      "regional": 0,
      "youth": 0,
      "womens": 0
    }}
  }}
}}
"""

def create_competition_extraction_agent(nation: str) -> ChatAgent:
    """Create a CAMEL agent for extracting competition data from HTML content, with nation interpolation.
    
    Args:
        nation (str): The nation to extract competitions for.
    Returns:
        ChatAgent: The configured competition extraction agent
    """
    try:
        # Create a model for the agent
        model = ModelFactory.create(
            model_platform=ModelPlatformType.GEMINI,
            model_type="gemini-2.5-flash"
        )
        
        # Interpolate the nation into the system prompt
        system_prompt = COMPETITION_EXTRACTION_PROMPT.format(nation=nation)
        
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

def extract_competitions_with_llm(html_content: str, site_name: str, nation: Optional[str] = None) -> Dict[str, Any]:
    """Extract competition list from HTML content using a CAMEL agent.
    
    Args:
        html_content (str): The HTML content to analyze
        site_name (str): The name of the site being analyzed
        nation (Optional[str]): Nation to filter competitions by
        
    Returns:
        Dict[str, Any]: Extracted competition data in structured format
    """
    try:
        logger.info(f"Starting competition extraction for {site_name}")
        
        # Create the competition extraction agent with nation interpolation
        agent = create_competition_extraction_agent(nation or "(not specified)")
        
        # Prepare the analysis prompt
        analysis_prompt = f"""
Please analyze the following HTML content from {site_name} and extract all football competitions, tournaments, and leagues mentioned.

HTML Content:
{html_content[:50000]}  # Limit content to avoid token limits

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
            logger.debug(f"Raw response: {agent_response}")
            
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

def save_competitions_to_file(competition_data: Dict[str, Any], site_name: str, sub_url: Optional[str] = None) -> str:
    """Save extracted competition data to a JSON file in the cache folder.
    
    Args:
        competition_data (Dict[str, Any]): The extracted competition data
        site_name (str): The name of the site
        sub_url (Optional[str]): The sub-URL or path for this extraction (for filename)
    Returns:
        str: The path to the saved file
    """
    try:
        # Create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', site_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if sub_url:
            # Replace '/' with '_' and strip leading/trailing underscores
            slug = sub_url.replace('/', '_').strip('_')
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
            model_type="gemini-2.5-flash",
            model_config_dict={"temperature": 0},
        )
        
        # Initialize the DocumentProcessingToolkit
        doc_toolkit = DocumentProcessingToolkit(model=model)
        
        logger.info(f"Extracting HTML from URL: {url}")
        
        # Extract the document content
        success, content = doc_toolkit.extract_document_content(url)
        
        if success:
            logger.info("HTML extraction successful")
            return content
        else:
            logger.error(f"Failed to extract HTML: {content}")
            return f"Error: {content}"
            
    except Exception as e:
        logger.error(f"Exception during HTML extraction: {str(e)}")
        logger.debug(traceback.format_exc())
        return f"Exception: {str(e)}"

def save_html_to_file(html_content: str, site_name: str, sub_url: Optional[str] = None) -> str:
    r"""Save HTML content to a file in the cache/<site>/ folder.

    Args:
        html_content (str): The HTML content to save.
        site_name (str): The name of the site for the filename.
        sub_url (Optional[str]): The sub-URL or path for this extraction (for filename)

    Returns:
        str: The path to the saved file.
    """
    try:
        # Create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', site_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if sub_url:
            slug = sub_url.replace('/', '_').strip('_')
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
    """Display extracted competitions in a nice format.
    
    Args:
        competition_data (Dict[str, Any]): The extracted competition data
        site_name (str): The name of the site
    """
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
            nation = comp.get("nation", "")
            season = comp.get("season", "")
            description = comp.get("description", "")
            
            print(f"  • {name}")
            if nation:
                print(f"    Nation: {nation}")
            if season:
                print(f"    Season: {season}")
            if description:
                print(f"    Description: {description}")
            print()

def is_competitions_cache_valid(site_name: str, cache_days_obj: dict) -> bool:
    cache_days = cache_days_obj.get("competition", cache_days_obj["default"])
    cache_file = get_competitions_cache_file_path(site_name)
    if cache_days == 0:
        return False
    if not cache_file.exists():
        return False
    file_age = time.time() - cache_file.stat().st_mtime
    max_age_seconds = cache_days * 24 * 60 * 60
    return file_age < max_age_seconds

def load_competitions_cache(site_name: str) -> Optional[dict]:
    try:
        cache_file = get_competitions_cache_file_path(site_name)
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load competitions cache for {site_name}: {str(e)}")
    return None

def save_competitions_cache(site_name: str, data: dict) -> bool:
    try:
        cache_file = get_competitions_cache_file_path(site_name)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached competitions for {site_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to cache competitions for {site_name}: {str(e)}")
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

def extract_all_competitions(site_name: str, site_info: dict, nation: Optional[str] = None, sub_url: Optional[str] = None) -> dict:
    """Extract competitions from all URLs in the site's competition list, merging results. If competitions.json exists, load and return its contents instead of fetching again.
    If nation and sub_url are specified, only process and cache that sub_url."""
    import os
    cache_file = get_competitions_cache_file_path(site_name)
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load competitions.json for {site_name}: {str(e)}. Refetching...")
    base_url = site_info["url"]
    comp_urls = site_info.get("competition", [base_url])
    if nation and sub_url:
        # If sub_url is a regex, match all comp_urls that match the pattern
        try:
            regex = re.compile(sub_url)
            matched_urls = [u for u in comp_urls if regex.search(u)]
            if not matched_urls:
                logger.warning(f"No sub-URLs matched the regex: {sub_url}")
            comp_urls = matched_urls
        except re.error as e:
            logger.error(f"Invalid regex for sub_url: {sub_url} ({e})")
            comp_urls = []
    teams_urls = site_info.get("teams", [base_url])  # Added support for 'teams' group
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
        # Resolve relative URLs
        if not comp_url.startswith("http://") and not comp_url.startswith("https://"):
            url = urljoin(base_url, comp_url)
        else:
            url = comp_url
        # Try to find a cached HTML file for this sub-URL
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
            # Save HTML for each sub-url
            save_html_to_file(html_content, site_name, sub_url=comp_url)
        if html_content.startswith("Error:") or html_content.startswith("Exception:"):
            errors.append(f"{url}: {html_content}")
            continue
        comp_data = extract_competitions_with_llm(html_content, site_name, nation=nation)
        # Save each sub-url's competitions file
        save_competitions_to_file(comp_data, site_name, sub_url=comp_url)
        comps = comp_data.get("competitions", [])
        for c in comps:
            key = (c.get("name"), c.get("type"), c.get("nation"), c.get("season"))
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
    # teams_urls is now available for future use
    return result

TEAM_EXTRACTION_PROMPT = """
You are a specialized football team data extraction agent. Your task is to analyze raw HTML content from a football competition page and extract a comprehensive list of all football teams and clubs participating in the specified competition.

Competition ID: {competition_id}

Your responsibilities include:
1. Identifying all football teams, clubs, and national teams participating in the competition (Competition ID: {competition_id})
2. Extracting team names, types, and relevant details
3. Organizing teams by category (club, national, youth, women, etc.)
4. Providing structured data in JSON format
5. Ensuring accuracy and completeness of the extracted information

IMPORTANT: For each team, you MUST extract the URL that points to the team's page. The URL is mandatory. If the URL is not directly visible, you must infer it from the context, links, or any available information. Do NOT omit the URL field. If you cannot find a URL, make a best effort to construct it based on the patterns used on the website, and clearly indicate it is inferred.

IMPORTANT: Only include teams that are confirmed to be participating in the specified competition (Competition ID: {competition_id}). If you are not sure that a team belongs to the specified competition, do NOT include it in the results.

Return ALL clubs, national teams, youth teams, and women's teams participating in the competition. Do not limit the results to a single team.

When analyzing content, look for:
- Club names and abbreviations
- National teams
- Youth teams
- Women's teams
- League or competition associations

Return ONLY a valid JSON object as your output, with no extra text or explanation.

Example output:
{{
  "teams": [
    {{
      "name": "Team name 1",
      "type": "club|national|youth|women",
      "competition": "{competition_id}",
      "url": "URL to the team page (MANDATORY)",
      "description": "Brief description if available"
    }},
    {{
      "name": "Team name 2",
      "type": "club|national|youth|women",
      "competition": "{competition_id}",
      "url": "URL to the team page (MANDATORY)",
      "description": "Brief description if available"
    }}
    // ... more teams ...
  ],
  "summary": {{
    "total_teams": 0,
    "categories": {{
      "club": 0,
      "national": 0,
      "youth": 0,
      "women": 0
    }}
  }}
}}

If no teams are found, return:
{{
  "teams": [],
  "summary": {{
    "total_teams": 0,
    "categories": {{
      "club": 0,
      "national": 0,
      "youth": 0,
      "women": 0
    }}
  }}
}}
"""

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

def create_team_extraction_agent(competition_id: str) -> ChatAgent:
    """Create a CAMEL agent for extracting team data from HTML content, with competition_id interpolation.
    Args:
        competition_id (str): The competition ID to extract teams for.
    Returns:
        ChatAgent: The configured team extraction agent
    """
    try:
        model = ModelFactory.create(
            model_platform=ModelPlatformType.GEMINI,
            model_type="gemini-2.5-flash",
            model_config_dict={"temperature": 1/3},
        )
        system_prompt = TEAM_EXTRACTION_PROMPT.format(competition_id=competition_id)
        agent = ChatAgent(
            model=model,
            system_message=system_prompt
        )
        logger.info("Team extraction agent created successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to create team extraction agent: {str(e)}")
        raise e

def extract_teams_with_llm(html_content: str, site_name: str, competition_id: str) -> Dict[str, Any]:
    """Extract team list from HTML content using a CAMEL agent.
    Args:
        html_content (str): The HTML content to analyze
        site_name (str): The name of the site being analyzed
        competition_id (str): Competition ID to filter teams by
    Returns:
        Dict[str, Any]: Extracted team data in structured format
    """
    try:
        logger.info(f"Starting team extraction for {site_name}")
        agent = create_team_extraction_agent(competition_id)
        analysis_prompt = f"""
Please analyze the following HTML content from {site_name} and extract all football teams and clubs participating in competition: {competition_id}.

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
            nation = team.get("nation", "")
            league = team.get("league", "")
            description = team.get("description", "")
            print(f"  • {name}")
            if nation:
                print(f"    Nation: {nation}")
            if league:
                print(f"    League: {league}")
            if description:
                print(f"    Description: {description}")
            print()

def get_teams_cache_file_path(site_name: str, competition_id: Optional[str] = None) -> pathlib.Path:
    """Get the cache file path for teams for a given site and competition ID."""
    cache_dir = get_cache_dir(site_name)
    if competition_id:
        safe_comp_id = re.sub(r'[^\w\-_.]', '_', competition_id)
        return cache_dir / f"teams_{safe_comp_id}.json"
    return cache_dir / "teams.json"

def is_teams_cache_valid(site_name: str, cache_days_obj: dict, competition_id: Optional[str] = None) -> bool:
    """Check if the teams cache is valid for a given site and competition ID."""
    cache_days = cache_days_obj.get("teams", cache_days_obj["default"])
    cache_file = get_teams_cache_file_path(site_name, competition_id)
    if cache_days == 0:
        return False
    if not cache_file.exists():
        return False
    file_age = time.time() - cache_file.stat().st_mtime
    max_age_seconds = cache_days * 24 * 60 * 60
    return file_age < max_age_seconds

def load_teams_cache(site_name: str, competition_id: Optional[str] = None) -> Optional[dict]:
    """Load teams cache for a given site and competition ID."""
    try:
        cache_file = get_teams_cache_file_path(site_name, competition_id)
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load teams cache for {site_name} (competition_id={competition_id}): {str(e)}")
    return None

def save_teams_cache(site_name: str, data: dict, competition_id: Optional[str] = None) -> bool:
    """Save teams cache for a given site and competition ID."""
    try:
        cache_file = get_teams_cache_file_path(site_name, competition_id)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached teams for {site_name} (competition_id={competition_id})")
        return True
    except Exception as e:
        logger.error(f"Failed to cache teams for {site_name} (competition_id={competition_id}): {str(e)}")
        return False

def scrape_site(site_name: str, url: str, description: str, cache_days_obj: dict = {"default": 1}, extract_competitions: bool = False, nation: Optional[str] = None, sub_url: Optional[str] = None, extract_teams: bool = False, competition_id: Optional[str] = None, force_fetch: bool = False) -> bool:
    """Scrape a specific site and optionally extract competitions or teams. If extracting teams, a competition ID must be provided.
    Args:
        site_name (str): The name of the site.
        url (str): The URL to scrape.
        description (str): Description of the site.
        cache_days_obj (dict): Number of days to cache content (0 = no caching).
        extract_competitions (bool): Whether to extract competitions using LLM.
        nation (Optional[str]): Nation to filter competitions/teams by.
        sub_url (Optional[str]): Sub-URL path to scrape (required with --nation).
        extract_teams (bool): Whether to extract teams using LLM.
        competition_id (Optional[str]): Competition ID (slug, URL, or identifier) to extract teams for. Required with --extract-teams.
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
    if extract_teams:
        print(f"\033[94mTeam extraction: Enabled (competition ID: {competition_id})\033[0m")

    # Only handle competitions or teams, never fetch the homepage by default
    if extract_teams:
        print(f"\033[94mExtracting teams from {site_name} (competition ID: {competition_id})...\033[0m")
        teams_info = SITE_URLS.get(site_name, {}).get("teams", {})
        competition_pattern = teams_info.get("competition")
        if competition_pattern and competition_id:
            sub_path = competition_pattern.replace("{competition}", competition_id)
            team_url = urljoin(url, sub_path)
        else:
            team_url = competition_id if competition_id and competition_id.startswith('http') else urljoin(url, competition_id or "")
        # Check for cached competition HTML page
        cache_dir = get_cache_dir(site_name)
        safe_competition_id = re.sub(r'[^\w\-_.]', '_', competition_id or "competition")
        cache_file = cache_dir / f"competition_{safe_competition_id}.html"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            print(f"\033[92m✓ Using cached competition HTML page: {cache_file}\033[0m")
        else:
            html_content = extract_html_from_url(team_url)
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"\033[92m✓ Cached competition HTML page: {cache_file}\033[0m")
        teams_data = extract_teams_with_llm(html_content, site_name, competition_id or "")
        save_teams_cache(site_name, teams_data, competition_id)
        display_teams(teams_data, site_name)
        return True

    if extract_competitions:
        print(f"\033[94mExtracting competitions from {site_name}...\033[0m")
        competition_data = extract_all_competitions(site_name, SITE_URLS[site_name], nation=nation, sub_url=sub_url)
        if "error" in competition_data:
            print(f"\033[93m⚠ Competition extraction failed: {competition_data['error']}\033[0m")
        else:
            save_competitions_cache(site_name, competition_data)
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

    print(f"\033[93mNo extraction type specified. Use --extract-competitions or --extract-teams.\033[0m")
    return False

def extract_team_historical(site_name: str, team_id: str, year: str) -> str:
    """Extract the historical page for a team for a given year, if the site provides a static link."""
    teams_info = SITE_URLS.get(site_name, {}).get("teams", {})
    pattern = teams_info.get("historical")
    if not pattern:
        raise ValueError(f"No historical pattern for site {site_name}")
    sub_path = pattern.replace("{team}", team_id).replace("{year}", year)
    url = urljoin(SITE_URLS[site_name]["url"], sub_path)
    html_content = extract_html_from_url(url)
    # Optionally cache here if desired
    return html_content

def extract_team_news(site_name: str, team_id: str) -> str:
    """Extract the news page for a team, if the site provides a static link."""
    teams_info = SITE_URLS.get(site_name, {}).get("teams", {})
    pattern = teams_info.get("news")
    if not pattern:
        raise ValueError(f"No news pattern for site {site_name}")
    sub_path = pattern.replace("{team}", team_id)
    url = urljoin(SITE_URLS[site_name]["url"], sub_path)
    html_content = extract_html_from_url(url)
    # Optionally cache here if desired
    return html_content

def extract_team_appearances(site_name: str, team_id: str, competition_id: str, year: str) -> str:
    """Extract the appearances page for a team in a competition, if the site provides a static link."""
    teams_info = SITE_URLS.get(site_name, {}).get("teams", {})
    pattern = teams_info.get("appearances")
    if not pattern:
        raise ValueError(f"No appearances pattern for site {site_name}")
    # Handle {year_prev} and {year} placeholders if present
    if "{year_prev}" in pattern or "{year}" in pattern:
        year_int = int(year)
        year_prev = str(year_int - 1)
        sub_path = pattern.replace("{team}", team_id).replace("{competition}", competition_id).replace("{year_prev}", year_prev).replace("{year}", year)
    else:
        sub_path = pattern.replace("{team}", team_id).replace("{competition}", competition_id)
    url = urljoin(SITE_URLS[site_name]["url"], sub_path)
    html_content = extract_html_from_url(url)
    # Optionally cache here if desired
    return html_content

def extract_team_squad(site_name: str, team_id: str, year: str) -> str:
    """Extract the squad page for a team for a given year, if the site provides a static link."""
    teams_info = SITE_URLS.get(site_name, {}).get("teams", {})
    pattern = teams_info.get("squad")
    if not pattern:
        raise ValueError(f"No squad pattern for site {site_name}")
    sub_path = pattern.replace("{team}", team_id).replace("{year}", year)
    url = urljoin(SITE_URLS[site_name]["url"], sub_path)
    html_content = extract_html_from_url(url)
    # Optionally cache here if desired
    return html_content

TEAM_DATA_EXTRACTION_PROMPT = """
You are a specialized football team data extraction agent. You will be given HTML content for a football team, and this HTML corresponds to EXACTLY ONE of the following data types. You MUST extract and return ONLY the section for that data type, and OMIT all other sections from your output.

The team ID for this extraction is: {team_id}

The possible data types and their meanings are:
- historical: Extract the team's match history for the given season/year. For each match, include the date, opponent, final score, venue (home/away or stadium name), and any other relevant match details (e.g., competition, round, result type).
- squad: Extract the full list of players registered for the team in the given season/year. For each player, include name, position, shirt number, nationality, and any other available details (e.g., date of birth, role/captaincy, appearances, goals).
- news: Extract recent news articles or updates about the team. For each article, include the title, URL, publication date, and a brief summary or excerpt if available. Only include news directly relevant to the team.
- appearances: Extract cumulative player statistics for the given season/year and competition. For each player, include total minutes played, number of matches, goals, assists, yellow cards, red cards, and any other available stats (e.g., substitutions, starts, penalties, clean sheets for goalkeepers).

IMPORTANT: Do NOT include the raw_html field in the output for any category.

For the provided HTML, extract the relevant information for the corresponding data type and organize it in a JSON object as shown below. Only include a section if the corresponding HTML content is provided.

Return ONLY a valid JSON object as your output, with no extra text or explanation.

Example output:
{{
  "team_id": "{team_id}",
  "historical": {{
    "year": "2023",
    "matches": [
      {{"date": "2023-05-01", "opponent": "Inter", "score": "2-1", "place": "San Siro", "competition": "Serie A", "round": "34", "result": "win"}},
      ...
    ]
  }}
}}

or

{{
  "team_id": "{team_id}",
  "squad": {{
    "year": "2023",
    "players": [
      {{"name": "...", "position": "...", "number": "...", "nationality": "...", "dob": "...", "role": "..."}},
      ...
    ]
  }}
}}

or

{{
  "team_id": "{team_id}",
  "news": {{
    "articles": [
      {{"title": "...", "url": "...", "date": "...", "summary": "..."}},
      ...
    ]
  }}
}}

or

{{
  "team_id": "{team_id}",
  "appearances": {{
    "competition": "ita-serie-a",
    "year": "2023",
    "players": [
      {{"name": "...", "time_played": "...", "matches": "...", "goals": "...", "assists": "...", "yellow_cards": "...", "red_cards": "...", "starts": "...", "subs": "...", "penalties": "...", "clean_sheets": "..."}},
      ...
    ]
  }}
}}

If the data cannot be extracted, return an empty object for the relevant section.
"""

def create_team_data_extraction_agent(team_id: str) -> ChatAgent:
    """Create a CAMEL agent for extracting structured team data from HTML content, formatting the prompt with the team_id."""
    model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type="gemini-2.5-flash-lite-preview-06-17",
        model_config_dict={"temperature": 1/3},
    )
    system_prompt = TEAM_DATA_EXTRACTION_PROMPT.format(team_id=team_id)
    agent = ChatAgent(
        model=model,
        system_message=system_prompt
    )
    logger.info("Team data extraction agent created successfully")
    return agent

def extract_team_data_with_llm(team_id: str, html_by_type: dict, meta: dict, save_dir: str = "", save_prefix: str = "") -> dict:
    """Extract structured team data from HTML content for each requested type using the specialized agent.
    If JSON extraction fails, save the raw response to a .txt file if save_dir and save_prefix are provided.
    """
    agent = create_team_data_extraction_agent(team_id)
    # Compose the prompt
    prompt = f"""
Extract the following data for team: {team_id}

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
        return extract_json_from_response(agent_response)
    except Exception as e:
        logger.error(f"Failed to extract JSON: {e}")
        if save_dir and save_prefix:
            raw_path = os.path.join(save_dir, save_prefix + "_raw.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(agent_response)
            print(f"\033[93m⚠ Failed to extract JSON, saved raw agent response to {raw_path}\033[0m")
        raise

def main():
    r"""Main function to scrape websites based on user selection."""
    parser = argparse.ArgumentParser(
        description="Web scraper for various football websites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scrape.py --list                    # List all available sites
  python scrape.py --site worldfootball      # Scrape worldfootball.net
  python scrape.py --site transfermarkt      # Scrape transfermarkt.com
  python scrape.py --site all                # Scrape all available sites
  python scrape.py --site worldfootball --extract-competitions --nation Italy --sub-url ita-serie-a  # Extract competitions for Italy
  python scrape.py --site worldfootball --extract-teams --nation Italy --sub-url ita-serie-a --competition-id ita-serie-a   # Extract teams for a specific competition
  python scrape.py --site worldfootball --extract-team-data all --team-id ac-milan --year 2023 --competition-id ita-serie-a   # Extract all team data for AC Milan
  python scrape.py --site worldfootball --extract-team-data historical,news --team-id ac-milan --year 2023   # Extract historical and news pages for AC Milan in 2023
  python scrape.py --site worldfootball --extract-team-data appearances --team-id ac-milan --competition-id ita-serie-a   # Extract appearances page for AC Milan in Serie A (requires --competition-id)
  python scrape.py --site worldfootball --extract-team-data squad --team-id ac-milan --year 2023         # Extract squad page for AC Milan in 2023
  python scrape.py --clear-cache             # Clear all cache files
  python scrape.py --site worldfootball --clear-cache           # Clear cache for specific site
  python scrape.py --site worldfootball --enable-file-logging   # Scrape with file logging enabled
        """
    )
    
    parser.add_argument(
        "--site", 
        type=str, 
        help="Name of the site to scrape (use 'all' for all sites, '--list' to see available sites)"
    )
    
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List all available sites"
    )
    
    parser.add_argument(
        "--extract-competitions",
        action="store_true",
        help="Extract competition data using LLM after scraping"
    )
    
    parser.add_argument(
        "--extract-teams",
        action="store_true",
        help="Extract football team data using LLM after scraping. Requires --competition-id."
    )
    
    parser.add_argument(
        "--cache-days",
        type=int,
        default=None,
        help="Override cache duration in days (0 = no caching, default = use site setting)"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cache files before scraping"
    )
    
    parser.add_argument(
        "--enable-file-logging",
        action="store_true",
        help="Enable logging to file (web_scraper.log)"
    )
    
    parser.add_argument(
        "--nation",
        type=str,
        default=None,
        help="Nation to filter competitions by (required with --extract-competitions)"
    )
    
    parser.add_argument(
        "--sub-url",
        type=str,
        default=None,
        help="Sub-URL path to scrape (required with --nation)"
    )
    
    parser.add_argument(
        "--competition-id",
        type=str,
        default=None,
        help="Competition ID (slug, URL, or identifier) to extract teams for. Required with --extract-teams."
    )
    
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Force fetch fresh data and skip all cache (HTML, competitions, teams)"
    )

    # New CLI flags for team data pages
    parser.add_argument(
        "--extract-team-data",
        type=str,
        default=None,
        help="Comma-separated list of team data to extract: historical,news,appearances,squad, or 'all' for all. Requires --site, --team-id, and --year for historical/squad. --competition-id is only required if 'appearances' is included."
    )
    parser.add_argument(
        "--team-id",
        type=str,
        default=None,
        help="Team ID or slug for team-specific fetches"
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Year for historical or squad extraction (default: latest year)"
    )

    args = parser.parse_args()
    
    # Set up logging based on arguments
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "web_scraper.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove all handlers associated with the root logger object (if any)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Enable file logging if requested
    if args.enable_file_logging:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(file_handler)
        logger.info("File logging enabled")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    root_logger.addHandler(console_handler)
    
    # Handle --list argument
    if args.list:
        list_available_sites()
        return
    
    # Handle --clear-cache argument
    if args.clear_cache:
        if args.site and args.site.lower() != "all":
            # Clear cache for specific site
            found, _, _, _ = get_site_info(args.site)
            if found:
                clear_cache(args.site)
            else:
                print(f"\033[91mError: Site '{args.site}' not found.\033[0m")
                print("\033[93mUse --list to see available sites.\033[0m")
        else:
            # Clear all cache files
            clear_cache()
        return
    
    # If no arguments provided, show help
    if not args.site:
        parser.print_help()
        print("\n\033[93mNo site specified. Use --list to see available sites or --site <name> to scrape.\033[0m")
        return
    
    # If --extract-competitions is used, --nation must be provided
    if args.extract_competitions and not args.nation:
        print("\033[91mError: --nation is required when using --extract-competitions.\033[0m")
        sys.exit(1)
    if args.extract_teams and not args.nation:
        print("\033[91mError: --nation is required when using --extract-teams.\033[0m")
        sys.exit(1)
    if args.extract_teams and not args.competition_id:
        print("\033[91mError: --competition-id is required when using --extract-teams.\033[0m")
        sys.exit(1)
    if args.nation and not args.sub_url:
        print("\033[91mError: --sub-url is required when using --nation.\033[0m")
        sys.exit(1)
    
    # Handle new team data fetch flags
    if args.extract_team_data:
        if not args.site or not args.team_id:
            print("\033[91mError: --site and --team-id are required for --extract-team-data.\033[0m")
            return
        requested = [x.strip().lower() for x in args.extract_team_data.split(",") if x.strip()]
        if len(requested) == 1 and requested[0] == "all":
            requested = ["historical", "news", "appearances", "squad"]
        current_year = str(datetime.datetime.now().year)
        today_str = datetime.datetime.now().strftime('%Y%m%d')
        data_dir = os.path.join(base_dir, "data", today_str)
        os.makedirs(data_dir, exist_ok=True)
        fname_parts = [args.site, args.team_id]
        if args.year:
            fname_parts.append(args.year)
        if args.competition_id:
            fname_parts.append(args.competition_id)
        filename = "_".join(fname_parts) + "_teamdata"
        meta = {"site": args.site, "team_id": args.team_id, "year": args.year or current_year, "competition_id": args.competition_id or None}
        for item in requested:
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
            fetch_url = None
            if item == "historical":
                year = args.year if args.year else current_year
                if not args.year:
                    print(f"\033[93mNo --year provided for historical extraction, using latest year: {year}\033[0m")
                fetch_url = SITE_URLS[args.site]["url"]
                teams_info = SITE_URLS.get(args.site, {}).get("teams", {})
                pattern = teams_info.get("historical")
                if pattern:
                    sub_path = pattern.replace("{team}", args.team_id).replace("{year}", year)
                    fetch_url = urljoin(SITE_URLS[args.site]["url"], sub_path)
                html = extract_team_historical(args.site, args.team_id, year)
                html_by_type["historical"] = html
            elif item == "news":
                fetch_url = SITE_URLS[args.site]["url"]
                teams_info = SITE_URLS.get(args.site, {}).get("teams", {})
                pattern = teams_info.get("news")
                if pattern:
                    sub_path = pattern.replace("{team}", args.team_id)
                    fetch_url = urljoin(SITE_URLS[args.site]["url"], sub_path)
                html = extract_team_news(args.site, args.team_id)
                html_by_type["news"] = html
            elif item == "appearances":
                if not args.competition_id:
                    print("\033[91mError: --competition-id is required for appearances extraction.\033[0m")
                    continue
                year = args.year if args.year else current_year
                if not args.year:
                    print(f"\033[93mNo --year provided for appearances extraction, using latest year: {year}\033[0m")
                fetch_url = SITE_URLS[args.site]["url"]
                teams_info = SITE_URLS.get(args.site, {}).get("teams", {})
                pattern = teams_info.get("appearances")
                if pattern:
                    year_int = int(year)
                    year_prev = str(year_int - 1)
                    sub_path = pattern.replace("{team}", args.team_id).replace("{competition}", args.competition_id).replace("{year_prev}", year_prev).replace("{year}", year)
                    fetch_url = urljoin(SITE_URLS[args.site]["url"], sub_path)
                html = extract_team_appearances(args.site, args.team_id, args.competition_id, year)
                html_by_type["appearances"] = html
            elif item == "squad":
                year = args.year if args.year else current_year
                if not args.year:
                    print(f"\033[93mNo --year provided for squad extraction, using latest year: {year}\033[0m")
                fetch_url = SITE_URLS[args.site]["url"]
                teams_info = SITE_URLS.get(args.site, {}).get("teams", {})
                pattern = teams_info.get("squad")
                if pattern:
                    sub_path = pattern.replace("{team}", args.team_id).replace("{year}", year)
                    fetch_url = urljoin(SITE_URLS[args.site]["url"], sub_path)
                html = extract_team_squad(args.site, args.team_id, year)
                html_by_type["squad"] = html
            else:
                print(f"\033[91mUnknown extract-team-data value: {item}\033[0m")
                continue
            if html_by_type:
                try:
                    meta_with_url = dict(meta)
                    meta_with_url["fetch_url"] = fetch_url
                    result = extract_team_data_with_llm(args.team_id, html_by_type, meta_with_url, save_dir=str(data_dir), save_prefix=f"{filename}_{item}")
                    # Add fetch_url to the result for traceability
                    result["fetch_url"] = fetch_url
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    save_team_data_cache(cache_path, result)
                    print(f"\033[92m✓ Saved extracted {item} data to {cache_path}\033[0m")
                except Exception:
                    # Error and raw already handled in extract_team_data_with_llm
                    pass
        return
    
    # Handle scraping all sites
    if args.site.lower() == "all":
        print(f"\033[94mScraping all {len(SITE_URLS)} available sites...\033[0m")
        if args.extract_competitions:
            print(f"\033[94mCompetition extraction enabled for all sites\033[0m")
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
            success = scrape_site(site_name, info["url"], info["description"], cache_days_obj, args.extract_competitions, nation=args.nation, sub_url=args.sub_url, extract_teams=args.extract_teams, competition_id=args.competition_id, force_fetch=args.force_fetch)
            if success:
                successful_scrapes += 1
            print(f"\033[94m{'='*50}\033[0m")
        print(f"\n\033[92mScraping completed! {successful_scrapes}/{len(SITE_URLS)} sites scraped successfully.\033[0m")
        return
    
    # Handle single site scraping
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
    success = scrape_site(args.site, url, description, cache_days_obj, args.extract_competitions, nation=args.nation, sub_url=args.sub_url, extract_teams=args.extract_teams, competition_id=args.competition_id, force_fetch=args.force_fetch)
    if success:
        print(f"\n\033[92m✓ Scraping completed successfully!\033[0m")
    else:
        print(f"\n\033[91m✗ Scraping failed.\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main() 