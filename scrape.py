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

base_dir = pathlib.Path(__file__).parent
env_path = base_dir / ".envrc"
load_dotenv(dotenv_path=str(env_path))

set_log_level(level="DEBUG")

# Use the root logger for this script
logger = logging.getLogger()

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
You are a specialized football competition data extraction agent. Your task is to analyze raw HTML content (as returned by Firecrawl or other scrapers from football websites) and extract a comprehensive list of all competitions, tournaments, and leagues mentioned on the page.

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
            model_type="gemini-2.5-flash", #-lite-preview-06-17",
            model_config_dict={"temperature": 1/3},  # Low temperature for consistent extraction
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

def scrape_site(site_name: str, url: str, description: str, cache_days_obj: dict = {"default": 1}, extract_competitions: bool = False, nation: Optional[str] = None, sub_url: Optional[str] = None) -> bool:
    """Scrape a specific site.
    
    Args:
        site_name (str): The name of the site.
        url (str): The URL to scrape.
        description (str): Description of the site.
        cache_days_obj (dict): Number of days to cache content (0 = no caching).
        extract_competitions (bool): Whether to extract competitions using LLM.
        nation (Optional[str]): Nation to filter competitions by.
        sub_url (Optional[str]): Sub-URL path to scrape (required with --nation)
        
    Returns:
        bool: True if successful, False otherwise.
    """
    cache_days = cache_days_obj["default"]
    logger.info(f"Starting HTML extraction from: {site_name} ({url})")
    print(f"\033[94mScraping: {site_name}\033[0m")
    print(f"\033[94mDescription: {description}\033[0m")
    print(f"\033[94mURL: {url}\033[0m")
    if cache_days == 0:
        print(f"\033[94mCache: Disabled\033[0m")
    else:
        print(f"\033[94mCache: {cache_days} day{'s' if cache_days != 1 else ''}\033[0m")
    if extract_competitions:
        print(f"\033[94mCompetition extraction: Enabled\033[0m")

    # Check if we can use cached content
    if cache_days > 0 and is_cache_valid(site_name, cache_days):
        cached_content = load_cached_content(site_name)
        if cached_content:
            print(f"\033[92m✓ Using cached content ({len(cached_content)} characters)\033[0m")
            file_path = save_html_to_file(cached_content, site_name)
            if file_path.startswith("Error"):
                print(f"\033[93m⚠ Failed to save cached content to file: {file_path}\033[0m")
            else:
                print(f"\033[92m✓ Cached content saved to: {file_path}\033[0m")
            if extract_competitions:
                # Check competitions cache
                if is_competitions_cache_valid(site_name, cache_days_obj):
                    competitions_data = load_competitions_cache(site_name)
                    if competitions_data:
                        print(f"\033[92m✓ Using cached competitions data\033[0m")
                        display_competitions(competitions_data, site_name)
                        return True
                # If competitions.json exists (even if expired), load and display it instead of fetching again
                cache_file = get_competitions_cache_file_path(site_name)
                if cache_file.exists():
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            competitions_data = json.load(f)
                        print(f"\033[93m⚠ Using existing competitions.json (expired cache)\033[0m")
                        display_competitions(competitions_data, site_name)
                        return True
                    except Exception as e:
                        logger.error(f"Failed to load competitions.json for {site_name}: {str(e)}. Refetching...")
                print(f"\033[94mExtracting competitions from cached content...\033[0m")
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

    max_retries = 1
    retry_delay = 2  # seconds, initial delay
    
    # Compile regex pattern once for performance
    rate_limit_pattern = re.compile(r'\b(?:429|HTTP\s*429|status\s*429|rate\s*limit|too\s*many\s*requests)\b', re.IGNORECASE)
    payment_required_pattern = re.compile(r'\b(?:402|HTTP\s*402|status\s*402|payment\s*required|insufficient\s*credits)\b', re.IGNORECASE)
    
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            logger.info(f"Attempt {attempt} to extract HTML from {site_name}.")
            
            # Extract HTML content
            html_content = extract_html_from_url(url)
            
            if html_content.startswith("Error:") or html_content.startswith("Exception:"):
                raise Exception(html_content)
            
            # Save HTML to file
            file_path = save_html_to_file(html_content, site_name)
            
            if file_path.startswith("Error"):
                raise Exception(file_path)
            
            # Cache the content if caching is enabled
            if cache_days > 0:
                if save_cached_content(site_name, html_content):
                    print(f"\033[92m✓ Content cached for {cache_days} day{'s' if cache_days != 1 else ''}\033[0m")
                else:
                    print(f"\033[93m⚠ Failed to cache content\033[0m")
            
            logger.info(f"HTML extraction and saving completed successfully for {site_name}.")
            print(f"\033[92m✓ HTML extracted and saved to: {file_path}\033[0m")
            print(f"\033[92m✓ Content length: {len(html_content)} characters\033[0m")
            
            # Extract competitions if requested
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
            
        except Exception as e:
            msg = str(e)
            logger.error(f"Error on attempt {attempt} for {site_name}: {msg}")
            logger.debug(traceback.format_exc())
            
            # Handle 429 errors with 3 retries, others with 1 retry
            if rate_limit_pattern.search(msg):
                max_retries = 3
                retry_match = re.search(r"'retryDelay': '([0-9]+)s'", msg)
                if retry_match:
                    retry_delay = int(retry_match.group(1))
                logger.warning(f"Rate limit error (429) for {site_name}. Retrying in {retry_delay} seconds... (attempt {attempt}/{max_retries})")
                print(f"\033[93m⚠ Rate limit error (429). Retrying in {retry_delay} seconds... (attempt {attempt}/{max_retries})\033[0m")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            elif payment_required_pattern.search(msg):
                logger.critical(f"Payment required error (402) for {site_name}. Terminating immediately.")
                print(f"\033[91m✗ Payment required error (402). Terminating immediately.\033[0m")
                return False
            elif ("RateLimitError" in msg or "RESOURCE_EXHAUSTED" in msg or "timed out" in msg or "Timeout" in msg):
                retry_match = re.search(r"'retryDelay': '([0-9]+)s'", msg)
                if retry_match:
                    retry_delay = int(retry_match.group(1))
                logger.warning(f"Retryable error (rate limit or timeout) for {site_name}. Retrying in {retry_delay} seconds... (attempt {attempt}/{max_retries})")
                print(f"\033[93m⚠ Retryable error (rate limit or timeout). Retrying in {retry_delay} seconds... (attempt {attempt}/{max_retries})\033[0m")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.critical(f"Non-retryable error encountered for {site_name}. Logging and continuing.")
                print(f"\033[91m✗ Non-retryable error encountered. See log for details.\033[0m")
                time.sleep(retry_delay)
                retry_delay *= 2
    else:
        logger.error(f"Failed after maximum retries due to persistent errors for {site_name}.")
        print(f"\033[91m✗ Failed after maximum retries due to persistent errors.\033[0m")
        return False

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
  python scrape.py --site worldfootball --extract-competitions  # Scrape and extract competitions
  python scrape.py --site worldfootball --cache-days 0          # Scrape without caching
  python scrape.py --site all --cache-days 7                    # Scrape all sites with 7-day cache
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
    # If --nation is provided, --sub-url must also be provided
    if args.nation and not args.sub_url:
        print("\033[91mError: --sub-url is required when using --nation.\033[0m")
        sys.exit(1)
    
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
            success = scrape_site(site_name, info["url"], info["description"], cache_days_obj, args.extract_competitions, nation=args.nation, sub_url=args.sub_url)
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
    success = scrape_site(args.site, url, description, cache_days_obj, args.extract_competitions, nation=args.nation, sub_url=args.sub_url)
    if success:
        print(f"\n\033[92m✓ Scraping completed successfully!\033[0m")
    else:
        print(f"\n\033[91m✗ Scraping failed.\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main() 