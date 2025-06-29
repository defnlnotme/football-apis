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

import pathlib
import os
import json
import logging
import asyncio
import time
import re
import tempfile
from typing import Optional, Dict, Any, List, Any
from dotenv import load_dotenv
from camel.logger import get_logger
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from playwright.async_api import async_playwright, Browser, Page
from prompts import ODDS_SYSTEM_PROMPT, WEB_SCRAPING_AGENT_SYSTEM_PROMPT, SELECTOR_AGENT_SYSTEM_PROMPT, Models

# Load environment variables
base_dir = pathlib.Path(__file__).parent
env_path = base_dir / ".envrc"
load_dotenv(dotenv_path=str(env_path))

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manages API key rotation to handle rate limits."""
    
    def __init__(self):
        self.api_keys_file = base_dir / "apikeys.json"
        self.rate_limit_counts = {}  # Track 429 errors per platform
        self.current_key_indices = {}  # Track current key index per platform
        self.api_keys_cache = {}  # Cache loaded API keys
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from apikeys.json file."""
        try:
            if self.api_keys_file.exists():
                with open(self.api_keys_file, 'r') as f:
                    self.api_keys_cache = json.load(f)
                logger.info(f"Loaded API keys for platforms: {list(self.api_keys_cache.keys())}")
                
                # Set initial environment variables for all platforms
                for platform, keys in self.api_keys_cache.items():
                    if isinstance(keys, list) and keys:  # For platforms with multiple keys
                        env_var_name = f"{platform.upper()}_API_KEY"
                        os.environ[env_var_name] = keys[0]  # Set first key as default
                        logger.info(f"Set initial API key for {platform}")
                    elif isinstance(keys, dict) and 'api_key' in keys:  # For platforms with single key
                        env_var_name = f"{platform.upper()}_API_KEY"
                        os.environ[env_var_name] = keys['api_key']
                        logger.info(f"Set initial API key for {platform}")
            else:
                logger.warning(f"API keys file not found: {self.api_keys_file}")
                self.api_keys_cache = {}
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            self.api_keys_cache = {}
    
    def _get_platform_from_model(self, model_type: str) -> str:
        """Extract platform name from model type."""
        if model_type.startswith('gemini'):
            return 'gemini'
        elif model_type.startswith('openai'):
            return 'openai'
        elif model_type.startswith('anthropic'):
            return 'anthropic'
        else:
            # Default to gemini for unknown models
            return 'gemini'
    
    def _get_api_keys_for_platform(self, platform: str) -> List[str]:
        """Get list of API keys for a platform."""
        return self.api_keys_cache.get(platform, [])
    
    def record_rate_limit_error(self, model_type: str):
        """Record a rate limit error and rotate keys immediately."""
        platform = self._get_platform_from_model(model_type)
        
        logger.warning(f"Rate limit for platform: {platform}")
        
        # Rotate the key immediately on first rate limit error
        self._rotate_api_key(platform)
    
    def _rotate_api_key(self, platform: str):
        """Rotate to the next API key for the platform."""
        api_keys = self._get_api_keys_for_platform(platform)
        if not api_keys:
            logger.warning(f"No API keys available for platform: {platform}")
            return
        
        if platform not in self.current_key_indices:
            self.current_key_indices[platform] = 0
        else:
            # Move to next key, cycle back to 0 if at end
            self.current_key_indices[platform] = (self.current_key_indices[platform] + 1) % len(api_keys)
        
        new_key = api_keys[self.current_key_indices[platform]]
        
        # Update environment variable
        env_var_name = f"{platform.upper()}_API_KEY"
        os.environ[env_var_name] = new_key
        
        logger.info(f"Rotated API key for {platform} to index {self.current_key_indices[platform]}")
    
    def get_current_api_key(self, platform: str) -> Optional[str]:
        """Get the current API key for a platform."""
        api_keys = self._get_api_keys_for_platform(platform)
        if not api_keys:
            return None
        
        if platform not in self.current_key_indices:
            self.current_key_indices[platform] = 0
        
        return api_keys[self.current_key_indices[platform]]


# Global API key manager instance
api_key_manager = APIKeyManager()


class WebScrapingAgent:
    """A web scraping agent that uses Playwright with LLM guidance for autonomous interaction."""
    
    def __init__(self, headless: bool = True, model_type: str = Models.flash_lite):
        self.headless = headless
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.model_type = model_type
        self.llm_agent: Optional[ChatAgent] = None
        
        # Data collection for markets with diff tracking
        self.collected_markets = []  # Store individual market data
        self.processed_markets = set()  # Track which markets have been processed to avoid duplicates
        self.market_states = {}     # Track which markets were processed
        self.last_market_count = 0  # Track number of markets found
        self.page_snapshots = []    # Store page snapshots for diffing
        self.current_page_html = "" # Current page HTML for diffing
        self.processed_categories = set()  # Track processed market categories/sections
        
        # State tracking for session management
        self.all_markets_clicked = False  # Track if "all markets" button has been clicked
        self.session_actions = []  # Track all actions taken in this session
        self.current_phase = "initial"  # Track current phase: initial, markets_expanded, collecting_markets
        
        # Initialize the LLM agent
        self._init_llm_agent()
        
        self.odds_system_prompt = ODDS_SYSTEM_PROMPT
        
        self.generic_system_prompt = """You are a web scraping agent that navigates websites and extracts content. Your task is to:

1. Navigate to the provided URL
2. Handle any popups, overlays, or modal dialogs that appear
3. Follow the specific instructions provided in the prompt
4. Wait for any dynamic content to load
5. Return the complete HTML content for further processing

Be thorough in your navigation and ensure all requested content is visible before scraping."""
    
    def _init_llm_agent(self):
        """Initialize the LLM agent with proper model configuration."""
        try:
            # Create the model using ModelFactory
            model = ModelFactory.create(
                model_platform=ModelPlatformType.GEMINI,
                model_type=self.model_type
            )
            
            # Create the LLM agent for intelligent decision making
            self.llm_agent = ChatAgent(
                model=model,
                system_message=WEB_SCRAPING_AGENT_SYSTEM_PROMPT
            )
            
            logger.info("LLM agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM agent: {e}")
            raise e
    
    def _mark_all_markets_clicked(self):
        """Mark that the 'all markets' button has been clicked."""
        self.all_markets_clicked = True
        logger.info("Marked 'all markets' button as clicked")

    def _reset_session_state(self):
        """Reset session state for a new scraping session."""
        self.all_markets_clicked = False
        self.session_actions = []
        self.current_phase = "initial"
        self.collected_markets = []
        self.processed_markets.clear()
        self.market_states = {}
        self.last_market_count = 0
        self.page_snapshots = []
        self.current_page_html = ""
        self.processed_categories.clear()
        logger.info("Session state reset for new scraping session")
    
    def _record_session_action(self, action: str, details: str = ""):
        """Record an action taken in the current session."""
        timestamp = time.time()
        action_record = {
            'timestamp': timestamp,
            'action': action,
            'details': details,
            'phase': self.current_phase
        }
        self.session_actions.append(action_record)
        logger.debug(f"Recorded session action: {action} - {details}")
    
    def _get_session_summary(self) -> str:
        """Get a summary of actions taken in the current session."""
        if not self.session_actions:
            return "No actions taken yet"
        
        summary_parts = []
        for action in self.session_actions[-10:]:  # Last 10 actions
            summary_parts.append(f"- {action['action']}: {action['details']}")
        
        return "\n".join(summary_parts)
    
    def _update_phase(self, new_phase: str):
        """Update the current phase of the scraping session."""
        old_phase = self.current_phase
        self.current_phase = new_phase
        logger.info(f"Phase transition: {old_phase} -> {new_phase}")
    
    def _reinit_llm_agent_with_new_key(self):
        """Reinitialize the LLM agent with a new API key after rotation."""
        try:
            platform = api_key_manager._get_platform_from_model(self.model_type)
            
            # Create the model using ModelFactory with potentially new API key
            model = ModelFactory.create(
                model_platform=ModelPlatformType.GEMINI,
                model_type=self.model_type
            )
            
            # Create the LLM agent for intelligent decision making
            self.llm_agent = ChatAgent(
                model=model,
                system_message=WEB_SCRAPING_AGENT_SYSTEM_PROMPT
            )
            
            logger.info("LLM agent reinitialized with new API key")
            
        except Exception as e:
            logger.error(f"Failed to reinitialize LLM agent: {e}")
            raise e
    
    async def _init_browser(self):
        """Initialize the browser and page."""
        if self.browser is None:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu'
                ]
            )
            self.page = await self.browser.new_page()
            
            # Set user agent to avoid detection
            await self.page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
    
    async def _close_browser(self):
        """Close the browser and cleanup."""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.page = None
    
    async def _capture_page_snapshot(self) -> str:
        """Capture a snapshot of the current page HTML for diffing."""
        if not self.page:
            return ""
        
        try:
            # Wait for any pending AJAX requests to complete
            await asyncio.sleep(1)
            
            # Get the full page HTML
            html = await self.page.content()
            self.current_page_html = html
            return html
        except Exception as e:
            logger.error(f"Error capturing page snapshot: {e}")
            return ""
    
    async def _find_page_changes(self, before_html: str, after_html: str) -> List[Dict[str, Any]]:
        """Find changes in the page HTML after a button click to identify new market data."""
        import difflib
        import re
        from bs4 import BeautifulSoup
        
        if not before_html or not after_html:
            return []
        
        try:
            # Parse both HTMLs
            soup_before = BeautifulSoup(before_html, 'html.parser')
            soup_after = BeautifulSoup(after_html, 'html.parser')
            
            # Find elements that contain odds (numbers with decimals)
            odds_pattern = re.compile(r'\b\d+\.\d+\b')
            
            changes = []
            processed_containers = set()
            
            # Use LLM to find elements that might contain market data
            user_goal = "find all elements (divs, tables, spans, buttons) that contain betting odds data or market information"
            previous_actions = "Analyzing page changes for market data"
            
            # Get LLM-suggested selectors for elements that might contain market data
            element_selector = self._llm_find_selector(after_html, user_goal, previous_actions)
            
            if not element_selector:
                logger.warning("LLM could not suggest selectors for market elements, skipping page change analysis")
                return []
            
            # Look for new elements with odds in the after HTML using LLM-suggested selector
            for element in soup_after.select(element_selector):
                text = element.get_text(strip=True)
                if odds_pattern.search(text):
                    # Check if this element is new (not in before HTML)
                    element_str = str(element)
                    
                    # Simple check: if this exact HTML string is not in before HTML, it's new
                    if element_str not in before_html:
                        # Find the parent container that likely contains the full market data
                        # Use LLM to find container selectors
                        container_user_goal = "find containers (table, div, section, article) that might hold market data"
                        container_previous_actions = "Finding parent containers for market data"
                        container_selector = self._llm_find_selector(after_html, container_user_goal, container_previous_actions)
                        
                        if not container_selector:
                            logger.warning("LLM could not suggest container selectors, skipping this element")
                            continue
                        
                        # Look for parent containers using LLM-suggested selector
                        container = element.find_parent(container_selector.split(', '))
                        if not container:
                            # If no container found, skip this element
                            continue
                        
                        if container:
                            # Get a unique identifier for this container to avoid duplicates
                            container_id = str(container)
                            if container_id in processed_containers:
                                continue
                            processed_containers.add(container_id)
                            
                            # Only process containers that look like actual market tables/containers
                            # Skip individual rows or small elements that are just part of a larger market
                            container_text = container.get_text(strip=True)
                            
                            # Check if this container has enough structure to be a market
                            # Look for multiple odds values and some text content
                            odds_matches = odds_pattern.findall(container_text)
                            
                            # Skip if this looks like just a single row (few odds values and short text)
                            if len(odds_matches) < 3 or len(container_text) < 20:
                                continue
                            
                            # Skip if this looks like just a single betting option (e.g., "Milan1.44")
                            # A proper market should have multiple options or bookmakers
                            if len(odds_matches) < 6 and not any(keyword in container_text.lower() for keyword in ['vincente', 'pareggio', 'sconfitta', 'over', 'under', 'totale']):
                                continue
                            
                            # Extract structured data from this container
                            change_data = {
                                'container_text': container_text,
                                'market_type': self._identify_market_type_from_text(container_text),
                                'timestamp': time.time()
                            }
                            changes.append(change_data)
            
            logger.info(f"Found {len(changes)} page changes with new market data")
            return changes
            
        except Exception as e:
            logger.error(f"Error finding page changes: {e}")
            return []
    
    def _identify_market_type_from_text(self, text: str) -> str:
        """Identify market type from text content (for use in page diffing)."""
        # Simplified approach - let the LLM handle market type identification
        # Just extract a meaningful name from the text
        text_lower = text.lower()
        
        # Basic fallback for common patterns, but don't restrict to these
        if any(word in text_lower for word in ['vincente', 'winner', 'win']):
            return 'match_winner'
        elif any(word in text_lower for word in ['totale gol', 'total goals', 'over under']):
            return 'total_goals'
        elif any(word in text_lower for word in ['handicap', 'asiatico']):
            return 'asian_handicap'
        elif any(word in text_lower for word in ['risultato esatto', 'exact score']):
            return 'exact_score'
        elif any(word in text_lower for word in ['primo tempo', 'first half']):
            return 'first_half'
        elif any(word in text_lower for word in ['secondo tempo', 'second half']):
            return 'second_half'
        elif any(word in text_lower for word in ['margine vittoria', 'victory margin']):
            return 'victory_margin'
        elif any(word in text_lower for word in ['both teams', 'entrambe le squadre']):
            return 'both_teams_score'
        else:
            # Extract a meaningful name from the text instead of just "unknown"
            words = text.split()
            if len(words) >= 2:
                # Take first two meaningful words
                meaningful_words = [w for w in words[:3] if len(w) > 2 and w.lower() not in ['the', 'and', 'for', 'with', 'bet', 'odds']]
                if meaningful_words:
                    return '_'.join(meaningful_words[:2]).lower()
            return 'market'
    
    def _identify_market_type(self, text: str) -> str:
        """Identify the type of betting market from text content."""
        # Use the same simplified approach as above
        return self._identify_market_type_from_text(text)
    
    async def _capture_market_data(self) -> List[Dict[str, Any]]:
        """Capture the market data that was just loaded via AJAX, focusing on odds containers."""
        if not self.page:
            return []
        
        try:
            await asyncio.sleep(2)
            all_markets = []
            seen_containers = set()
            
            # Use LLM to find odds container selectors dynamically
            page_html = await self.page.content()
            
            user_goal = "find all containers (divs, tables, sections) that contain betting odds data. These are elements that hold market information like odds values, betting options, and market names."
            previous_actions = "Capturing market data after AJAX load"
            
            # Get LLM-suggested selectors for odds containers
            container_selector = self._llm_find_selector(page_html, user_goal, previous_actions)
            
            if not container_selector:
                logger.warning("LLM could not suggest selectors for odds containers, using minimal fallback")
                odds_containers = await self.page.query_selector_all('div, table')
            else:
                odds_containers = await self.page.query_selector_all(container_selector)
            
            for container in odds_containers:
                try:
                    # Get a unique identifier for this container
                    container_id = await container.evaluate('el => el.id || el.className || el.outerHTML.slice(0,100)')
                    if container_id in seen_containers:
                        continue
                    seen_containers.add(container_id)
                    
                    # Extract structured odds data
                    market_data = await self._extract_structured_odds_data(container)
                    
                    if market_data and market_data.get('structure') != 'text':  # Skip if only text content
                        # Check if this market has already been processed
                        market_name = market_data.get('market_name', 'Unknown Market')
                        if not self._is_market_already_processed(market_name):
                            all_markets.append(market_data)
                            self._mark_market_as_processed(market_name)
                        else:
                            logger.info(f"Skipping already processed market: {market_name}")
                
                except Exception as e:
                    logger.warning(f"Error processing market container: {e}")
                    continue
            
            logger.info(f"Total structured market containers captured: {len(all_markets)}")
            if all_markets:
                self.collected_markets.extend(all_markets)
            else:
                logger.warning("No structured market containers captured in this iteration")
            return all_markets
        except Exception as e:
            logger.error(f"Error capturing market data: {e}")
            return []
    
    async def _capture_market_data_with_diff(self, before_html: str) -> List[Dict[str, Any]]:
        """Capture market data using page diffing to find exactly what changed after a button click."""
        if not self.page:
            return []
        
        try:
            # Wait for AJAX to load
            await asyncio.sleep(2)
            
            # Capture the page after the button click
            after_html = await self._capture_page_snapshot()
            
            # Find changes between before and after
            changes = await self._find_page_changes(before_html, after_html)
            
            # For each change, try to find the corresponding container and extract structured data
            market_data = []
            processed_containers = set()
            
            for change in changes:
                # Look for containers that match the change
                # Use LLM to find container selectors dynamically
                page_html = await self.page.content()
                
                user_goal = "find all containers (divs, tables, sections) that contain betting markets or odds data. These are elements that hold market information and betting options."
                previous_actions = "Finding containers for market data extraction"
                
                # Get LLM-suggested selectors for market containers
                container_selector = self._llm_find_selector(page_html, user_goal, previous_actions)
                
                if not container_selector:
                    logger.warning("LLM could not suggest selectors for market containers, using minimal fallback")
                    containers = await self.page.query_selector_all('div, table')
                else:
                    containers = await self.page.query_selector_all(container_selector)
                
                for container in containers:
                    try:
                        container_text = await container.evaluate('el => el.innerText')
                        
                        # Check if this container matches the change (contains the same text)
                        if change['container_text'] in container_text:
                            # Get a unique identifier for this container to avoid duplicates
                            container_id = await container.evaluate('el => el.outerHTML.slice(0,200)')
                            if container_id in processed_containers:
                                continue
                            processed_containers.add(container_id)
                            
                            # Extract structured data from this container
                            structured_data = await self._extract_structured_odds_data(container)
                            
                            # Only process if it's a proper market structure (not just text)
                            if structured_data and structured_data.get('structure') != 'text':
                                # Check if this market has already been processed
                                market_name = structured_data.get('market_name', 'Unknown Market')
                                if not self._is_market_already_processed(market_name):
                                    structured_data['source'] = 'page_diff'
                                    market_data.append(structured_data)
                                    self._mark_market_as_processed(market_name)
                                    logger.info(f"Added market from diff: {market_name}")
                                    break
                                else:
                                    logger.info(f"Skipping already processed market from diff: {market_name}")
                    except Exception as e:
                        logger.warning(f"Error extracting structured data from container: {e}")
                        continue
            
            # Add to collected markets
            if market_data:
                self.collected_markets.extend(market_data)
                logger.info(f"Added {len(market_data)} new structured markets via diff")
            else:
                logger.warning("No structured market data found via page diffing")
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error capturing market data with diff: {e}")
            return []
    
    async def _get_market_buttons(self) -> List[Dict[str, Any]]:
        """Get all market buttons that can be clicked to load market data, and mark odds buttons."""
        import re
        if not self.page:
            return []
        
        try:
            # Use LLM to find market button selectors dynamically
            page_html = await self.page.content()
            
            user_goal = "find all clickable elements (buttons, links, divs) that are market buttons or betting controls. These are elements that when clicked will load or expand betting markets. Look for buttons with text like 'Vincente', 'Totale gol', 'Handicap Asiatico', 'Risultato Esatto', etc. Also include any expansion buttons like 'Show More', 'All Markets', 'Tutti i Mercati'."
            previous_actions = "Analyzing page for market buttons"
            
            # Get LLM-suggested selectors for market buttons
            button_selector = self._llm_find_selector(page_html, user_goal, previous_actions)
            
            if not button_selector:
                logger.warning("LLM could not suggest selectors for market buttons, skipping market button detection")
                return []
            
            button_selectors = [button_selector]
            
            float_regex = re.compile(r'^\d+(\.\d+)?$')
            # Remove hard-coded market keywords - let LLM handle all filtering
            market_buttons = []
            for selector in button_selectors:
                elements = await self.page.query_selector_all(selector)
                
                for element in elements:
                    try:
                        is_visible = await element.is_visible()
                        if not is_visible:
                            continue
                        
                        text_content = await element.text_content()
                        if not text_content or len(text_content.strip()) < 2:
                            continue
                        
                        text_stripped = text_content.strip()
                        text_lower = text_stripped.lower()
                        is_float = bool(float_regex.match(text_stripped.replace(',', '.')))
                        
                        # Let LLM determine if this is a market-related button
                        # For now, assume all non-float text buttons are potential market buttons
                        is_market_button = not is_float
                        is_odds_button = is_float
                        is_market_related = not is_float  # Let LLM handle this later
                        
                        classes = await element.evaluate('el => el.className')
                        id_attr = await element.evaluate('el => el.id')
                        tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                        
                        button_data = {
                            'text': text_stripped,
                            'classes': classes,
                            'id': id_attr,
                            'tag': tag_name,
                            'selector': selector,
                            'is_market_button': is_market_button,
                            'is_odds_button': is_odds_button,
                            'is_market_related': is_market_related,
                            'element': element
                        }
                        
                        if is_market_button:
                            logger.info(f"Found potential market button: '{text_stripped}'")
                        
                        market_buttons.append(button_data)
                        
                    except Exception as e:
                        logger.warning(f"Error processing market button: {e}")
                        continue
            
            logger.info(f"Total market buttons found: {len(market_buttons)}")
            return market_buttons
            
        except Exception as e:
            logger.error(f"Error getting market buttons: {e}")
            return []
    
    async def _get_page_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the current page state."""
        if not self.page:
            return {}
        
        try:
            # Get page title and URL
            title = await self.page.title()
            url = self.page.url
            
            # Use LLM to find button selectors dynamically
            page_html = await self.page.content()
            
            # Find buttons using LLM
            button_user_goal = "find all clickable elements (buttons, links, divs) that are interactive controls on the page"
            button_previous_actions = "Analyzing page for interactive elements"
            button_selector = self._llm_find_selector(page_html, button_user_goal, button_previous_actions)
            
            if not button_selector:
                logger.warning("LLM could not suggest button selectors, skipping button analysis")
                buttons = []
            else:
                buttons = await self.page.query_selector_all(button_selector)
            
            button_info = []
            
            for button in buttons:
                try:
                    text = await button.text_content()
                    is_visible = await button.is_visible()
                    tag_name = await button.evaluate('el => el.tagName.toLowerCase()')
                    classes = await button.evaluate('el => el.className')
                    id_attr = await button.evaluate('el => el.id')
                    href = await button.evaluate('el => el.href || ""')
                    
                    if is_visible and text and text.strip():
                        button_info.append({
                            'text': text.strip(),
                            'tag': tag_name,
                            'classes': classes,
                            'id': id_attr,
                            'href': href,
                            'visible': is_visible
                        })
                except Exception:
                    continue
            
            # Use LLM to find market container selectors dynamically
            container_user_goal = "find all containers (divs, sections, articles) that contain betting markets or odds data"
            container_previous_actions = "Analyzing page for market containers"
            container_selector = self._llm_find_selector(page_html, container_user_goal, container_previous_actions)
            
            if not container_selector:
                logger.warning("LLM could not suggest container selectors, skipping container analysis")
                market_containers = []
            else:
                market_containers = await self.page.query_selector_all(container_selector)
            
            container_info = []
            
            for container in market_containers:
                try:
                    text = await container.text_content()
                    is_visible = await container.is_visible()
                    classes = await container.evaluate('el => el.className')
                    
                    if is_visible and text and len(text.strip()) > 10:
                        container_info.append({
                            'text_preview': text.strip()[:100] + '...' if len(text.strip()) > 100 else text.strip(),
                            'classes': classes,
                            'visible': is_visible
                        })
                except Exception:
                    continue
            
            # Get market buttons specifically
            market_buttons = await self._get_market_buttons()
            market_button_info = []
            for button in market_buttons:
                market_button_info.append({
                    'text': button['text'],
                    'classes': button['classes'],
                    'id': button['id'],
                    'selector': button['selector']
                })
            
            # Get page content preview
            body_text = await self.page.evaluate('() => document.body.innerText.substring(0, 500)')
            
            # Update current market count
            current_market_count = len(container_info)
            
            return {
                'title': title,
                'url': url,
                'body_preview': body_text,
                'buttons': button_info,
                'market_containers': container_info,
                'market_buttons': market_button_info,
                'current_market_count': current_market_count,
                'collected_markets_count': len(self.collected_markets)
            }
            
        except Exception as e:
            logger.warning(f"Error getting page info: {e}")
            return {}
    
    def _execute_llm_instruction(self, instruction: str, page_info: Dict[str, Any]) -> str:
        """Use LLM to analyze page state and determine next action."""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Check if LLM agent is initialized
                if self.llm_agent is None:
                    logger.error("LLM agent is not initialized")
                    return self._fallback_heuristics(instruction, page_info)
                
                # Log brief summary instead of full content
                buttons_count = len(page_info.get('buttons', []))
                market_containers_count = len(page_info.get('market_containers', []))
                market_buttons_count = len(page_info.get('market_buttons', []))
                current_market_count = page_info.get('current_market_count', 0)
                collected_markets_count = page_info.get('collected_markets_count', 0)
                
                logger.info(f"Analyzing page state: {buttons_count} buttons, {market_containers_count} containers, {market_buttons_count} market buttons, {current_market_count} current markets, {collected_markets_count} collected")
                
                # Get session summary for context
                session_summary = self._get_session_summary()
                
                # Prepare the prompt for the LLM
                prompt = f"""Current page state:
Title: {page_info.get('title', 'Unknown')}
URL: {page_info.get('url', 'Unknown')}
Body preview: {page_info.get('body_preview', 'No content')}

Current phase: {self.current_phase}
All markets button clicked: {self.all_markets_clicked}

Available buttons:
{json.dumps(page_info.get('buttons', []), indent=2)}

Market containers found:
{json.dumps(page_info.get('market_containers', []), indent=2)}

Market buttons found:
{json.dumps(page_info.get('market_buttons', []), indent=2)}

Current market count: {page_info.get('current_market_count', 0)}
Collected markets: {page_info.get('collected_markets_count', 0)}

SESSION MEMORY - Previous actions taken:
{session_summary}

User goal: {instruction}

CRITICAL INSTRUCTIONS FOR ODDSCHECKER:
1. You MUST click market buttons to load odds data via AJAX
2. Look for buttons like "Vincente", "Totale gol", "Handicap Asiatico", etc.
3. Click "Tutti i Mercati" first if available to expand all markets
4. Then click each individual market button one by one
5. Wait after each click for AJAX to load
6. Only stop when all market buttons have been clicked

IMPORTANT SESSION CONTEXT:
- The "all markets" button can be clicked after clicking a market to expand all categories
- Use the session memory to avoid repeating actions unnecessarily
- Focus on clicking individual market buttons to collect odds data
- The "all markets" button is used to reveal additional market categories

Based on the current page state, session memory, and user goal, what is the next best action? 
Return only the action command (e.g., CLICK_BUTTON:"Vincente", CLICK_BUTTON:"Tutti i Mercati", WAIT:3, STOP, etc.)"""

                # Create the human message
                human_message = BaseMessage.make_user_message(
                    role_name="WebCrawler",
                    content=prompt
                )

                # Get response from LLM (synchronous call)
                response = self.llm_agent.step(human_message)
                action = response.msgs[0].content.strip()
                
                logger.info(f"LLM suggested action: {action}")
                return action
                
            except Exception as e:
                error_str = str(e).lower()
                if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str:
                    logger.warning(f"Rate limit ({retry_count + 1}/{max_retries}): {e}")
                    
                    # Record the rate limit error
                    api_key_manager.record_rate_limit_error(self.model_type)
                    
                    # Reinitialize the agent with the new API key
                    self._reinit_llm_agent_with_new_key()
                    
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying with new API key...")
                        time.sleep(2)  # Brief pause before retry
                        continue
                    else:
                        logger.error(f"Max retries reached for rate limit errors")
                        return self._fallback_heuristics(instruction, page_info)
                else:
                    logger.error(f"Error getting LLM instruction: {e}")
                    return self._fallback_heuristics(instruction, page_info)
        
        return self._fallback_heuristics(instruction, page_info)
    
    def _fallback_heuristics(self, instruction: str, page_info: Dict[str, Any]) -> str:
        """Fallback heuristics when LLM is not available."""
        # This method has been removed to comply with the no-hardcoded-selectors rule
        # All filtering and decision making must be done by the LLM
        logger.error("LLM must handle all decisions - no fallback heuristics available")
        raise Exception("LLM failed to provide instruction - no fallback mechanisms allowed")
    
    async def _execute_action(self, action: str) -> bool:
        """Execute the action determined by the LLM."""
        if not self.page:
            return False
        
        try:
            action = action.strip()
            
            if action.startswith("CLICK_BUTTON:"):
                button_text = action.split(":", 1)[1].strip('"')
                logger.info(f"Attempting to click button: {button_text}")
                
                # Check if this is the "all markets" button
                # Let LLM determine this instead of hard-coded keywords
                is_all_markets_button = False  # Will be determined by LLM context
                
                # Capture page state before clicking
                before_html = await self._capture_page_snapshot()
                
                # Find and click the button by exact text (case-insensitive, trimmed)
                market_buttons = await self._get_market_buttons()
                button_clicked = False
                
                for button in market_buttons:
                    try:
                        text = button.get('text', '').strip()
                        is_market_button = button.get('is_market_button', False)
                        is_odds_button = button.get('is_odds_button', False)
                        element = button.get('element')
                        if (
                            is_market_button and not is_odds_button and element and await element.is_visible() and
                            text.lower() == button_text.strip().lower()
                        ):
                            await element.click()
                            logger.info(f"Successfully clicked market button: {text}")
                            
                            # Record the action
                            if is_all_markets_button:
                                self._mark_all_markets_clicked()
                                self._record_session_action("CLICK_BUTTON", f"All markets button: {text}")
                            else:
                                self._record_session_action("CLICK_BUTTON", f"Market button: {text}")
                            
                            # Use diff-based capture to find exactly what changed
                            await self._capture_market_data_with_diff(before_html)
                            
                            await asyncio.sleep(2)
                            button_clicked = True
                            break
                    except Exception as e:
                        logger.warning(f"Error clicking market button: {e}")
                        continue
                
                if not button_clicked:
                    logger.warning(f"Could not find or click market button: {button_text}")
                    self._record_session_action("FAILED_CLICK", f"Could not click: {button_text}")
                    return False
                
                return True
            
            elif action.startswith("WAIT:"):
                seconds = int(action.split(":", 1)[1])
                logger.info(f"Waiting for {seconds} seconds")
                self._record_session_action("WAIT", f"{seconds} seconds")
                await asyncio.sleep(seconds)
                return True
            
            elif action.startswith("SCROLL:"):
                direction = action.split(":", 1)[1].lower()
                logger.info(f"Scrolling {direction}")
                self._record_session_action("SCROLL", direction)
                
                if direction == "down":
                    await self.page.evaluate("window.scrollBy(0, 500)")
                elif direction == "up":
                    await self.page.evaluate("window.scrollBy(0, -500)")
                elif direction == "left":
                    await self.page.evaluate("window.scrollBy(-500, 0)")
                elif direction == "right":
                    await self.page.evaluate("window.scrollBy(500, 0)")
                
                await asyncio.sleep(1)
                return True
            
            elif action == "STOP":
                logger.info("LLM instructed to stop")
                self._record_session_action("STOP", "LLM decided to stop")
                return True
            
            else:
                logger.warning(f"Unknown action: {action}")
                self._record_session_action("UNKNOWN_ACTION", action)
                return False
                
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            self._record_session_action("ERROR", f"Error executing {action}: {str(e)}")
            return False
    
    async def _autonomous_interaction(self, instruction: str, max_iterations: int = 15) -> List[Dict[str, Any]]:
        """Autonomously interact with the page based on the instruction."""
        logger.info(f"Starting autonomous interaction with instruction: {instruction}")
        
        # Reset session state for this interaction
        self._reset_session_state()
        
        # Initial capture of page state
        logger.info("Capturing initial page state...")
        initial_html = await self._capture_page_snapshot()
        self.page_snapshots.append(initial_html)
        
        # Phase 1: Find and click "all markets" button
        logger.info("Phase 1: Looking for 'all markets' button...")
        all_markets_clicked = False
        
        for iteration in range(5):  # Try for 5 iterations to find "all markets"
            page_info = await self._get_page_info()
            
            # Look specifically for "all markets" button
            action = self._execute_llm_instruction(
                "Find and click the 'all markets' or 'tutti i mercati' button to expand all available markets. This is the first priority.",
                page_info
            )
            
            if action.startswith("CLICK_BUTTON:"):
                button_text = action.split(":", 1)[1].strip('"')
                # Let LLM determine if this is an "all markets" button instead of hard-coded keywords
                success = await self._execute_action(action)
                if success:
                    all_markets_clicked = True
                    logger.info("Successfully clicked button that LLM identified as 'all markets'")
                    break
            
            await asyncio.sleep(1)
        
        if not all_markets_clicked:
            logger.warning("Could not find 'all markets' button, proceeding with individual market buttons")
        
        # Phase 2: Click individual market buttons and capture data
        logger.info("Phase 2: Clicking individual market buttons and capturing data...")
        self._update_phase("collecting_markets")
        
        for iteration in range(max_iterations):
            logger.info(f"Market button iteration {iteration + 1}/{max_iterations}")
            
            # Get current page state
            page_info = await self._get_page_info()
            
            # Get LLM instruction for next action
            action = self._execute_llm_instruction(
                "You are on a betting odds page. Your goal is to collect ALL available market types and their odds data. "
                "Systematically click each market button (like 'Vincente', 'Totale gol', 'Handicap Asiatico', 'Risultato Esatto', etc.) "
                "to load their odds data. The LLM will automatically categorize each market type based on its content. "
                "Only click market buttons, not individual odds buttons. Continue until all market buttons have been clicked and their data loaded. "
                "Stop when you've clicked all available market buttons.",
                page_info
            )
            
            # Execute the action
            success = await self._execute_action(action)
            
            if not success:
                logger.warning(f"Action failed: {action}")
            
            # Check if we should stop
            if action == "STOP":
                logger.info("LLM decided to stop interaction")
                break
            
            # Wait a bit for any changes to take effect
            await asyncio.sleep(1)
            
            # Check if we've collected enough markets
            if len(self.collected_markets) > 0:
                logger.info(f"Collected {len(self.collected_markets)} markets so far")
                
                # If we've been collecting markets but no new ones for a few iterations, we might be done
                if iteration > 5 and len(self.collected_markets) >= 3:
                    logger.info("Collected sufficient markets, interaction may be complete")
                    break
            
            # If no markets collected yet, try clicking some buttons manually
            if len(self.collected_markets) == 0 and iteration > 3:
                logger.info("No markets collected yet, trying manual button clicking...")
                await self._try_manual_market_clicking()
        
        # Final capture of any remaining market data
        logger.info("Performing final market data capture...")
        final_html = await self._capture_page_snapshot()
        if final_html != initial_html:
            await self._capture_market_data_with_diff(initial_html)
        
        logger.info(f"Interaction complete. Collected {len(self.collected_markets)} unique markets total.")
        logger.info(f"Total markets processed: {len(self.processed_markets)}")
        return self.collected_markets
    
    async def _try_manual_market_clicking(self):
        """Try clicking market buttons manually when LLM approach fails."""
        try:
            logger.info("Attempting manual market button clicking...")
            self._record_session_action("MANUAL_CLICKING", "Starting manual market button clicking")
            
            # Get all market buttons
            market_buttons = await self._get_market_buttons()
            
            # Filter out already processed markets
            unprocessed_buttons = self._get_unprocessed_markets(market_buttons)
            logger.info(f"Found {len(unprocessed_buttons)} unprocessed market buttons for manual clicking")
            
            # Click the first few unprocessed market-related buttons
            clicked_count = 0
            for button in unprocessed_buttons[:5]:  # Try first 5 unprocessed buttons
                try:
                    if button.get('is_market_related', False):
                        element = button.get('element')
                        if element and await element.is_visible():
                            # Mark as processed before clicking
                            self._mark_market_as_processed(button['text'])
                            
                            # Capture page state before clicking
                            before_html = await self._capture_page_snapshot()
                            
                            await element.click()
                            logger.info(f"Manually clicked market button: {button['text']}")
                            self._record_session_action("MANUAL_CLICK", button['text'])
                            
                            # Use diff-based capture with structured extraction
                            await self._capture_market_data_with_diff(before_html)
                            
                            await asyncio.sleep(2)
                            clicked_count += 1
                            
                            if clicked_count >= 3:  # Limit to 3 clicks
                                break
                except Exception as e:
                    logger.warning(f"Error manually clicking button: {e}")
                    self._record_session_action("ERROR_MANUAL_CLICK", f"{button.get('text', 'unknown')}: {str(e)}")
                    continue
            
            logger.info(f"Manual clicking completed. Clicked {clicked_count} unprocessed buttons.")
            self._record_session_action("MANUAL_CLICKING_COMPLETE", f"Clicked {clicked_count} buttons")
            
        except Exception as e:
            logger.error(f"Error in manual market clicking: {e}")
            self._record_session_action("ERROR_MANUAL_CLICKING", str(e))
    
    async def scrape_odds_markets(self, url: str, team: str, vs_team: str, competition: str) -> Dict[str, Any]:
        """
        Scrape betting odds markets following the exact manual workflow:
        1. Navigate to URL
        2. Click "all markets" button
        3. Use LLM to find popup or inline expanded markets
        4. Iterate through market categories
        5. For each category, click each market one by one
        6. Diff HTML after each click to capture AJAX-loaded odds data
        7. Store in flattened JSON schema
        
        Args:
            url: The URL to scrape
            team: The home team name
            vs_team: The away team name  
            competition: The competition name
            
        Returns:
            Flattened odds data in the format:
            {
                "$market_name": {
                    "$bet_condition": {
                        "$bookmaker": "$odd_float"
                    }
                }
            }
            
        Raises:
            Exception: If scraping fails, this will terminate the script
        """
        try:
            logger.info(f"Starting LLM-driven odds market scraping for {team} vs {vs_team} in {competition}")
            logger.info(f"URL: {url}")
            
            # Reset session state for this scraping session
            self._reset_session_state()
            
            await self._init_browser()
            
            # Ensure page is initialized
            if not self.page:
                raise Exception("Failed to initialize browser page")
            
            # Navigate to the URL
            await self.page.goto(url, wait_until='networkidle')
            logger.info("Successfully navigated to the page")
            self._record_session_action("NAVIGATE", f"Navigated to {url}")
            
            # Step 1: Find and click the "all markets" button
            logger.info("Step 1: Looking for 'all markets' button...")
            all_markets_button = await self._find_all_markets_button()
            if not all_markets_button:
                raise Exception("Could not find 'all markets' button")
            
            # Capture page state before clicking
            before_html = await self._capture_page_snapshot()
            
            # Click the all markets button
            await all_markets_button.click()
            logger.info("Clicked 'all markets' button")
            
            # Wait for popup to appear and content to stabilize
            popup_appeared = await self._wait_for_popup_to_appear(max_wait_time=10)
            if not popup_appeared:
                logger.warning("Popup did not appear, but continuing with scraping...")
            
            await self._wait_for_network_idle(max_wait_time=5)
            await self._wait_for_content_stabilization(max_wait_time=5)
            
            # Step 2: Use LLM to find the popup or market container
            logger.info("Step 2: Using LLM to find markets popup or container...")
            after_html = await self._capture_page_snapshot()
            
            user_goal = f"find the main container or popup that contains ALL the betting odds markets for {team} vs {vs_team} in {competition}. This should be a large container that holds multiple market categories like 'Vincente', 'Totale gol', 'Handicap Asiatico', etc. Do NOT select individual buttons or small elements - find the main container that contains many market elements."
            previous_actions = "Clicked 'Tutti i Mercati' button to expand all markets"
            
            # Use LLM to find the selector for the popup or market container
            selector = self._llm_find_selector(after_html, user_goal, previous_actions)
            
            if not selector:
                raise Exception("LLM could not suggest a selector for finding markets")
            
            # Use the LLM-suggested selector to find the popup or market container
            popup_or_container = await self._find_element_with_selector(selector)
            if not popup_or_container:
                logger.warning(f"Could not find element with LLM-suggested selector: {selector}")
                logger.info("Falling back to manual popup detection...")
                popup_or_container = await self._find_markets_popup()
                if not popup_or_container:
                    logger.warning("Could not find markets popup with manual detection either")
                    logger.info("Falling back to inline expansion scraping...")
                    return await self._scrape_from_inline_expansion()
            
            logger.info(f"Found markets container using selector: {selector}")
            
            # Step 3: Use LLM to find market categories within the container
            logger.info("Step 3: Using LLM to find market categories...")
            container_html = await popup_or_container.evaluate('el => el.outerHTML')
            
            user_goal = "find all market category containers (like 'Esiti incontro', 'Risultato finale', 'Handicap Asiatico', 'Over/Under') within the markets popup. These are the main category sections that contain multiple individual markets within them."
            previous_actions = "Found markets popup"
            
            category_selector = self._llm_find_selector(container_html, user_goal, previous_actions)
            
            if not category_selector:
                logger.warning("LLM could not suggest a selector for market categories, using fallback")
                # Use fallback approach
                return await self._scrape_from_inline_expansion()
            
            # Find market categories using LLM-suggested selector
            market_categories = await self._get_market_categories_with_selector(popup_or_container, category_selector)
            logger.info(f"Found {len(market_categories)} market categories using LLM")
            
            # If no categories found with LLM, fall back to manual detection
            if not market_categories:
                logger.warning("No market categories found with LLM selector, trying manual detection...")
                market_categories = await self._get_market_categories(popup_or_container)
                if not market_categories:
                    logger.warning("No market categories found with manual detection either")
                    logger.info("Falling back to inline expansion scraping...")
                    return await self._scrape_from_inline_expansion()
            
            # Step 4: Iterate through each category and click each market
            logger.info("Step 4: Iterating through categories and markets...")
            all_odds_data = {}
            
            for category_index, category in enumerate(market_categories):
                logger.info(f"Processing category {category_index + 1}/{len(market_categories)}: {category['name']}")
                
                # Click on the category to expand it (if needed)
                await self._click_category(category)
                await asyncio.sleep(1)
                
                # Get all markets in this category using LLM
                markets_in_category = await self._get_markets_in_category_with_llm(category, popup_or_container)
                logger.info(f"Found {len(markets_in_category)} markets in category '{category['name']}'")
                
                # Filter out already processed markets
                unprocessed_markets = self._get_unprocessed_markets(markets_in_category)
                logger.info(f"Found {len(unprocessed_markets)} unprocessed markets in category '{category['name']}'")
                
                # Click each unprocessed market one by one and capture odds data
                for market_index, market in enumerate(unprocessed_markets):
                    logger.info(f"Processing market {market_index + 1}/{len(unprocessed_markets)}: {market['name']}")
                    
                    # Mark this market as processed before clicking to avoid re-processing
                    self._mark_market_as_processed(market['name'])
                    
                    # Capture page state before clicking market
                    before_market_html = await self._capture_page_snapshot()
                    
                    # Click the market
                    await self._click_market(market)
                    await asyncio.sleep(2)  # Wait for AJAX to load
                    
                    # Capture page state after clicking market
                    after_market_html = await self._capture_page_snapshot()
                    
                    # Diff HTML to find new odds data
                    new_odds_data = await self._extract_odds_from_diff(before_market_html, after_market_html, market['name'])
                    
                    # Add to collected data
                    if new_odds_data:
                        all_odds_data[market['name']] = new_odds_data
                        logger.info(f"Captured odds data for market: {market['name']}")
                    else:
                        logger.warning(f"No odds data found for market: {market['name']}")
            
            logger.info(f"Successfully collected odds data for {len(all_odds_data)} unique markets using LLM-driven approach")
            logger.info(f"Total markets processed: {len(self.processed_markets)}")
            return all_odds_data
            
        except Exception as e:
            error_msg = f"LLM-driven odds scraping failed: {str(e)}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            import sys
            sys.exit(1)
        finally:
            await self._close_browser()
    
    async def _scrape_from_popup(self, popup) -> Dict[str, Any]:
        """Scrape odds data from a popup that contains market categories."""
        try:
            # Step 3: Get all market categories in the popup
            logger.info("Step 3: Getting market categories from popup...")
            market_categories = await self._get_market_categories(popup)
            logger.info(f"Found {len(market_categories)} market categories")
            
            # Step 4: Iterate through each category and click each market
            logger.info("Step 4: Iterating through categories and markets...")
            all_odds_data = {}
            
            for category_index, category in enumerate(market_categories):
                if category['name'] in self.processed_categories:
                    logger.info(f"Skipping already processed category: {category['name']}")
                    continue
                self.processed_categories.add(category['name'])
                logger.info(f"Processing category {category_index + 1}/{len(market_categories)}: {category['name']}")
                
                # Click on the category to expand it (if needed)
                await self._click_category(category)
                await asyncio.sleep(1)
                
                # Get all markets in this category
                markets_in_category = await self._get_markets_in_category(category)
                logger.info(f"Found {len(markets_in_category)} markets in category '{category['name']}'")
                
                # Click each market one by one and capture odds data
                for market_index, market in enumerate(markets_in_category):
                    logger.info(f"Processing market {market_index + 1}/{len(markets_in_category)}: {market['name']}")
                    
                    # Capture page state before clicking market
                    before_market_html = await self._capture_page_snapshot()
                    
                    # Click the market
                    await self._click_market(market)
                    await asyncio.sleep(2)  # Wait for AJAX to load
                    
                    # Capture page state after clicking market
                    after_market_html = await self._capture_page_snapshot()
                    
                    # Diff HTML to find new odds data
                    new_odds_data = await self._extract_odds_from_diff(before_market_html, after_market_html, market['name'])
                    
                    # Add to collected data
                    if new_odds_data:
                        all_odds_data[market['name']] = new_odds_data
                        logger.info(f"Captured odds data for market: {market['name']}")
                    else:
                        logger.warning(f"No odds data found for market: {market['name']}")
            
            logger.info(f"Successfully collected odds data for {len(all_odds_data)} markets from popup")
            return all_odds_data
            
        except Exception as e:
            logger.error(f"Error scraping from popup: {e}")
            return {}
    
    async def _scrape_from_inline_expansion(self) -> Dict[str, Any]:
        """Scrape odds data from inline expanded markets on the main page."""
        try:
            logger.info("Scraping from inline expanded markets...")
            
            # Wait a bit more for content to fully load
            await asyncio.sleep(2)
            
            # Find all market sections on the page
            market_sections = await self._find_market_sections()
            logger.info(f"Found {len(market_sections)} market sections on the page")
            
            all_odds_data = {}
            
            # Process each market section
            for section_index, section in enumerate(market_sections):
                if section['name'] in self.processed_categories:
                    logger.info(f"Skipping already processed section: {section['name']}")
                    continue
                self.processed_categories.add(section['name'])
                logger.info(f"Processing market section {section_index + 1}/{len(market_sections)}: {section['name']}")
                
                # Get all markets in this section
                markets_in_section = await self._get_markets_in_section(section)
                logger.info(f"Found {len(markets_in_section)} markets in section '{section['name']}'")
                
                # Click each market one by one and capture odds data
                for market_index, market in enumerate(markets_in_section):
                    logger.info(f"Processing market {market_index + 1}/{len(markets_in_section)}: {market['name']}")
                    
                    # Capture page state before clicking market
                    before_market_html = await self._capture_page_snapshot()
                    
                    # Click the market
                    await self._click_market(market)
                    await asyncio.sleep(2)  # Wait for AJAX to load
                    
                    # Capture page state after clicking market
                    after_market_html = await self._capture_page_snapshot()
                    
                    # Diff HTML to find new odds data
                    new_odds_data = await self._extract_odds_from_diff(before_market_html, after_market_html, market['name'])
                    
                    # Add to collected data
                    if new_odds_data:
                        all_odds_data[market['name']] = new_odds_data
                        logger.info(f"Captured odds data for market: {market['name']}")
                    else:
                        logger.warning(f"No odds data found for market: {market['name']}")
            
            logger.info(f"Successfully collected odds data for {len(all_odds_data)} markets from inline expansion")
            return all_odds_data
            
        except Exception as e:
            logger.error(f"Error scraping from inline expansion: {e}")
            return {}
    
    async def _find_market_sections(self) -> List[Dict[str, Any]]:
        """Find all market sections on the main page after expansion using LLM."""
        try:
            if not self.page:
                return []
            
            logger.info("Using LLM to find market sections on the page...")
            
            # Get the full page HTML for LLM analysis
            page_html = await self.page.content()
            
            user_goal = "find all market sections or containers that contain betting markets (like 'Vincente', 'Totale gol', 'Handicap Asiatico') on the page"
            previous_actions = "Expanded all markets and looking for market sections"
            
            # Use LLM to find market section selectors
            section_selector = self._llm_find_selector(page_html, user_goal, previous_actions)
            
            if not section_selector:
                logger.warning("LLM could not suggest a selector for market sections")
                return []
            
            # Use the LLM-suggested selector to find market sections
            sections = []
            elements = await self.page.query_selector_all(section_selector)
            
            for element in elements:
                try:
                    if await element.is_visible():
                        text = await element.text_content()
                        if text and any(keyword in text.lower() for keyword in ['market', 'mercato', 'odds', 'betting', 'vincente', 'totale gol', 'handicap', 'risultato', 'primo tempo', 'secondo tempo']):
                            # Extract a meaningful name for this section
                            section_name = self._extract_section_name(text)
                            sections.append({
                                'name': section_name,
                                'element': element
                            })
                            logger.info(f"Found market section: {section_name}")
                except Exception as e:
                    logger.warning(f"Error processing market section element: {e}")
                    continue
            
            # If no sections found with LLM selector, try a fallback approach
            if not sections:
                logger.info("No sections found with LLM selector, trying fallback approach...")
                sections = await self._find_market_sections_fallback()
            
            logger.info(f"Found {len(sections)} market sections using LLM")
            return sections
            
        except Exception as e:
            logger.error(f"Error finding market sections: {e}")
            return []
    
    async def _find_market_sections_fallback(self) -> List[Dict[str, Any]]:
        """Fallback method to find market sections when LLM approach fails."""
        try:
            if not self.page:
                return []
            
            # Use LLM to find container selectors dynamically
            page_html = await self.page.content()
            
            user_goal = "find all containers (divs, sections, articles) that might contain betting markets or odds data"
            previous_actions = "LLM fallback search for market sections"
            
            # Get LLM-suggested selectors for containers
            container_selector = self._llm_find_selector(page_html, user_goal, previous_actions)
            
            if not container_selector:
                logger.error("LLM could not suggest selectors for containers in fallback. No fallback mechanisms allowed.")
                raise Exception("LLM failed to suggest selectors for market sections - no fallback mechanisms allowed")
            
            elements = await self.page.query_selector_all(container_selector)
            logger.info(f"LLM fallback selector '{container_selector}' found {len(elements)} elements")
            
            sections = []
            for element in elements:
                try:
                    if await element.is_visible():
                        text = await element.text_content()
                        if text and text.strip():
                            # Let LLM determine if this contains market content
                            sections.append({
                                'name': text.strip()[:50] + '...' if len(text.strip()) > 50 else text.strip(),
                                'element': element
                            })
                except Exception as e:
                    continue
            
            # Apply LLM filtering to remove non-market sections
            sections = self._llm_filter_market_candidates(sections)
            logger.info(f"Found {len(sections)} market sections using LLM fallback selector")
            
            if not sections:
                logger.error("LLM fallback approach found no market sections. No additional fallback mechanisms allowed.")
                raise Exception("No market sections found with LLM fallback approach - no additional fallback mechanisms allowed")
            
            return sections
            
        except Exception as e:
            logger.error(f"Error in LLM fallback market section finding: {e}")
            raise
    
    def _llm_filter_market_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Use the LLM to filter a list of candidate market elements, returning only those that are actual betting markets.
        Each candidate should have at least a 'name' key.
        """
        if self.llm_agent is None or not candidates:
            return candidates
        
        # Use LLM to filter all candidates - no hard-coded filtering
        items = "\n".join([f"{i+1}. {c['name']}" for i, c in enumerate(candidates)])
        prompt = f"""
You are a football odds market extraction agent. Here is a list of button or container texts found on the page:

{items}

CRITICAL: You need to identify which of these are actual betting market CATEGORIES (container headers) vs individual markets or other content.

VALID MARKET CATEGORIES (CONTAINER HEADERS):
- "Esiti incontro" (Match Outcomes)
- "Risultato finale" (Final Result) 
- "Handicap Asiatico" (Asian Handicap)
- "Over/Under" (Total Goals)
- "Primo Tempo" (First Half)
- "Secondo Tempo" (Second Half)
- "Margine Vittoria" (Victory Margin)
- "Entrambe le Squadre Segnano" (Both Teams Score)
- "Corner" markets
- "Card" markets (Yellow/Red cards)

INDIVIDUAL MARKETS (REJECT THESE - they are NOT categories):
- "Vincente" (Winner) - this is an individual market option
- "Pareggio" (Draw) - this is an individual market option  
- "Sconfitta" (Loss) - this is an individual market option
- "Over 2.5" - this is an individual market option
- "Under 2.5" - this is an individual market option
- "1" (Home win) - this is an individual market option
- "X" (Draw) - this is an individual market option
- "2" (Away win) - this is an individual market option

INVALID (REJECT THESE):
- Text containing team names + odds (e.g., "MilanPareggioCremonese1.441.45...")
- Long strings with multiple decimal numbers
- Navigation elements, ads, or promotional text
- Individual odds values only
- Data rows with bookmaker names and odds
- Any text that looks like a data row rather than a category name

Return only the valid market category names as a JSON list. If none are valid, return an empty list [].
"""
        human_message = BaseMessage.make_user_message(
            role_name="WebCrawler",
            content=prompt
        )
        response = self.llm_agent.step(human_message)
        import json as _json
        try:
            filtered_names = _json.loads(response.msgs[0].content)
            if not isinstance(filtered_names, list):
                logger.warning("LLM did not return a list, using all candidates")
                return candidates
        except Exception as e:
            logger.warning(f"LLM did not return valid JSON: {e}, using all candidates")
            return candidates
        
        filtered = [c for c in candidates if c['name'] in filtered_names]
        logger.info(f"LLM filtered {len(filtered)} valid markets from {len(candidates)} candidates.")
        return filtered
    
    def _looks_like_data_row(self, text: str) -> bool:
        """Check if text looks like a data row with team names and odds."""
        # Check for patterns like "Team1Team2Odds1Odds2Odds3..."
        import re
        
        # Look for multiple decimal numbers (odds) in sequence
        odds_pattern = r'\d+\.\d+'
        odds_matches = re.findall(odds_pattern, text)
        
        # If there are many odds values, it's likely a data row
        if len(odds_matches) >= 3:
            return True
        
        # If it's just odds values (no meaningful text), it's a data row
        if len(odds_matches) >= 2 and len(re.sub(r'[\d\.\s]', '', text)) < 3:
            return True
        
        # Look for team name patterns (common Italian team names)
        team_patterns = [
            r'milan', r'inter', r'juventus', r'roma', r'lazio', r'napoli', 
            r'atalanta', r'fiorentina', r'torino', r'sassuolo', r'udinese',
            r'cremonese', r'lecce', r'salernitana', r'verona', r'spezia'
        ]
        
        text_lower = text.lower()
        team_count = sum(1 for pattern in team_patterns if re.search(pattern, text_lower))
        
        # If there are team names and odds, it's likely a data row
        if team_count >= 1 and len(odds_matches) >= 2:
            return True
        
        # If the text is very long and contains many numbers, it's likely a data row
        if len(text) > 50 and len(re.findall(r'\d', text)) > 5:
            return True
        
        return False
    
    def _looks_like_odds_only(self, text: str) -> bool:
        """Check if text contains only odds values."""
        import re
        
        # Remove common separators and check if it's mostly numbers
        cleaned = re.sub(r'[^\d\.]', '', text)
        if len(cleaned) > 0 and len(cleaned) / len(text) > 0.7:
            return True
        
        # Check if it's just a single odds value
        if re.match(r'^\d+\.\d+$', text.strip()):
            return True
        
        return False

    # Integrate into LLM-driven scraping loop
    # In _get_market_categories_with_selector and _get_markets_in_category_with_llm, filter the candidates before returning
    async def _get_market_categories_with_selector(self, container, category_selector: str) -> List[Dict[str, Any]]:
        """Get market categories using the LLM-suggested selector, then filter with LLM."""
        try:
            categories = []
            elements = await container.query_selector_all(category_selector)
            for element in elements:
                if await element.is_visible():
                    text = await element.text_content()
                    if text and text.strip():
                        categories.append({
                            'name': text.strip(),
                            'element': element
                        })
            # LLM filter
            categories = self._llm_filter_market_candidates(categories)
            logger.info(f"Found {len(categories)} market categories using LLM selector: {category_selector}")
            return categories
        except Exception as e:
            logger.error(f"Error getting market categories with selector {category_selector}: {e}")
            return []

    async def _get_markets_in_category_with_llm(self, category, container) -> List[Dict[str, Any]]:
        """Get markets within a category using LLM to suggest selectors, then filter with LLM."""
        try:
            if not category or not category.get('element'):
                return []
            category_html = await category['element'].evaluate('el => el.outerHTML')
            user_goal = f"find all individual market buttons (like 'Vincente', 'Totale gol', 'Handicap Asiatico') within the category '{category['name']}'"
            previous_actions = f"Found category: {category['name']}"
            market_selector = self._llm_find_selector(category_html, user_goal, previous_actions)
            if not market_selector:
                logger.warning(f"LLM could not suggest selector for markets in category '{category['name']}', using fallback")
                return await self._get_markets_in_category(category)
            markets = []
            elements = await category['element'].query_selector_all(market_selector)
            for element in elements:
                if await element.is_visible():
                    text = await element.text_content()
                    if text and text.strip() and len(text.strip()) > 2:
                        # Let LLM handle all filtering decisions - no hard-coded filtering
                        markets.append({
                            'name': text.strip(),
                            'element': element
                        })
            # LLM filter
            markets = self._llm_filter_market_candidates(markets)
            logger.info(f"Found {len(markets)} markets in category '{category['name']}' using LLM selector: {market_selector}")
            
            # If no markets found with LLM, fall back to manual detection
            if not markets:
                logger.warning(f"No markets found with LLM selector in category '{category['name']}', trying manual detection...")
                markets = await self._get_markets_in_category(category)
                if markets:
                    # Still apply LLM filtering to manual results
                    markets = self._llm_filter_market_candidates(markets)
                    logger.info(f"Found {len(markets)} markets in category '{category['name']}' using manual detection")
            
            return markets
        except Exception as e:
            logger.error(f"Error getting markets in category {category['name']} with LLM: {e}")
            # Fall back to manual detection on error
            logger.info(f"Falling back to manual detection for category '{category['name']}'")
            return await self._get_markets_in_category(category)
    
    def _is_market_already_processed(self, market_name: str) -> bool:
        """Check if a market has already been processed to avoid duplicates."""
        # Normalize the market name for comparison
        normalized_name = self._normalize_market_name(market_name)
        return normalized_name in self.processed_markets
    
    def _normalize_market_name(self, market_name: str) -> str:
        """Normalize market name for consistent duplicate detection."""
        # Convert to lowercase and remove extra whitespace
        normalized = market_name.lower().strip()
        # Remove common variations and extra characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def _mark_market_as_processed(self, market_name: str):
        """Mark a market as processed to avoid future duplicates."""
        normalized_name = self._normalize_market_name(market_name)
        self.processed_markets.add(normalized_name)
        logger.info(f"Marked market as processed: {market_name} (normalized: {normalized_name})")
    
    def _get_unprocessed_markets(self, markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out markets that have already been processed."""
        unprocessed = []
        for market in markets:
            market_name = market.get('name', '')
            if market_name and not self._is_market_already_processed(market_name):
                unprocessed.append(market)
            else:
                logger.info(f"Skipping already processed market: {market_name}")
        return unprocessed

    async def _get_markets_in_section(self, section) -> List[Dict[str, Any]]:
        """Get all markets within a section using LLM-driven approach."""
        try:
            if not section or not section.get('element'):
                return []
            
            logger.info(f"Using LLM to find markets in section: {section['name']}")
            
            # Get the section HTML for LLM analysis
            section_html = await section['element'].evaluate('el => el.outerHTML')
            
            user_goal = f"find all individual market buttons (like 'Vincente', 'Totale gol', 'Handicap Asiatico') within the section '{section['name']}'"
            previous_actions = f"Found section: {section['name']}"
            
            # Use LLM to find market selectors within the section
            market_selector = self._llm_find_selector(section_html, user_goal, previous_actions)
            
            if not market_selector:
                logger.warning(f"LLM could not suggest selector for markets in section '{section['name']}', using fallback")
                return await self._get_markets_in_section_fallback(section)
            
            markets = []
            elements = await section['element'].query_selector_all(market_selector)
            
            for element in elements:
                try:
                    if await element.is_visible():
                        text = await element.text_content()
                        if text and text.strip() and len(text.strip()) > 2:
                            # Let LLM handle all filtering decisions - no hard-coded filtering
                            markets.append({
                                'name': text.strip(),
                                'element': element
                            })
                            logger.info(f"Found market in section '{section['name']}': {text.strip()}")
                except Exception as e:
                    logger.warning(f"Error processing market element: {e}")
                    continue
            
            # Apply LLM filtering to remove non-market elements
            markets = self._llm_filter_market_candidates(markets)
            logger.info(f"Found {len(markets)} markets in section '{section['name']}' using LLM selector: {market_selector}")
            
            # If no markets found with LLM, fall back to manual detection
            if not markets:
                logger.warning(f"No markets found with LLM selector in section '{section['name']}', trying fallback...")
                markets = await self._get_markets_in_section_fallback(section)
            
            return markets
            
        except Exception as e:
            logger.error(f"Error getting markets in section {section['name']}: {e}")
            # Fall back to manual detection on error
            return await self._get_markets_in_section_fallback(section)
    
    async def _get_markets_in_section_fallback(self, section) -> List[Dict[str, Any]]:
        """Fallback method to find markets in a section when LLM approach fails."""
        try:
            if not section or not section.get('element'):
                return []
            
            # Use LLM to find clickable element selectors dynamically
            section_html = await section['element'].evaluate('el => el.outerHTML')
            
            user_goal = "find all clickable elements (buttons, links, divs) within this section that might be market buttons or betting controls"
            previous_actions = f"LLM fallback search for markets in section: {section['name']}"
            
            # Get LLM-suggested selectors for clickable elements
            clickable_selector = self._llm_find_selector(section_html, user_goal, previous_actions)
            
            if not clickable_selector:
                logger.error("LLM could not suggest selectors for clickable elements in fallback. No fallback mechanisms allowed.")
                raise Exception(f"LLM failed to suggest selectors for markets in section '{section['name']}' - no fallback mechanisms allowed")
            
            clickables = await section['element'].query_selector_all(clickable_selector)
            logger.info(f"LLM fallback selector '{clickable_selector}' found {len(clickables)} elements")
            
            markets = []
            for clickable in clickables:
                try:
                    if await clickable.is_visible():
                        text = await clickable.text_content()
                        if text and text.strip() and len(text.strip()) > 2:
                            # Let LLM handle all filtering decisions - no hard-coded filtering
                            markets.append({
                                'name': text.strip(),
                                'element': clickable
                            })
                            logger.info(f"Found potential market in section '{section['name']}' (LLM fallback): {text.strip()}")
                except Exception as e:
                    continue
            
            # Apply LLM filtering to remove non-market elements
            markets = self._llm_filter_market_candidates(markets)
            logger.info(f"Found {len(markets)} markets in section '{section['name']}' using LLM fallback selector")
            
            if not markets:
                logger.error(f"LLM fallback approach found no markets in section '{section['name']}'. No additional fallback mechanisms allowed.")
                raise Exception(f"No markets found in section '{section['name']}' with LLM fallback approach - no additional fallback mechanisms allowed")
            
            return markets
            
        except Exception as e:
            logger.error(f"Error in LLM fallback market finding for section {section['name']}: {e}")
            raise

    async def _find_all_markets_button(self):
        """Find the 'all markets' button on the page using LLM-driven approach with retries."""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not self.page:
                    logger.error("Page is not initialized")
                    return None
                
                logger.info(f"Using LLM to find 'all markets' button (attempt {retry_count + 1}/{max_retries})...")
                
                # Get the page HTML for LLM analysis
                page_html = await self.page.content()
                
                # Try different approaches for finding the button
                if retry_count == 0:
                    # First attempt: Look for the most specific "all markets" button
                    user_goal = "find the 'all markets' or 'tutti i mercati' button that expands all available betting markets. This is NOT an individual market button like 'Vincente', 'Pareggio', or 'Sconfitta' - it is a control button that reveals more market categories. Look for buttons with text like 'All Markets', 'Tutti i Mercati', 'Show More', 'Expand', or 'More Markets' that are typically in headers, toolbars, or navigation areas."
                    previous_actions = "Navigated to the betting odds page"
                elif retry_count == 1:
                    # Second attempt: Look for any expansion/control button
                    user_goal = "find any button that expands, shows more, or reveals additional betting markets. Look for buttons with text containing 'more', 'expand', 'show', 'all', 'tutti', 'mercati', 'markets', 'additional', 'extra' that are not individual betting options."
                    previous_actions = "First attempt to find 'all markets' button failed, trying broader search"
                else:
                    # Third attempt: Look for any clickable element that might expand markets
                    user_goal = "find any clickable element (button, link, or div) that might expand or show more betting markets. Look for elements with text suggesting expansion, more content, or additional options."
                    previous_actions = "Previous attempts failed, trying most generic approach"
                
                # Use LLM to find the all markets button selector
                button_selector = self._llm_find_selector(page_html, user_goal, previous_actions)
                
                if not button_selector:
                    logger.warning(f"LLM could not suggest a selector for 'all markets' button on attempt {retry_count + 1}")
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying in 2 seconds...")
                        await asyncio.sleep(2)
                        continue
                    else:
                        logger.error("All attempts to find 'all markets' button failed. Terminating.")
                        raise Exception("LLM could not suggest a selector for 'all markets' button after multiple attempts.")
                
                # Use the LLM-suggested selector to find the button
                element = await self.page.query_selector(button_selector)
                if element and await element.is_visible():
                    logger.info(f"Found 'all markets' button using LLM selector: {button_selector}")
                    return element
                
                logger.warning(f"Could not find 'all markets' button with LLM selector: {button_selector} on attempt {retry_count + 1}")
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying in 2 seconds...")
                    await asyncio.sleep(2)
                    continue
                else:
                    logger.error("All attempts to find 'all markets' button failed. Terminating.")
                    raise Exception(f"Could not find 'all markets' button with any LLM selector after {max_retries} attempts.")
                    
            except Exception as e:
                logger.error(f"Error finding all markets button on attempt {retry_count + 1}: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying in 2 seconds...")
                    await asyncio.sleep(2)
                    continue
                else:
                    raise e
        
        return None

    async def _find_markets_popup(self):
        """Find the markets popup that appears after clicking 'all markets' button using LLM only. No hardcoded fallback."""
        try:
            if not self.page:
                logger.error("Page is not initialized")
                return None
            logger.info("Using LLM to find markets popup...")
            page_html = await self.page.content()
            user_goal = "find the popup or container that contains market categories (like 'Vincente', 'Totale gol', 'Handicap Asiatico') that appeared after clicking 'all markets'"
            previous_actions = "Clicked 'all markets' button to expand markets"
            popup_selector = self._llm_find_selector(page_html, user_goal, previous_actions)
            if not popup_selector:
                logger.error("LLM could not suggest a selector for markets popup. Terminating.")
                raise Exception("LLM could not suggest a selector for markets popup.")
            element = await self.page.query_selector(popup_selector)
            if element and await element.is_visible():
                logger.info(f"Found markets popup using LLM selector: {popup_selector}")
                return element
            logger.error(f"Could not find markets popup with LLM selector: {popup_selector}. Terminating.")
            raise Exception(f"Could not find markets popup with LLM selector: {popup_selector}")
        except Exception as e:
            logger.error(f"Error finding markets popup: {e}")
            raise

    async def _get_market_categories(self, popup):
        """Get all market categories from the popup using LLM, with robust fallback if LLM fails or returns 0."""
        try:
            if not popup:
                logger.error("Popup is not provided")
                return []
            logger.info("Using LLM to find market categories in popup...")
            popup_html = await popup.evaluate('el => el.outerHTML')
            user_goal = "find all market category containers (like 'Esiti incontro', 'Risultato finale', 'Handicap Asiatico', 'Over/Under') within the markets popup. These are the main category sections that contain multiple individual markets within them. DO NOT select individual betting options or data rows with team names and odds - only select the category headers/containers. If you only see one market (like match winner), look for buttons or controls that can load additional markets or expand to show more market categories."
            previous_actions = "Found markets popup"
            category_selector = self._llm_find_selector(popup_html, user_goal, previous_actions)
            categories = []
            tried_selectors = set()
            if category_selector:
                tried_selectors.add(category_selector)
                elements = await popup.query_selector_all(category_selector)
                logger.info(f"Found {len(elements)} elements with selector: {category_selector}")
                for i, element in enumerate(elements):
                    try:
                        is_visible = await element.is_visible()
                        text = await element.text_content()
                        if is_visible and text and text.strip():
                            categories.append({'name': text.strip(), 'element': element})
                    except Exception as e:
                        logger.warning(f"Error processing category element {i}: {e}")
                        continue
                categories = self._llm_filter_market_candidates(categories)
                logger.info(f"Found {len(categories)} market categories using LLM")
                # Fallbacks if LLM returns nothing or 0 categories
                if not categories:
                    logger.warning(f"LLM selector '{category_selector}' returned 0 categories, dumping HTML for debugging")
                    await self._dump_html_for_debugging(popup_html, category_selector, "llm_selector_zero_categories")
            
            # If no categories found, try to find buttons that might load additional markets
            if not categories:
                logger.info("No market categories found, looking for buttons to load additional markets...")
                additional_markets_selector = self._llm_find_selector(popup_html, "find buttons or controls that can load additional markets, expand market categories, or show more betting options. Look for buttons with text like 'More Markets', 'Show All', 'Expand', 'Load More', or similar text that suggests loading additional content.", "Looking for additional markets buttons")
                
                if additional_markets_selector:
                    additional_elements = await popup.query_selector_all(additional_markets_selector)
                    logger.info(f"Found {len(additional_elements)} potential additional markets buttons")
                    for element in additional_elements:
                        try:
                            if await element.is_visible():
                                text = await element.text_content()
                                if text and text.strip():
                                    categories.append({'name': f"Additional Markets: {text.strip()}", 'element': element})
                        except Exception as e:
                            logger.warning(f"Error processing additional markets element: {e}")
                            continue
            
            # If no categories found with LLM, raise an error - no fallback mechanisms allowed
            if not categories:
                logger.error("LLM could not find any market categories. No fallback mechanisms allowed.")
                raise Exception("No market categories found - LLM must handle all selector generation")
            
            return categories
        except Exception as e:
            logger.error(f"Error getting market categories: {e}")
            raise


    async def _get_markets_in_category(self, category):
        """Get all markets within a category using LLM only. No hardcoded fallback."""
        try:
            if not category or not category.get('element'):
                logger.error("Category or category element is not provided")
                return []
            logger.info(f"Using LLM to find markets in category: {category['name']}")
            category_html = await category['element'].evaluate('el => el.outerHTML')
            user_goal = f"find all individual market buttons (like 'Vincente', 'Totale gol', 'Handicap Asiatico') within the category '{category['name']}'"
            previous_actions = f"Found category: {category['name']}"
            market_selector = self._llm_find_selector(category_html, user_goal, previous_actions)
            if not market_selector:
                logger.error(f"LLM could not suggest selector for markets in category '{category['name']}'. Terminating.")
                raise Exception(f"LLM could not suggest selector for markets in category '{category['name']}'.")
            
            # Debug: Log the selector and HTML content
            logger.info(f"Using selector: {market_selector}")
            logger.debug(f"Category HTML length: {len(category_html)}")
            
            markets = []
            elements = await category['element'].query_selector_all(market_selector)
            logger.info(f"Found {len(elements)} elements with selector: {market_selector}")
            
            for i, element in enumerate(elements):
                try:
                    is_visible = await element.is_visible()
                    text = await element.text_content()
                    logger.debug(f"Element {i}: visible={is_visible}, text='{text}'")
                    
                    if is_visible and text and text.strip() and len(text.strip()) > 2:
                        if not re.match(r'^\d+\.\d+$', text.strip()):
                            markets.append({
                                'name': text.strip(),
                                'element': element
                            })
                except Exception as e:
                    logger.warning(f"Error processing market element {i}: {e}")
                    continue
            
            markets = self._llm_filter_market_candidates(markets)
            logger.info(f"Found {len(markets)} markets in category '{category['name']}' using LLM selector: {market_selector}")
            
            if not markets:
                # Instead of terminating, try a more generic approach
                logger.warning(f"No markets found with LLM selector in category '{category['name']}'. Trying fallback approach...")
                return await self._get_markets_in_category_fallback(category)
            
            return markets
        except Exception as e:
            logger.error(f"Error getting markets in category {category['name']}: {e}")
            raise

    async def _get_markets_in_category_fallback(self, category):
        """Fallback method to find markets in a category using LLM-driven approach."""
        try:
            logger.info(f"Trying LLM-driven fallback approach for markets in category: {category['name']}")
            
            # Use LLM to find market selectors dynamically
            category_html = await category['element'].evaluate('el => el.outerHTML')
            
            user_goal = f"find all elements that could be individual market buttons or controls within the category '{category['name']}'. Look for any clickable elements that might represent individual markets."
            previous_actions = f"LLM fallback search for markets in category: {category['name']}"
            
            # Get LLM-suggested selectors for market elements
            market_selector = self._llm_find_selector(category_html, user_goal, previous_actions)
            
            if not market_selector:
                logger.error("LLM could not suggest selectors for markets in fallback. No fallback mechanisms allowed.")
                raise Exception(f"LLM failed to suggest selectors for markets in category '{category['name']}' - no fallback mechanisms allowed")
            
            # Use only the LLM-suggested selector - no hardcoded fallbacks
            elements = await category['element'].query_selector_all(market_selector)
            logger.info(f"LLM fallback selector '{market_selector}' found {len(elements)} elements")
            
            markets = []
            for element in elements:
                try:
                    if await element.is_visible():
                        text = await element.text_content()
                        if text and text.strip() and len(text.strip()) > 2:
                            # Let LLM handle all filtering decisions - no hard-coded filtering
                            markets.append({
                                'name': text.strip(),
                                'element': element
                            })
                except Exception as e:
                    continue
            
            # Apply LLM filtering to remove non-market elements
            markets = self._llm_filter_market_candidates(markets)
            logger.info(f"Found {len(markets)} markets in category '{category['name']}' using LLM fallback selector")
            
            if not markets:
                logger.error(f"LLM fallback approach found no markets in category '{category['name']}'. No additional fallback mechanisms allowed.")
                raise Exception(f"No markets found in category '{category['name']}' with LLM fallback approach - no additional fallback mechanisms allowed")
            
            return markets
            
        except Exception as e:
            logger.error(f"Error in LLM fallback markets for category {category['name']}: {e}")
            raise

    async def _click_category(self, category):
        """Click on a market category to expand it."""
        try:
            element = category['element']
            if element and await element.is_visible():
                await element.click()
                logger.info(f"Clicked category: {category['name']}")
                self._record_session_action("CLICK_CATEGORY", category['name'])
                await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"Error clicking category {category['name']}: {e}")
            self._record_session_action("ERROR_CLICK_CATEGORY", f"{category['name']}: {str(e)}")
    
    async def _click_market(self, market):
        """Click on a specific market to load its odds data."""
        try:
            element = market['element']
            if element and await element.is_visible():
                await element.click()
                logger.info(f"Clicked market: {market['name']}")
                self._record_session_action("CLICK_MARKET", market['name'])
        except Exception as e:
            logger.warning(f"Error clicking market {market['name']}: {e}")
            self._record_session_action("ERROR_CLICK_MARKET", f"{market['name']}: {str(e)}")
    
    async def _extract_odds_from_diff(self, before_html: str, after_html: str, market_name: str) -> Dict[str, Any]:
        """Extract odds data from HTML diff after clicking a market."""
        try:
            if not before_html or not after_html:
                return {}
            
            # Find changes between before and after HTML
            changes = await self._find_page_changes(before_html, after_html)
            
            # Extract structured odds data from the changes
            odds_data = {}
            
            for change in changes:
                # Parse the change to extract odds information
                odds_info = await self._parse_odds_from_change(change, market_name)
                if odds_info:
                    odds_data.update(odds_info)
            
            return odds_data
            
        except Exception as e:
            logger.error(f"Error extracting odds from diff for market {market_name}: {e}")
            return {}

    def _extract_section_name(self, text: str) -> str:
        """Extract a meaningful section name from text content."""
        try:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 3 and len(line) < 50:
                    # Remove common prefixes/suffixes
                    line = line.replace('Odds', '').replace('Quotes', '').strip()
                    if line:
                        return line
            return 'Unknown Section'
        except Exception:
            return 'Unknown Section'
    
    async def _extract_structured_odds_data(self, container_element) -> Dict[str, Any]:
        """Extract structured odds data from a container element, preserving table structure."""
        if not container_element:
            return {}
        
        try:
            # Get the container's text content to identify the market type
            container_text = await container_element.evaluate('el => el.innerText')
            
            # Skip if this looks like just a single row of odds data
            # A proper market should have multiple betting options or bookmakers
            import re
            odds_pattern = re.compile(r'\b\d+\.\d+\b')
            odds_matches = odds_pattern.findall(container_text)
            
            # If this looks like just a single row (e.g., "MilanPareggioCremonese1.441.451.42...")
            # skip it as it's not a market container
            if len(odds_matches) < 6 and len(container_text) < 100:
                # Check if this looks like a single betting row
                if any(team_name in container_text for team_name in ['Milan', 'Cremonese', 'Inter', 'Juventus', 'Roma', 'Lazio']):
                    # This looks like a single row, not a market container
                    return {}
            
            # Try to identify the market type from the text
            market_type = self._identify_market_type(container_text)
            
            # Look for table structure (tr, td elements)
            # Use LLM to find table element selectors dynamically
            container_html = await container_element.evaluate('el => el.outerHTML')
            
            table_user_goal = "find all table rows (tr elements) within this container that contain betting data"
            table_previous_actions = "Extracting structured odds data from container"
            table_selector = self._llm_find_selector(container_html, table_user_goal, table_previous_actions)
            
            if not table_selector:
                logger.warning("LLM could not suggest selectors for table rows, skipping table structure analysis")
                table_rows = []
            else:
                table_rows = await container_element.query_selector_all(table_selector)
            
            if table_rows and len(table_rows) > 1:  # Need at least 2 rows (header + data)
                # This is a table structure - extract rows and columns
                odds_table = []
                headers = []
                
                for i, row in enumerate(table_rows):
                    # Use LLM to find cell selectors dynamically
                    row_html = await row.evaluate('el => el.outerHTML')
                    
                    cell_user_goal = "find all table cells (td, th elements) within this table row"
                    cell_previous_actions = f"Extracting cells from table row {i+1}"
                    cell_selector = self._llm_find_selector(row_html, cell_user_goal, cell_previous_actions)
                    
                    if not cell_selector:
                        logger.warning("LLM could not suggest selectors for table cells, skipping this row")
                        continue
                    
                    cells = await row.query_selector_all(cell_selector)
                    
                    row_data = []
                    
                    for cell in cells:
                        cell_text = await cell.text_content()
                        if cell_text:
                            cell_text = cell_text.strip()
                            row_data.append(cell_text)
                    
                    if row_data:
                        if i == 0:  # First row might be headers
                            headers = row_data
                        else:
                            odds_table.append(row_data)
                
                # If we have a proper table structure with multiple rows, format it properly
                if odds_table and len(odds_table) > 0:
                    return {
                        'market_type': market_type,
                        'market_name': self._extract_market_name(container_text),
                        'structure': 'table',
                        'headers': headers,
                        'rows': odds_table,
                        'timestamp': time.time()
                    }
            
            # If no table structure, try to extract individual odds with context
            # Use LLM to find odds element selectors dynamically
            odds_user_goal = "find all elements that contain odds values or betting prices within this container"
            odds_previous_actions = "Extracting individual odds from container"
            odds_selector = self._llm_find_selector(container_html, odds_user_goal, odds_previous_actions)
            
            if not odds_selector:
                logger.warning("LLM could not suggest selectors for odds elements, skipping odds extraction")
                odds_elements = []
            else:
                odds_elements = await container_element.query_selector_all(odds_selector)
            
            odds_data = []
            
            for element in odds_elements:
                try:
                    text = await element.text_content()
                    if text and text.strip():
                        # Check if this looks like an odds value
                        if re.match(r'^\d+\.\d+$', text.strip()):
                            odds_data.append({
                                'odds': float(text.strip()),
                                'text': text.strip()
                            })
                except Exception:
                    continue
            
            # Only return odds_list if we have multiple odds (indicating a proper market)
            if odds_data and len(odds_data) > 3:
                return {
                    'market_type': market_type,
                    'market_name': self._extract_market_name(container_text),
                    'structure': 'odds_list',
                    'odds': odds_data,
                    'timestamp': time.time()
                }
            
            # Fallback: return as text content only if it looks substantial
            if len(container_text) > 50 and len(odds_matches) > 3:
                return {
                    'market_type': market_type,
                    'market_name': self._extract_market_name(container_text),
                    'structure': 'text',
                    'content': container_text,
                    'timestamp': time.time()
                }
            
            # If we get here, this doesn't look like a proper market container
            return {}
            
        except Exception as e:
            logger.error(f"Error extracting structured odds data: {e}")
            return {
                'market_type': 'unknown',
                'market_name': 'Unknown Market',
                'structure': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _extract_market_name(self, text: str) -> str:
        """Extract a meaningful market name from text content."""
        try:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 3 and len(line) < 50:
                    # Remove common prefixes/suffixes
                    line = line.replace('Odds', '').replace('Quotes', '').strip()
                    if line:
                        return line
            return 'Unknown Market'
        except Exception:
            return 'Unknown Market'
    
    def _llm_find_selector(self, html_content: str, user_goal: str, previous_actions: str) -> Optional[str]:
        """Use LLM to find the best CSS selector for a given goal."""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Create a dedicated agent for query selector functionality using flash model
                selector_agent = self._create_selector_agent()
                if selector_agent is None:
                    return None
                
                # Log brief summary instead of full content
                html_length = len(html_content)
                logger.info(f"Finding selector for goal: {user_goal} (HTML length: {html_length} chars)")
                
                # Clean HTML before sending to LLM
                cleaned_html = self._clean_html_for_llm(html_content)
                
                prompt = f"""HTML Content:
{cleaned_html}

You are a web scraping expert. Given this HTML content and user goal, suggest the BEST and MOST SPECIFIC CSS selector.

User Goal: {user_goal}
Previous Actions: {previous_actions}

CRITICAL REQUIREMENTS:
1. The selector MUST be specific enough to find ONLY the target elements
2. The selector MUST be valid CSS/Playwright syntax
3. The selector MUST work reliably across page refreshes
4. Prefer attributes like id, data-*, aria-* over generic class names
5. Avoid overly broad selectors like 'div' or 'button' alone
6. The selector MUST actually exist in the provided HTML content
7. DO NOT return the example selectors below - they are just examples to show the pattern
8. ALWAYS prefer :has-text() variants for text-based selectors

CRITICAL DISTINCTION FOR "ALL MARKETS" BUTTON:
When looking for the "all markets" or "tutti i mercati" button:
- You are looking for a button that EXPANDS or SHOWS MORE markets
- This is NOT an individual market button like "Vincente", "Pareggio", "Sconfitta"
- This is NOT a market category like "Esiti incontro", "Risultato finale"
- This button typically has text like "All Markets", "Tutti i Mercati", "Show More", "Expand", "More Markets"
- This button is usually separate from the actual betting markets
- This button is often in a header, toolbar, or navigation area
- This button is designed to reveal additional market categories, not individual betting options

AVOID CONFUSION WITH EXISTING MARKETS:
- Do NOT select buttons that are already visible individual markets (like "Vincente", "Pareggio", "Sconfitta")
- Do NOT select market category containers that are already expanded
- Do NOT select promotional or navigation elements
- The "all markets" button is typically a control button, not a betting option itself

IMPORTANT: For market category containers, you are looking for:
- MAIN CONTAINERS that hold multiple betting markets (like "Esiti incontro", "Risultato finale", "Handicap Asiatico")
- NOT individual market buttons (like "Vincente", "Pareggio", "Sconfitta")
- NOT promotional elements (like "Bonus€€", "Promozioni", "Offerte")
- NOT navigation elements (like "Menu", "Home", "Account")
- NOT utility buttons (like "Close", "Back", "Settings")

A market category container should:
- Contain multiple individual betting options
- Have a category title/header
- Be part of the main betting interface
- Not be promotional or navigational

SELECTOR PRIORITY (in order of preference):
1. ID selectors: '#specific-id'
2. Data attributes: '[data-testid="button"]', '[data-market="winner"]'
3. Aria attributes: '[aria-label="All Markets"]', '[role="button"]'
4. Specific class combinations: 'button.market-button.primary'
5. Text-based selectors: 'button:has-text("Vincente")' - ALWAYS use :has-text() for text matching
6. Position-based: 'div:nth-child(2) button'

TEXT-BASED SELECTOR RULES:
- ALWAYS use :has-text() for matching text content
- NEVER use :contains() (invalid syntax)
- Prefer exact text matches: 'button:has-text("Vincente")'
- For partial matches: 'button:has-text("Vincente", "exact")' for exact, 'button:has-text("Vincente", "partial")' for partial
- Text-based selectors are often the most reliable for dynamic content

EXAMPLES OF SELECTOR PATTERNS (DO NOT USE THESE EXACTLY - FIND SIMILAR ONES IN THE HTML):
- 'button[data-testid="all-markets"]' (look for data-testid attributes)
- '#markets-popup button' (look for ID attributes)
- 'button[aria-label="Winner market"]' (look for aria-label attributes)
- 'div[class*="market-categories"] button' (look for class names containing keywords)
- 'button:has-text("SuperDuperBet")' (look for buttons with specific text - ALWAYS use :has-text())
- 'button[data-market-type="winner"]' (look for data attributes)
- 'div[class*="betting-container"] button' (look for class names containing keywords)

INVALID EXAMPLES (DO NOT USE):
- 'button' (too generic)
- 'div' (too generic)
- 'button:contains("text")' (invalid syntax, use :has-text() instead)
- 'CLICK_BUTTON:text' (action command, not selector)
- '.class' (too generic class)
- 'button[class*="btn"]' (too generic class pattern)
- Selectors that don't exist in the HTML
- The exact example selectors above (they are just patterns)

ANALYSIS STEPS:
1. First, identify the exact target elements in the HTML
2. Look for unique identifiers (id, data-*, aria-*)
3. If no unique identifiers, find the most specific combination of attributes
4. For text-based selection, ALWAYS use :has-text() syntax
5. Test your selector mentally - would it find ONLY the target elements?
6. Ensure the selector is specific enough to be reliable
7. Verify the selector actually exists in the provided HTML
8. Use the example patterns above as inspiration, but find actual selectors in the HTML

IMPORTANT: 
- Only return selectors that you can see actually exist in the HTML content above
- Do not guess or make up selectors
- Do not return the example selectors from above - they are just patterns
- Look for similar patterns in the actual HTML and return those
- ALWAYS use :has-text() for text-based selectors, never :contains()
- DO NOT wrap selectors in backticks or any other formatting - return the raw selector only
- For market categories, focus on containers that hold multiple betting options, not individual buttons or promotional elements
- For "all markets" button, focus on expansion/control buttons, not individual betting options


Return ONLY the most specific, valid CSS selector that will reliably find the target elements. If no good selector can be determined, return 'NONE'."""
                
                human_message = BaseMessage.make_user_message(
                    role_name="WebCrawler",
                    content=prompt
                )
                
                response = selector_agent.step(human_message)
                selector = response.msgs[0].content.strip()
                
                # Clean up the response - remove any action command prefixes
                if selector.startswith('CLICK_BUTTON:'):
                    # Extract the button text and convert to a text-based selector
                    button_text = selector.split(':', 1)[1].strip('"')
                    return f'button:has-text("{button_text}")'
                
                # Fix invalid :contains() selectors
                if ':contains(' in selector:
                    selector = selector.replace(':contains(', ':has-text(')
                
                # Remove backticks if present (common LLM formatting issue)
                if selector.startswith('`') and selector.endswith('`'):
                    selector = selector[1:-1].strip()
                    logger.info(f"Removed backticks from selector: {selector}")
                
                if selector.upper() == 'NONE':
                    # Dump cleaned HTML for debugging when LLM returns NONE
                    self._dump_cleaned_html_for_debugging(cleaned_html, user_goal, previous_actions, "llm_returned_none")
                    return None
                
                # Enhanced validation that it looks like a CSS selector
                if not any(char in selector for char in ['[', ':', '.', '#', '>', ' ']):
                    logger.warning(f"LLM returned something that doesn't look like a CSS selector: {selector}")
                    # Dump cleaned HTML for debugging when LLM returns invalid selector
                    self._dump_cleaned_html_for_debugging(cleaned_html, user_goal, previous_actions, f"invalid_selector_{selector[:50]}")
                    return None
                
                # Additional validation for overly generic selectors
                overly_generic = ['button', 'div', 'span', 'a', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
                if selector.strip() in overly_generic:
                    logger.warning(f"LLM returned overly generic selector: {selector}")
                    # Dump cleaned HTML for debugging when LLM returns overly generic selector
                    self._dump_cleaned_html_for_debugging(cleaned_html, user_goal, previous_actions, f"overly_generic_{selector}")
                    return None
                
                # Log the selected selector for debugging
                logger.info(f"LLM suggested selector: {selector}")
                
                # Check if LLM returned one of the example selectors from the prompt
                example_selectors = [
                    'button[data-testid="all-markets"]',
                    '#markets-popup button',
                    'button[aria-label="Winner market"]',
                    'div[class*="market-categories"] button',
                    'button:has-text("SuperDuperBet")',
                    'button[data-market-type="winner"]',
                    'div[class*="betting-container"] button'
                ]
                
                if selector in example_selectors:
                    logger.warning(f"LLM returned example selector from prompt: {selector}")
                    logger.warning("This selector was just an example and likely doesn't exist in the HTML")
                    # Dump cleaned HTML for debugging when LLM returns example selector
                    self._dump_cleaned_html_for_debugging(cleaned_html, user_goal, previous_actions, f"example_selector_{selector}")
                    return None
                
                return selector
                
            except Exception as e:
                error_str = str(e).lower()
                if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str:
                    logger.warning(f"Rate limit ({retry_count + 1}/{max_retries}): {e}")
                    
                    # Record the rate limit error for the flash model
                    api_key_manager.record_rate_limit_error("gemini-2.5-flash")
                    
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying _llm_find_selector with new API key...")
                        time.sleep(2)  # Brief pause before retry
                        continue
                    else:
                        logger.error(f"Max retries reached for rate limit errors in _llm_find_selector")
                        # Dump cleaned HTML for debugging when max retries reached
                        self._dump_cleaned_html_for_debugging(cleaned_html, user_goal, previous_actions, "max_retries_rate_limit")
                        return None
                else:
                    logger.error(f"Error getting LLM selector: {e}")
                    # Dump cleaned HTML for debugging when error occurs
                    self._dump_cleaned_html_for_debugging(cleaned_html, user_goal, previous_actions, f"error_{str(e)[:50]}")
                    return None
        
        # Dump cleaned HTML for debugging when all retries exhausted
        self._dump_cleaned_html_for_debugging(cleaned_html, user_goal, previous_actions, "all_retries_exhausted")
        return None
    
    def _create_selector_agent(self) -> Optional[ChatAgent]:
        """Create a dedicated agent for query selector functionality using flash model."""
        try:
            # Set the correct API key for the flash model
            platform = api_key_manager._get_platform_from_model(Models.flash)
            api_keys = api_key_manager._get_api_keys_for_platform(platform)
            if api_keys:
                idx = api_key_manager.current_key_indices.get(platform, 0)
                os.environ[f"{platform.upper()}_API_KEY"] = api_keys[idx]
            
            # Create a model for the selector agent using flash
            model = ModelFactory.create(
                model_platform=ModelPlatformType.GEMINI,
                model_type=Models.flash,
            )
            
            # Create the selector agent with a focused system prompt
            selector_agent = ChatAgent(
                model=model,
                system_message=SELECTOR_AGENT_SYSTEM_PROMPT
            )
            
            logger.info("Selector agent created successfully with flash model")
            return selector_agent
            
        except Exception as e:
            logger.error(f"Failed to create selector agent: {str(e)}")
            return None
    
    def _validate_selector(self, selector: str) -> bool:
        """Validate that a selector is specific enough and likely to work."""
        if not selector or not selector.strip():
            return False
        
        selector = selector.strip()
        
        # Check for overly generic selectors
        overly_generic = ['button', 'div', 'span', 'a', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        if selector in overly_generic:
            logger.warning(f"Selector too generic: {selector}")
            return False
        
        # Check for basic CSS selector structure
        if not any(char in selector for char in ['[', ':', '.', '#', '>', ' ']):
            logger.warning(f"Selector doesn't look like valid CSS: {selector}")
            return False
        
        # Check for common invalid patterns
        invalid_patterns = [
            ':contains(',  # Should be :has-text()
            'CLICK_BUTTON:',  # Action command, not selector
            'click:',  # Action command, not selector
        ]
        
        for pattern in invalid_patterns:
            if pattern in selector:
                logger.warning(f"Selector contains invalid pattern '{pattern}': {selector}")
                return False
        
        # Prefer selectors with specific attributes
        specific_attributes = ['id=', 'data-', 'aria-', 'role=']
        has_specific_attr = any(attr in selector for attr in specific_attributes)
        
        if not has_specific_attr and len(selector) < 10:
            logger.warning(f"Selector may be too generic (no specific attributes): {selector}")
            # Don't return False here, just warn
        
        return True
    
    async def _find_element_with_selector(self, selector: str):
        """Find an element using the given selector, prioritizing containers over individual elements."""
        try:
            if not self.page:
                return None
            
            # Validate the selector first
            if not self._validate_selector(selector):
                logger.error(f"Invalid selector rejected: {selector}")
                return None
            
            # Try to find all matching elements
            elements = await self.page.query_selector_all(selector)
            if not elements:
                logger.warning(f"No elements found with selector: {selector}")
                # Dump HTML for debugging
                page_html = await self.page.content()
                await self._dump_html_for_debugging(page_html, selector, "no_elements_found")
                return None
            
            logger.info(f"Found {len(elements)} elements with selector: {selector}")
            
            # If only one element, return it
            if len(elements) == 1:
                element = elements[0]
                if await element.is_visible():
                    logger.info(f"Found single element with selector: {selector}")
                    return element
                else:
                    logger.warning(f"Single element found but not visible with selector: {selector}")
                    # Dump HTML for debugging
                    page_html = await self.page.content()
                    await self._dump_html_for_debugging(page_html, selector, "element_not_visible")
                    return None
            
            # If multiple elements, find the largest/most comprehensive one
            best_element = None
            max_complexity = 0
            
            for i, element in enumerate(elements):
                try:
                    if not await element.is_visible():
                        continue
                    
                    # Get the HTML content of this element
                    element_html = await element.evaluate('el => el.outerHTML')
                    
                    # Calculate complexity based on HTML size and structure
                    # Larger HTML usually means more content
                    html_size = len(element_html)
                    
                    # Count interactive elements (buttons, links, inputs) as they indicate functionality
                    interactive_elements = element_html.count('<button') + element_html.count('<a') + element_html.count('<input')
                    
                    # Count structural elements (div, span, etc.) as they indicate content organization
                    structural_elements = element_html.count('<div') + element_html.count('<span') + element_html.count('<section')
                    
                    # Calculate a complexity score
                    complexity_score = html_size + (interactive_elements * 10) + (structural_elements * 5)
                    
                    logger.debug(f"Element {i}: size={html_size}, interactive={interactive_elements}, structural={structural_elements}, score={complexity_score}")
                    
                    if complexity_score > max_complexity:
                        max_complexity = complexity_score
                        best_element = element
                        
                except Exception as e:
                    logger.debug(f"Error evaluating element {i}: {e}")
                    continue
            
            if best_element:
                logger.info(f"Selected best element with complexity score {max_complexity} using selector: {selector}")
                return best_element
            else:
                logger.warning(f"No suitable element found among {len(elements)} elements with selector: {selector}")
                # Dump HTML for debugging
                page_html = await self.page.content()
                await self._dump_html_for_debugging(page_html, selector, "no_suitable_element")
                return None
            
        except Exception as e:
            logger.error(f"Error finding element with selector {selector}: {e}")
            # Dump HTML for debugging
            try:
                if self.page:
                    page_html = await self.page.content()
                    await self._dump_html_for_debugging(page_html, selector, f"error_{str(e)[:50]}")
            except:
                pass
            return None
    
    async def _parse_odds_from_change(self, change: Dict[str, Any], market_name: str) -> Dict[str, Any]:
        """Parse odds information from a page change."""
        try:
            container_text = change.get('container_text', '')
            market_type = change.get('market_type', 'unknown')
            
            # Extract odds values from the text
            odds_pattern = re.compile(r'\b(\d+\.\d+)\b')
            odds_matches = odds_pattern.findall(container_text)
            
            if odds_matches:
                odds_data = {}
                for i, odds in enumerate(odds_matches):
                    odds_data[f"option_{i+1}"] = float(odds)
                
                return {
                    market_name: odds_data
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error parsing odds from change: {e}")
            return {}

    async def _wait_for_popup_to_appear(self, max_wait_time: int = 10) -> bool:
        """Wait for the markets popup to appear in the DOM after clicking the all markets button, using only LLM-driven selectors."""
        try:
            if not self.page:
                logger.error("Page is not initialized")
                return False

            logger.info("Waiting for markets popup to appear (LLM-driven only)...")
            start_time = time.time()
            initial_html = await self.page.content()

            while time.time() - start_time < max_wait_time:
                try:
                    current_html = await self.page.content()
                    # Use LLM to suggest a selector for the popup/modal
                    user_goal = "find the popup or container that contains market categories (like 'Vincente', 'Totale gol', 'Handicap Asiatico') that appeared after clicking 'all markets'"
                    previous_actions = "Clicked 'all markets' button to expand markets"
                    popup_selector = self._llm_find_selector(current_html, user_goal, previous_actions)
                    if not popup_selector:
                        logger.info("LLM could not suggest a selector for markets popup, retrying...")
                        await asyncio.sleep(1)
                        continue
                    element = await self.page.query_selector(popup_selector)
                    if element and await element.is_visible():
                        logger.info(f"Found markets popup using LLM selector: {popup_selector}")
                        return True
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.debug(f"Error during LLM-driven popup detection: {e}")
                    await asyncio.sleep(1)
                    continue
            logger.error("LLM could not find a visible markets popup within the wait time. No fallback or heuristics allowed.")
            return False
        except Exception as e:
            logger.error(f"Error waiting for popup (LLM-only): {e}")
            return False

    async def _analyze_html_diff_for_popup(self, before_html: str, after_html: str) -> bool:
        """Analyze HTML diff to detect if a popup with market content has appeared."""
        try:
            import difflib
            from bs4 import BeautifulSoup, Tag
            
            # Parse both HTMLs
            soup_before = BeautifulSoup(before_html, 'html.parser')
            soup_after = BeautifulSoup(after_html, 'html.parser')
            
            # Find elements that are new in the after HTML
            before_elements = set(str(el) for el in soup_before.find_all())
            after_elements = set(str(el) for el in soup_after.find_all())
            
            # Get new elements
            new_elements = after_elements - before_elements
            
            if not new_elements:
                await self._dump_html_for_debugging(after_html, "body")
                return False
            
            logger.info(f"Found {len(new_elements)} new HTML elements")
            
            # Look for popup-like elements in the new content
            popup_indicators = []
            market_keywords = ['vincente', 'pareggio', 'sconfitta', 'totale', 'handicap', 'over', 'under', 'mercati', 'markets', 'esiti', 'risultato', 'primo tempo', 'secondo tempo']
            
            for element_str in new_elements:
                # Parse the new element
                new_soup = BeautifulSoup(element_str, 'html.parser')
                
                # Check if this element looks like a popup/modal
                element = new_soup.find()
                if element and isinstance(element, Tag):
                    # Check for popup-like attributes
                    classes = element.get('class')
                    if classes is None:
                        class_str = ''
                    elif isinstance(classes, list):
                        class_str = ' '.join(classes).lower()
                    else:
                        class_str = str(classes).lower()
                    
                    id_attr = element.get('id')
                    if id_attr:
                        id_attr = str(id_attr).lower()
                    else:
                        id_attr = ''
                    
                    popup_indicators_found = []
                    if any(indicator in class_str for indicator in ['popup', 'modal', 'overlay', 'drawer', 'dialog']):
                        popup_indicators_found.append('popup_class')
                    if any(indicator in id_attr for indicator in ['popup', 'modal', 'overlay', 'drawer', 'dialog']):
                        popup_indicators_found.append('popup_id')
                    if element.get('role') == 'dialog':
                        popup_indicators_found.append('dialog_role')
                    
                    # Check if element contains market-related content
                    element_text = element.get_text().lower()
                    market_content_found = any(keyword in element_text for keyword in market_keywords)
                    
                    if popup_indicators_found and market_content_found:
                        logger.info(f"Found popup element with indicators: {popup_indicators_found}, market content: {market_content_found}")
                        return True
                    
                    # Also check for elements with significant market content even without obvious popup indicators
                    if market_content_found and len(element_text) > 100:  # Substantial market content
                        logger.info(f"Found substantial market content in new element: {element_text[:200]}...")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error analyzing HTML diff for popup: {e}")
            return False

    async def _wait_for_content_stabilization(self, max_wait_time: int = 5) -> None:
        """Wait for the page content to stabilize after clicking the all markets button."""
        try:
            if not self.page:
                logger.error("Page is not initialized")
                return
                
            logger.info("Waiting for page content to stabilize...")
            start_time = time.time()
            last_html = None
            stable_count = 0
            
            while time.time() - start_time < max_wait_time:
                current_html = await self.page.content()
                
                if last_html is not None:
                    if current_html == last_html:
                        stable_count += 1
                        if stable_count >= 3:  # Content is stable for 3 consecutive checks
                            logger.info("Page content has stabilized")
                            return
                    else:
                        stable_count = 0
                
                last_html = current_html
                await asyncio.sleep(0.5)
            
            logger.info("Content stabilization timeout reached")
            
        except Exception as e:
            logger.error(f"Error waiting for content stabilization: {e}")

    async def _wait_for_network_idle(self, max_wait_time: int = 5) -> None:
        """Wait for network requests to complete after clicking the all markets button."""
        try:
            if not self.page:
                logger.error("Page is not initialized")
                return
                
            logger.info("Waiting for network requests to complete...")
            
            # Wait for network to be idle (no requests for 500ms)
            try:
                await self.page.wait_for_load_state('networkidle', timeout=max_wait_time * 1000)
                logger.info("Network is idle")
            except Exception as e:
                logger.debug(f"Network idle timeout or error: {e}")
                # Continue anyway, as some sites may not reach networkidle state
            
        except Exception as e:
            logger.error(f"Error waiting for network idle: {e}")

    async def _dump_html_for_debugging(self, html_content: str, selector: str, context: str = ""):
        """Dump HTML content to a temporary file for debugging when selectors fail."""
        try:
            # Create logs directory if it doesn't exist
            logs_dir = base_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Create a temporary file with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"debug_html_{timestamp}_{context}.html"
            filepath = logs_dir / filename
            
            # Write HTML content with debugging info
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"<!-- Debug HTML dump for failed selector: {selector} -->\n")
                f.write(f"<!-- Context: {context} -->\n")
                f.write(f"<!-- Timestamp: {timestamp} -->\n")
                f.write(f"<!-- Selector that failed: {selector} -->\n")
                f.write("<!-- HTML content below: -->\n")
                f.write(html_content)
            
            logger.warning(f"HTML dumped to {filepath} for debugging failed selector: {selector}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to dump HTML for debugging: {e}")
            return None

    def _clean_html_for_llm(self, html_content: str) -> str:
        """Clean HTML content by removing JavaScript, CSS, images, base64 data, and other noise before sending to LLM."""
        try:
            from bs4 import BeautifulSoup, Tag
            from bs4.element import NavigableString
            import re
            
            # Parse the HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script tags (JavaScript)
            for script in soup.find_all('script'):
                script.decompose()
            
            # Remove style tags (CSS)
            for style in soup.find_all('style'):
                style.decompose()
            
            # Remove link tags (external CSS)
            for link in soup.find_all('link'):
                link.decompose()
            
            # Remove img tags (images) - but preserve alt text if it contains important info
            for img in soup.find_all('img'):
                if isinstance(img, Tag):
                    alt_text = str(img.get('alt', ''))
                    if alt_text and any(keyword in alt_text.lower() for keyword in ['market', 'mercato', 'expand', 'more', 'all', 'tutti']):
                        # Replace img with its alt text if it might be important
                        new_text = soup.new_string(f" {alt_text} ")
                        img.replace_with(new_text)
                    else:
                        img.decompose()
                else:
                    img.decompose()
            
            # Remove svg tags (SVG graphics) - but preserve text content
            for svg in soup.find_all('svg'):
                if isinstance(svg, Tag):
                    svg_text = svg.get_text(strip=True)
                    if svg_text and any(keyword in svg_text.lower() for keyword in ['market', 'mercato', 'expand', 'more', 'all', 'tutti']):
                        # Replace svg with its text content if it might be important
                        new_text = soup.new_string(f" {svg_text} ")
                        svg.replace_with(new_text)
                    else:
                        svg.decompose()
                else:
                    svg.decompose()
            
            # Remove canvas tags (canvas elements)
            for canvas in soup.find_all('canvas'):
                canvas.decompose()
            
            # Remove video and audio tags
            for media in soup.find_all(['video', 'audio']):
                media.decompose()
            
            # Remove iframe tags
            for iframe in soup.find_all('iframe'):
                iframe.decompose()
            
            # Remove meta tags (except important ones)
            for meta in soup.find_all('meta'):
                if isinstance(meta, Tag) and meta.get('name') not in ['viewport', 'description', 'keywords']:
                    meta.decompose()
            
            # Remove elements with base64 data in attributes (but be more careful)
            for element in soup.find_all():
                if isinstance(element, Tag) and element.attrs:
                    for attr_name, attr_value in element.attrs.items():
                        if isinstance(attr_value, str) and 'data:image' in attr_value:
                            # Remove base64 encoded images
                            element.decompose()
                            break
                        elif isinstance(attr_value, str) and len(attr_value) > 2000:  # Increased threshold
                            # Remove very long attribute values (likely base64 or encoded data)
                            # But don't remove if it's a button or interactive element
                            if element.name not in ['button', 'a', 'input', 'select']:
                                element.decompose()
                                break
            
            # Remove elements with very long text content (likely encoded data)
            for text_element in soup.find_all(text=True):
                if isinstance(text_element, NavigableString) and len(text_element.strip()) > 10000:  # Increased threshold
                    if text_element.parent:
                        text_element.parent.decompose()
            
            # Be more careful about removing empty elements - preserve buttons and interactive elements
            for element in soup.find_all():
                if isinstance(element, Tag) and element.name:
                    # Don't remove buttons, links, or interactive elements even if they appear empty
                    if element.name in ['button', 'a', 'input', 'select', 'label']:
                        continue
                    
                    # Don't remove elements with important attributes
                    if element.attrs:
                        important_attrs = ['onclick', 'data-testid', 'data-market', 'aria-label', 'role']
                        if any(attr in element.attrs for attr in important_attrs):
                            continue
                    
                    # Only remove if truly empty (no text, no children, no important attributes)
                    if element.get_text(strip=True) == '' and not element.find_all():
                        element.decompose()
            
            # Be more selective about noise elements - don't remove everything with "popup" or "modal"
            # Only remove obvious ads and promotional content
            noise_selectors = [
                '[class*="ad-"]', '[class*="banner-"]', '[class*="promo-"]',
                '[id*="ad-"]', '[id*="banner-"]', '[id*="promo-"]',
                '[data-testid*="ad-"]', '[data-testid*="banner-"]'
            ]
            
            for selector in noise_selectors:
                for element in soup.select(selector):
                    # Don't remove if it might contain market-related content
                    # Let LLM handle content analysis instead of hard-coded keywords
                    element_text = element.get_text(strip=True).lower()
                    # For now, preserve elements with substantial content
                    if len(element_text) > 20:  # Substantial content threshold
                        continue
                    element.decompose()
            
            # Clean up excessive whitespace
            cleaned_html = str(soup)
            
            # Remove excessive newlines and whitespace
            cleaned_html = re.sub(r'\n\s*\n', '\n', cleaned_html)
            cleaned_html = re.sub(r'\s+', ' ', cleaned_html)
            
            # Remove comments
            cleaned_html = re.sub(r'<!--.*?-->', '', cleaned_html, flags=re.DOTALL)
            
            logger.info(f"Cleaned HTML: {len(html_content)} -> {len(cleaned_html)} characters")
            return cleaned_html
            
        except Exception as e:
            logger.warning(f"Error cleaning HTML for LLM: {e}, returning original")
            return html_content

    def _dump_cleaned_html_for_debugging(self, cleaned_html: str, user_goal: str, previous_actions: str, context: str = ""):
        """Dump cleaned HTML content to a file for debugging when LLM fails to suggest selectors."""
        try:
            # Create logs directory if it doesn't exist
            logs_dir = base_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Create a temporary file with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"cleaned_html_debug_{timestamp}_{context}.html"
            filepath = logs_dir / filename
            
            # Write cleaned HTML content with debugging info
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"<!-- Debug cleaned HTML dump for failed LLM selector finding -->\n")
                f.write(f"<!-- Context: {context} -->\n")
                f.write(f"<!-- Timestamp: {timestamp} -->\n")
                f.write(f"<!-- User Goal: {user_goal} -->\n")
                f.write(f"<!-- Previous Actions: {previous_actions} -->\n")
                f.write("<!-- This is the cleaned HTML that was sent to the LLM: -->\n")
                f.write(cleaned_html)
            
            logger.warning(f"Cleaned HTML dumped to {filepath} for debugging failed LLM selector finding")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to dump cleaned HTML for debugging: {e}")
            return None

    async def _find_updated_elements(self, before_html: str, after_html: str) -> List[Dict[str, Any]]:
        """Find the specific elements that were added or modified between two HTML states."""
        try:
            import difflib
            from bs4 import BeautifulSoup, Tag
            
            # Parse both HTMLs
            soup_before = BeautifulSoup(before_html, 'html.parser')
            soup_after = BeautifulSoup(after_html, 'html.parser')
            
            # Get all elements from both states
            before_elements = {}
            after_elements = {}
            
            # Index elements by their content hash for comparison
            for element in soup_before.find_all():
                if isinstance(element, Tag):
                    content_hash = hash(element.get_text(strip=True))
                    before_elements[content_hash] = element
            
            for element in soup_after.find_all():
                if isinstance(element, Tag):
                    content_hash = hash(element.get_text(strip=True))
                    after_elements[content_hash] = element
            
            # Find new elements (in after but not in before)
            new_elements = []
            for content_hash, element in after_elements.items():
                if content_hash not in before_elements:
                    new_elements.append(element)
            
            # Find modified elements (same content hash but different attributes/structure)
            modified_elements = []
            for content_hash, after_element in after_elements.items():
                if content_hash in before_elements:
                    before_element = before_elements[content_hash]
                    if str(after_element) != str(before_element):
                        modified_elements.append(after_element)
            
            updated_elements = []
            # Remove hard-coded market keywords - let LLM handle all filtering
            
            # Analyze new elements
            for element in new_elements:
                element_info = await self._analyze_element_for_markets(element, "new")
                if element_info:
                    updated_elements.append(element_info)
            
            # Analyze modified elements
            for element in modified_elements:
                element_info = await self._analyze_element_for_markets(element, "modified")
                if element_info:
                    updated_elements.append(element_info)
            
            logger.info(f"Found {len(new_elements)} new elements and {len(modified_elements)} modified elements")
            return updated_elements
            
        except Exception as e:
            logger.error(f"Error finding updated elements: {e}")
            return []

    async def _analyze_element_for_markets(self, element: Any, element_type: str) -> Optional[Dict[str, Any]]:
        """Analyze a single element to determine if it contains market content."""
        try:
            # Get element attributes
            classes = element.get('class')
            if classes is None:
                class_str = ''
            elif isinstance(classes, list):
                class_str = ' '.join(classes).lower()
            else:
                class_str = str(classes).lower()
            
            id_attr = element.get('id')
            if id_attr:
                id_attr = str(id_attr).lower()
            else:
                id_attr = ''
            
            # Check for popup-like attributes
            popup_indicators = []
            if any(indicator in class_str for indicator in ['popup', 'modal', 'overlay', 'drawer', 'dialog']):
                popup_indicators.append('popup_class')
            if any(indicator in id_attr for indicator in ['popup', 'modal', 'overlay', 'drawer', 'dialog']):
                popup_indicators.append('popup_id')
            if element.get('role') == 'dialog':
                popup_indicators.append('dialog_role')
            
            # Check if element contains market-related content - let LLM handle this
            element_text = element.get_text().lower()
            # For now, assume substantial content might be market-related
            market_content_found = len(element_text) > 50  # Substantial content threshold
            
            # Determine element type
            element_tag = element.name if element.name else 'unknown'
            
            # Check if this looks like a substantial market container
            is_substantial = len(element_text) > 100 and market_content_found
            
            # Check if this looks like a popup/modal
            is_popup_like = len(popup_indicators) > 0
            
            # Check if this contains market content
            contains_markets = market_content_found and (is_substantial or is_popup_like)
            
            if contains_markets or is_popup_like:
                return {
                    'element_type': element_type,
                    'tag': element_tag,
                    'classes': class_str,
                    'id': id_attr,
                    'popup_indicators': popup_indicators,
                    'contains_markets': contains_markets,
                    'text_preview': element_text[:200] + '...' if len(element_text) > 200 else element_text,
                    'is_substantial': is_substantial,
                    'is_popup_like': is_popup_like
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing element for markets: {e}")
            return None


def create_scraping_agent(headless: bool = True, model_type: str = "gemini-2.5-flash-lite-preview-06-17") -> 'WebScrapingAgent':
    """Create a Playwright-based web scraping agent for autonomous dynamic odds market scraping."""
    return WebScrapingAgent(headless=headless, model_type=model_type)