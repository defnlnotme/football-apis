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
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from camel.logger import get_logger
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from playwright.async_api import async_playwright, Browser, Page
from prompts import ODDS_SYSTEM_PROMPT

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
        
        logger.warning(f"Rate limit error detected for platform: {platform}")
        
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
    
    def __init__(self, headless: bool = True, model_type: str = "gemini-2.5-flash-lite-preview-06-17"):
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
                system_message="""You are an intelligent web scraping assistant that analyzes web page content and determines the best actions to take for scraping betting odds markets.

Your task is to:
1. Analyze the current page state (buttons, content, structure)
2. Understand the user's goal (scraping all markets)
3. Determine the next best action to achieve that goal
4. Return a clear action command

IMPORTANT: For betting odds sites like oddschecker, you MUST click on market buttons to load the market data via AJAX. Look for buttons with text like:
- "Vincente" (Winner)
- "Totale gol - Under/Over" (Total Goals)
- "Handicap Asiatico" (Asian Handicap)
- "Margine Vittoria" (Victory Margin)
- "Risultato Esatto" (Exact Result)
- "Primo Tempo / Secondo Tempo" (First Half/Second Half)
- "Tutti i Mercati" (All Markets)

Available actions:
- CLICK_BUTTON:"button text" - Click a specific button (use exact button text)
- WAIT:seconds - Wait for specified seconds
- SCROLL:direction - Scroll up/down/left/right
- CHECK_CONTENT - Analyze if goal is achieved
- STOP - Stop if goal is achieved or no more actions needed

PRIORITY RULES:
1. ALWAYS click market buttons first - these load the actual odds data
2. Click "Tutti i Mercati" (All Markets) if available to expand all markets
3. Click individual market buttons one by one to load their data
4. Wait after each click for AJAX to load
5. Only stop when all market buttons have been clicked and data loaded

Always prioritize user safety and avoid clicking suspicious elements. Be thorough in your analysis."""
            )
            
            logger.info("LLM agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM agent: {e}")
            raise e
    
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
                system_message="""You are an intelligent web scraping assistant that analyzes web page content and determines the best actions to take for scraping betting odds markets.

Your task is to:
1. Analyze the current page state (buttons, content, structure)
2. Understand the user's goal (scraping all markets)
3. Determine the next best action to achieve that goal
4. Return a clear action command

IMPORTANT: For betting odds sites like oddschecker, you MUST click on market buttons to load the market data via AJAX. Look for buttons with text like:
- "Vincente" (Winner)
- "Totale gol - Under/Over" (Total Goals)
- "Handicap Asiatico" (Asian Handicap)
- "Margine Vittoria" (Victory Margin)
- "Risultato Esatto" (Exact Result)
- "Primo Tempo / Secondo Tempo" (First Half/Second Half)
- "Tutti i Mercati" (All Markets)

Available actions:
- CLICK_BUTTON:"button text" - Click a specific button (use exact button text)
- WAIT:seconds - Wait for specified seconds
- SCROLL:direction - Scroll up/down/left/right
- CHECK_CONTENT - Analyze if goal is achieved
- STOP - Stop if goal is achieved or no more actions needed

PRIORITY RULES:
1. ALWAYS click market buttons first - these load the actual odds data
2. Click "Tutti i Mercati" (All Markets) if available to expand all markets
3. Click individual market buttons one by one to load their data
4. Wait after each click for AJAX to load
5. Only stop when all market buttons have been clicked and data loaded

Always prioritize user safety and avoid clicking suspicious elements. Be thorough in your analysis."""
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
            
            # Look for new elements with odds in the after HTML
            for element in soup_after.find_all(['div', 'tr', 'td', 'span', 'button']):
                text = element.get_text(strip=True)
                if odds_pattern.search(text):
                    # Check if this element is new (not in before HTML)
                    element_str = str(element)
                    
                    # Simple check: if this exact HTML string is not in before HTML, it's new
                    if element_str not in before_html:
                        # Find the parent container that likely contains the full market data
                        container = element.find_parent(['div', 'section', 'article', 'tr'])
                        if not container:
                            container = element
                        
                        # Extract structured data from this container
                        change_data = {
                            'container_text': container.get_text(strip=True),
                            'market_type': self._identify_market_type_from_text(container.get_text(strip=True)),
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
            
            # Find all odds containers
            odds_containers = await self.page.query_selector_all('div[class*="market"], div[class*="odds"], div[class*="betting"], table, tr')
            
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
            for change in changes:
                # Look for containers that match the change
                containers = await self.page.query_selector_all('div, table, tr')
                
                for container in containers:
                    try:
                        container_text = await container.evaluate('el => el.innerText')
                        if change['container_text'] in container_text:
                            # Extract structured data from this container
                            structured_data = await self._extract_structured_odds_data(container)
                            
                            if structured_data and structured_data.get('structure') != 'text':
                                # Check if this market has already been processed
                                market_name = structured_data.get('market_name', 'Unknown Market')
                                if not self._is_market_already_processed(market_name):
                                    structured_data['source'] = 'page_diff'
                                    market_data.append(structured_data)
                                    self._mark_market_as_processed(market_name)
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
            button_selectors = [
                'button[class*="market"]',
                'button[class*="odds"]',
                'button[class*="betting"]',
                'a[class*="market"]',
                'a[class*="odds"]',
                'a[class*="betting"]',
                '[onclick*="market"]',
                '[onclick*="odds"]',
                '[onclick*="betting"]',
                'button[class*="_2lrskQ"]',
                'button[class*="market-button"]',
                'button[class*="odds-button"]',
                'button[class*="betting-button"]',
                'button',
                'a[role="button"]',
                '[onclick]',
                '[class*="clickable"]',
                '[class*="selectable"]',
                '[class*="expandable"]'
            ]
            
            float_regex = re.compile(r'^\d+(\.\d+)?$')
            market_keywords = [
                'market', 'odds', 'betting', 'over', 'under', 'total', 'gol', 'goal',
                'win', 'draw', 'lose', 'both', 'teams', 'score', 'corner', 'card',
                'assist', 'clean sheet', 'penalty', 'red card', 'yellow card',
                'vincente', 'totale gol', 'handicap', 'margine', 'risultato', 'primo tempo', 'secondo tempo', 'tutti i mercati'
            ]
            
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
                        is_market_related = any(keyword in text_lower for keyword in market_keywords)
                        is_market_button = is_market_related and not is_float
                        is_odds_button = is_float
                        
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
                            logger.info(f"Found market button: '{text_stripped}' (market)")
                        
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
            
            # Get all visible buttons and links
            buttons = await self.page.query_selector_all('button, a[role="button"], [onclick], a[href]')
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
            
            # Get all divs that might be market containers
            market_containers = await self.page.query_selector_all('div[class*="market"], div[class*="odds"], div[class*="betting"], div[class*="selection"], div[class*="price"]')
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
                
                # Prepare the prompt for the LLM
                prompt = f"""Current page state:
Title: {page_info.get('title', 'Unknown')}
URL: {page_info.get('url', 'Unknown')}
Body preview: {page_info.get('body_preview', 'No content')}

Available buttons:
{json.dumps(page_info.get('buttons', []), indent=2)}

Market containers found:
{json.dumps(page_info.get('market_containers', []), indent=2)}

Market buttons found:
{json.dumps(page_info.get('market_buttons', []), indent=2)}

Current market count: {page_info.get('current_market_count', 0)}
Collected markets: {page_info.get('collected_markets_count', 0)}

User goal: {instruction}

CRITICAL INSTRUCTIONS FOR ODDSCHECKER:
1. You MUST click market buttons to load odds data via AJAX
2. Look for buttons like "Vincente", "Totale gol - Under/Over", "Handicap Asiatico", etc.
3. Click "Tutti i Mercati" first if available to expand all markets
4. Then click each individual market button one by one
5. Wait after each click for AJAX to load
6. Only stop when all market buttons have been clicked

Based on the current page state and user goal, what is the next best action? 
Return only the action command (e.g., CLICK_BUTTON:"Vincente", CLICK_BUTTON:"Tutti i Mercati", WAIT:3, STOP, etc.)"""

                # Create the human message
                human_message = BaseMessage.make_user_message(
                    role_name="Human",
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
                    logger.warning(f"Rate limit error detected (attempt {retry_count + 1}/{max_retries}): {e}")
                    
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
        instruction_lower = instruction.lower()
        
        # Look for buttons that might expand markets
        if "markets" in instruction_lower or "expand" in instruction_lower:
            buttons = page_info.get('buttons', [])
            for button in buttons:
                button_text = button.get('text', '').lower()
                if any(keyword in button_text for keyword in ['all markets', 'expand', 'show all', 'more', 'markets']):
                    return f"CLICK_BUTTON:{button.get('text', '')}"
        
        # Look for close buttons
        if "popup" in instruction_lower or "close" in instruction_lower:
            buttons = page_info.get('buttons', [])
            for button in buttons:
                button_text = button.get('text', '').lower()
                if any(keyword in button_text for keyword in ['close', 'x', 'accept', 'ok', 'dismiss']):
                    return f"CLICK_BUTTON:{button.get('text', '')}"
        
        return "WAIT:2"
    
    async def _execute_action(self, action: str) -> bool:
        """Execute the action determined by the LLM."""
        if not self.page:
            return False
        
        try:
            action = action.strip()
            
            if action.startswith("CLICK_BUTTON:"):
                button_text = action.split(":", 1)[1].strip('"')
                logger.info(f"Attempting to click button: {button_text}")
                
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
                    return False
                
                return True
            
            elif action.startswith("WAIT:"):
                seconds = int(action.split(":", 1)[1])
                logger.info(f"Waiting for {seconds} seconds")
                await asyncio.sleep(seconds)
                return True
            
            elif action.startswith("SCROLL:"):
                direction = action.split(":", 1)[1].lower()
                logger.info(f"Scrolling {direction}")
                
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
                return True
            
            else:
                logger.warning(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            return False
    
    async def _autonomous_interaction(self, instruction: str, max_iterations: int = 15) -> List[Dict[str, Any]]:
        """Autonomously interact with the page based on the instruction."""
        logger.info(f"Starting autonomous interaction with instruction: {instruction}")
        
        # Reset collected markets and duplicate tracking for this interaction
        self.collected_markets = []
        self.processed_markets.clear()
        self.market_states = {}
        self.last_market_count = 0
        self.page_snapshots = []
        logger.info("Reset duplicate tracking for autonomous interaction")
        
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
                if any(keyword in button_text.lower() for keyword in ['all markets', 'tutti i mercati', 'all mercati']):
                    success = await self._execute_action(action)
                    if success:
                        all_markets_clicked = True
                        logger.info("Successfully clicked 'all markets' button")
                        break
            
            await asyncio.sleep(1)
        
        if not all_markets_clicked:
            logger.warning("Could not find 'all markets' button, proceeding with individual market buttons")
        
        # Phase 2: Click individual market buttons and capture data
        logger.info("Phase 2: Clicking individual market buttons and capturing data...")
        
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
                            
                            # Use diff-based capture with structured extraction
                            await self._capture_market_data_with_diff(before_html)
                            
                            await asyncio.sleep(2)
                            clicked_count += 1
                            
                            if clicked_count >= 3:  # Limit to 3 clicks
                                break
                except Exception as e:
                    logger.warning(f"Error manually clicking button: {e}")
                    continue
            
            logger.info(f"Manual clicking completed. Clicked {clicked_count} unprocessed buttons.")
            
        except Exception as e:
            logger.error(f"Error in manual market clicking: {e}")
    
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
            
            # Reset duplicate tracking for this scraping session
            self.processed_markets.clear()
            self.collected_markets = []
            logger.info("Reset duplicate tracking for new scraping session")
            
            await self._init_browser()
            
            # Ensure page is initialized
            if not self.page:
                raise Exception("Failed to initialize browser page")
            
            # Navigate to the URL
            await self.page.goto(url, wait_until='networkidle')
            logger.info("Successfully navigated to the page")
            
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
            
            # Wait for content to load
            await asyncio.sleep(3)
            
            # Step 2: Use LLM to find the popup or market container
            logger.info("Step 2: Using LLM to find markets popup or container...")
            after_html = await self._capture_page_snapshot()
            
            user_goal = f"scrape all betting odds markets for {team} vs {vs_team} in {competition}"
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
            
            user_goal = f"find all market categories (like 'Vincente', 'Totale gol', 'Handicap Asiatico') within the markets container"
            previous_actions = f"Found markets container using selector: {selector}"
            
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
            
            sections = []
            
            # Get page text for analysis
            page_text = await self.page.evaluate('() => document.body.innerText')
            logger.info(f"Page text preview: {page_text[:500]}...")
            
            # Look for any div that contains market-related text
            all_divs = await self.page.query_selector_all('div, section, article')
            logger.info(f"Found {len(all_divs)} total containers on page")
            
            for i, div in enumerate(all_divs[:50]):  # Check first 50 divs to avoid too much processing
                try:
                    if await div.is_visible():
                        text = await div.text_content()
                        if text and len(text.strip()) > 10:  # Only consider divs with substantial text
                            if any(keyword in text.lower() for keyword in ['market', 'mercato', 'odds', 'betting', 'vincente', 'totale gol', 'handicap', 'risultato', 'primo tempo', 'secondo tempo', '1', '2', 'x', 'over', 'under']):
                                section_name = f"Market Section {i+1}"
                                sections.append({
                                    'name': section_name,
                                    'element': div
                                })
                                logger.info(f"Found potential market section {i+1}: {text[:100]}...")
                except Exception:
                    continue
            
            return sections
            
        except Exception as e:
            logger.error(f"Error in fallback market section finding: {e}")
            return []
    
    def _llm_filter_market_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Use the LLM to filter a list of candidate market elements, returning only those that are actual betting markets.
        Each candidate should have at least a 'name' key.
        """
        if self.llm_agent is None or not candidates:
            return candidates
        items = "\n".join([f"{i+1}. {c['name']}" for i, c in enumerate(candidates)])
        prompt = f"""
You are a football odds market extraction agent. Here is a list of button or container texts found on the page:

{items}

Which of these are actual betting market categories or market buttons (not navigation, ads, or unrelated controls)? 
Return only the valid market names as a JSON list.
"""
        human_message = BaseMessage.make_user_message(
            role_name="Human",
            content=prompt
        )
        response = self.llm_agent.step(human_message)
        import json as _json
        try:
            filtered_names = _json.loads(response.msgs[0].content)
        except Exception:
            logger.warning("LLM did not return valid JSON, using all candidates.")
            return candidates
        filtered = [c for c in candidates if c['name'] in filtered_names]
        logger.info(f"LLM filtered {len(filtered)} valid markets from {len(candidates)} candidates.")
        return filtered

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
                        if not re.match(r'^\d+\.\d+$', text.strip()):
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
                            # Filter out odds values (decimal numbers) and very short text
                            if not re.match(r'^\d+\.\d+$', text.strip()):
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
            
            markets = []
            
            # Look for any clickable elements that might be markets
            clickables = await section['element'].query_selector_all('button, a, [onclick], [class*="clickable"]')
            
            for clickable in clickables:
                try:
                    if await clickable.is_visible():
                        text = await clickable.text_content()
                        if text and text.strip() and len(text.strip()) > 2:
                            if not re.match(r'^\d+\.\d+$', text.strip()):
                                # Check if this looks like a market name
                                if any(keyword in text.lower() for keyword in ['vincente', 'totale gol', 'handicap', 'risultato', 'primo tempo', 'secondo tempo', 'margine', 'both teams', 'corner', 'card', 'goal', 'assist', 'clean sheet', 'penalty']):
                                    markets.append({
                                        'name': text.strip(),
                                        'element': clickable
                                    })
                                    logger.info(f"Found market in section '{section['name']}' (fallback): {text.strip()}")
                except Exception:
                    continue
            
            logger.info(f"Found {len(markets)} markets in section '{section['name']}' using fallback")
            return markets
            
        except Exception as e:
            logger.error(f"Error in fallback market finding for section {section['name']}: {e}")
            return []

    async def _find_all_markets_button(self):
        """Find the 'all markets' button on the page using LLM-driven approach only. No hardcoded fallback."""
        try:
            if not self.page:
                logger.error("Page is not initialized")
                return None
            
            logger.info("Using LLM to find 'all markets' button...")
            # Get the page HTML for LLM analysis
            page_html = await self.page.content()
            user_goal = "find the 'all markets' or 'tutti i mercati' button that expands all available betting markets"
            previous_actions = "Navigated to the betting odds page"
            # Use LLM to find the all markets button selector
            button_selector = self._llm_find_selector(page_html, user_goal, previous_actions)
            if not button_selector:
                logger.error("LLM could not suggest a selector for 'all markets' button. Terminating.")
                raise Exception("LLM could not suggest a selector for 'all markets' button.")
            # Use the LLM-suggested selector to find the button
            element = await self.page.query_selector(button_selector)
            if element and await element.is_visible():
                logger.info(f"Found 'all markets' button using LLM selector: {button_selector}")
                return element
            logger.error(f"Could not find 'all markets' button with LLM selector: {button_selector}. Terminating.")
            raise Exception(f"Could not find 'all markets' button with LLM selector: {button_selector}")
        except Exception as e:
            logger.error(f"Error finding all markets button: {e}")
            raise

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
        """Get all market categories from the popup using LLM only. No hardcoded fallback."""
        try:
            if not popup:
                logger.error("Popup is not provided")
                return []
            logger.info("Using LLM to find market categories in popup...")
            popup_html = await popup.evaluate('el => el.outerHTML')
            user_goal = "find all market category buttons (like 'Vincente', 'Totale gol', 'Handicap Asiatico') within the markets popup"
            previous_actions = "Found markets popup"
            category_selector = self._llm_find_selector(popup_html, user_goal, previous_actions)
            if not category_selector:
                logger.error("LLM could not suggest a selector for market categories. Terminating.")
                raise Exception("LLM could not suggest a selector for market categories.")
            categories = []
            elements = await popup.query_selector_all(category_selector)
            for element in elements:
                try:
                    if await element.is_visible():
                        text = await element.text_content()
                        if text and text.strip():
                            categories.append({
                                'name': text.strip(),
                                'element': element
                            })
                except Exception as e:
                    logger.warning(f"Error processing category element: {e}")
                    continue
            categories = self._llm_filter_market_candidates(categories)
            logger.info(f"Found {len(categories)} market categories using LLM")
            if not categories:
                logger.error("No market categories found with LLM selector. Terminating.")
                raise Exception("No market categories found with LLM selector.")
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
            markets = []
            elements = await category['element'].query_selector_all(market_selector)
            for element in elements:
                try:
                    if await element.is_visible():
                        text = await element.text_content()
                        if text and text.strip() and len(text.strip()) > 2:
                            if not re.match(r'^\d+\.\d+$', text.strip()):
                                markets.append({
                                    'name': text.strip(),
                                    'element': element
                                })
                except Exception as e:
                    logger.warning(f"Error processing market element: {e}")
                    continue
            markets = self._llm_filter_market_candidates(markets)
            logger.info(f"Found {len(markets)} markets in category '{category['name']}' using LLM selector: {market_selector}")
            if not markets:
                logger.error(f"No markets found with LLM selector in category '{category['name']}'. Terminating.")
                raise Exception(f"No markets found with LLM selector in category '{category['name']}'.")
            return markets
        except Exception as e:
            logger.error(f"Error getting markets in category {category['name']}: {e}")
            raise

    async def _click_category(self, category):
        """Click on a market category to expand it."""
        try:
            element = category['element']
            if element and await element.is_visible():
                await element.click()
                logger.info(f"Clicked category: {category['name']}")
                await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"Error clicking category {category['name']}: {e}")
    
    async def _click_market(self, market):
        """Click on a specific market to load its odds data."""
        try:
            element = market['element']
            if element and await element.is_visible():
                await element.click()
                logger.info(f"Clicked market: {market['name']}")
        except Exception as e:
            logger.warning(f"Error clicking market {market['name']}: {e}")
    
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
            
            # Try to identify the market type from the text
            market_type = self._identify_market_type(container_text)
            
            # Look for table structure (tr, td elements)
            table_rows = await container_element.query_selector_all('tr')
            
            if table_rows:
                # This is a table structure - extract rows and columns
                odds_table = []
                headers = []
                
                for i, row in enumerate(table_rows):
                    cells = await row.query_selector_all('td, th')
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
                
                # If we have a table structure, format it properly
                if odds_table:
                    return {
                        'market_type': market_type,
                        'market_name': self._extract_market_name(container_text),
                        'structure': 'table',
                        'headers': headers,
                        'rows': odds_table,
                        'timestamp': time.time()
                    }
            
            # If no table structure, try to extract individual odds with context
            odds_elements = await container_element.query_selector_all('[class*="odds"], [class*="price"], [class*="bet"], span, div')
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
            
            if odds_data:
                return {
                    'market_type': market_type,
                    'market_name': self._extract_market_name(container_text),
                    'structure': 'odds_list',
                    'odds': odds_data,
                    'timestamp': time.time()
                }
            
            # Fallback: return as text content
            return {
                'market_type': market_type,
                'market_name': self._extract_market_name(container_text),
                'structure': 'text',
                'content': container_text,
                'timestamp': time.time()
            }
            
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
                if self.llm_agent is None:
                    return None
                
                # Log brief summary instead of full content
                html_length = len(html_content)
                logger.info(f"Finding selector for goal: {user_goal} (HTML length: {html_length} chars)")
                
                prompt = f"""HTML Content:
{html_content}

You are a web scraping expert. Given this HTML content and user goal, suggest the best CSS selector.

User Goal: {user_goal}
Previous Actions: {previous_actions}

IMPORTANT: Return ONLY a valid CSS selector that would best find the target elements. 
Valid examples:
- 'button[class*="market"]' (button with class containing "market")
- 'div[class*="popup"]' (div with class containing "popup")
- 'button:has-text("Vincente")' (Playwright selector foelec button containing text)
- 'a[href*="market"]' (link with href containing "market")
- '[class*="market-categories"]' (any element with class containing "market-categories")

Do NOT return:
- Action commands like "CLICK_BUTTON:Vincente"
- Invalid selectors like "button:contains()" (use "button:has-text()" instead)
- Just text without selectors

If no good selector can be determined, return 'NONE'."""
                
                human_message = BaseMessage.make_user_message(
                    role_name="Human",
                    content=prompt
                )
                
                response = self.llm_agent.step(human_message)
                selector = response.msgs[0].content.strip()
                
                # Clean up the response - remove any action command prefixes
                if selector.startswith('CLICK_BUTTON:'):
                    # Extract the button text and convert to a text-based selector
                    button_text = selector.split(':', 1)[1].strip('"')
                    return f'button:has-text("{button_text}")'
                
                # Fix invalid :contains() selectors
                if ':contains(' in selector:
                    selector = selector.replace(':contains(', ':has-text(')
                
                if selector.upper() == 'NONE':
                    return None
                
                # Validate that it looks like a CSS selector
                if not any(char in selector for char in ['[', ':', '.', '#', '>', ' ']):
                    logger.warning(f"LLM returned something that doesn't look like a CSS selector: {selector}")
                    return None
                
                return selector
                
            except Exception as e:
                error_str = str(e).lower()
                if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str:
                    logger.warning(f"Rate limit error detected in _llm_find_selector (attempt {retry_count + 1}/{max_retries}): {e}")
                    
                    # Record the rate limit error
                    api_key_manager.record_rate_limit_error(self.model_type)
                    
                    # Reinitialize the agent with the new API key
                    self._reinit_llm_agent_with_new_key()
                    
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying _llm_find_selector with new API key...")
                        time.sleep(2)  # Brief pause before retry
                        continue
                    else:
                        logger.error(f"Max retries reached for rate limit errors in _llm_find_selector")
                        return None
                else:
                    logger.error(f"Error getting LLM selector: {e}")
                    return None
        
        return None
    
    async def _find_element_with_selector(self, selector: str):
        """Find an element using the given selector."""
        try:
            if not self.page:
                return None
            
            element = await self.page.query_selector(selector)
            if element and await element.is_visible():
                return element
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding element with selector {selector}: {e}")
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


def create_scraping_agent(headless: bool = True, model_type: str = "gemini-2.5-flash-lite-preview-06-17") -> 'WebScrapingAgent':
    """Create a Playwright-based web scraping agent for autonomous dynamic odds market scraping."""
    return WebScrapingAgent(headless=headless, model_type=model_type) 