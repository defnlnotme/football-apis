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
from prompts import SELECTOR_AGENT_SYSTEM_PROMPT, Models, MARKET_DATA_EXTRACTION_SYSTEM_PROMPT

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
        self.market_data_extraction_agent: Optional[ChatAgent] = None
        
        # Data collection for markets with diff tracking
        self.collected_markets = []  # Store individual market data
        self.processed_markets = set()  # Track which markets have been processed to avoid duplicates
        self.market_states = {}     # Track which markets were processed
        self.last_market_count = 0  # Track number of markets found
        self.page_snapshots = []    # Store page snapshots for diffing
        self.current_page_html = "" # Current page HTML for diffing
        self.processed_categories = set()  # Track processed market categories/sections
        
        # State tracking for session management
        self.session_actions = []  # Track all actions taken in this session
        self.current_phase = "initial"  # Track current phase: initial, markets_expanded, collecting_markets
        
        # Initialize the market data extraction agent
        self._init_market_data_extraction_agent()
        
    def _init_market_data_extraction_agent(self):
        """Initialize the market data extraction agent with the dedicated system prompt."""
        try:
            model = ModelFactory.create(
                model_platform=ModelPlatformType.GEMINI,
                model_type=self.model_type
            )
            self.market_data_extraction_agent = ChatAgent(
                model=model,
                system_message=MARKET_DATA_EXTRACTION_SYSTEM_PROMPT
            )
            logger.info("Market data extraction agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize market data extraction agent: {e}")
            self.market_data_extraction_agent = None
    
    def _reset_session_state(self):
        """Reset session state for a new scraping session."""
        self.session_actions = []
        self.current_phase = "initial"
        self.collected_markets = []
        self.processed_markets.clear()
        self.market_states = {}
        self.last_market_count = 0
        self.page_snapshots = []
        self.current_page_html = ""
        self.processed_categories.clear()
        # Track previously extracted markets and their odds to avoid duplication
        self.previously_extracted_markets = {}
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
    
    def _identify_market_type_from_text(self, text: str) -> str:
        """Identify market type from text content (for use in page diffing)."""
        # approach - let the LLM handle market type identification
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
        # Use the same approach as above
        return self._identify_market_type_from_text(text)
    
    async def _get_market_buttons(self) -> List[Dict[str, Any]]:
        """Find all market buttons on the page using LLM."""
        try:
            if not self.page:
                return []
            
            logger.info("Finding market buttons using LLM...")
            
            # Get the full page HTML for LLM analysis
            page_html = await self.page.content()
            
            user_goal = "find all clickable market buttons on this betting page"
            previous_actions = "Finding market buttons for odds extraction"
            
            # Use LLM to find market button selectors
            market_selectors = self._llm_find_selector(page_html, user_goal, previous_actions)
            
            if not market_selectors:
                logger.warning("LLM could not suggest market button selectors")
                return []
            
            # Use the LLM-suggested selector to find market buttons
            market_buttons = []
            for market_selector in market_selectors:
                elements = await self.page.query_selector_all(market_selector)
                
                for element in elements:
                    try:
                        if await element.is_visible():
                            text = await element.text_content()
                            if text and text.strip():
                                # Exclude elements whose text is a number (int or float)
                                stripped_text = text.strip()
                                try:
                                    float_val = float(stripped_text.replace(',', '.'))
                                    continue
                                except ValueError:
                                    pass
                                # Check if this looks like a market button
                                if any(keyword in text.lower() for keyword in ['market', 'mercato', 'odds', 'betting', 'vincente', 'totale gol', 'handicap', 'risultato', 'primo tempo', 'secondo tempo']):
                                    market_buttons.append({
                                        'text': text.strip(),
                                        'element': element
                                    })
                                    logger.info(f"Found market button: {text.strip()}")
                    except Exception as e:
                        logger.warning(f"Error processing market button element: {e}")
                        continue
            
            # If no market buttons found with LLM selector, try manual approach
            if not market_buttons:
                logger.info("No market buttons found with LLM selector, trying manual approach...")
                market_buttons = await self._try_manual_market_clicking()
            
            logger.info(f"Found {len(market_buttons)} market buttons using LLM")
            return market_buttons
            
        except Exception as e:
            logger.error(f"Error finding market buttons: {e}")
            return []

    async def _get_page_info(self) -> Dict[str, Any]:
        """Get comprehensive page information for LLM analysis."""
        try:
            if not self.page:
                return {}
            
            # Get page title
            title = await self.page.title()
            
            # Get page URL
            url = self.page.url
            
            # Get page HTML
            html_content = await self.page.content()
            
            # Get visible text content
            text_content = await self.page.evaluate('() => document.body.innerText')
            
            # Get all clickable elements
            clickable_elements = await self.page.query_selector_all('button, a, [role="button"], [onclick], [data-action]')
            clickable_texts = []
            
            for element in clickable_elements[:20]:  # Limit to first 20 for performance
                try:
                    if await element.is_visible():
                        text = await element.text_content()
                        if text and text.strip():
                            clickable_texts.append(text.strip())
                except Exception:
                    continue
            
            return {
                'title': title,
                'url': url,
                'html_length': len(html_content),
                'text_content_preview': text_content[:1000] if text_content else '',
                'clickable_elements': clickable_texts[:10],  # Limit to first 10
                'session_actions': self._get_session_summary()
            }
            
        except Exception as e:
            logger.error(f"Error getting page info: {e}")
            return {}

    def _execute_llm_instruction(self, instruction: str, page_info: Dict[str, Any]) -> str:
        """Execute an LLM instruction with page context."""
        try:
            if not self.llm_agent:
                logger.error("LLM agent not initialized")
                return ""
            
            # Build context from page info
            context = f"""
Page Information:
- Title: {page_info.get('title', 'Unknown')}
- URL: {page_info.get('url', 'Unknown')}
- HTML Length: {page_info.get('html_length', 0)} characters
- Clickable Elements: {', '.join(page_info.get('clickable_elements', []))}
- Session Actions: {page_info.get('session_actions', 'None')}

Text Content Preview:
{page_info.get('text_content_preview', 'No text content available')}

Instruction: {instruction}
"""
            
            # Create the human message
            human_message = BaseMessage.make_user_message(
                role_name="WebScraper",
                content=context
            )
            
            # Get response from the agent
            response = self.llm_agent.step(human_message)
            
            if response.msgs:
                return response.msgs[0].content.strip()
            else:
                logger.warning("No response received from LLM agent")
                return ""
                
        except Exception as e:
            logger.error(f"Error executing LLM instruction: {e}")
            return ""

    def _fallback_heuristics(self, instruction: str, page_info: Dict[str, Any]) -> str:
        """Fallback heuristics when LLM fails."""
        try:
            clickable_elements = page_info.get('clickable_elements', [])
            text_content = page_info.get('text_content_preview', '').lower()
            
            # Simple heuristics based on common patterns
            if 'market' in instruction.lower() or 'button' in instruction.lower():
                # Look for market-related buttons
                market_keywords = ['market', 'mercato', 'odds', 'betting', 'vincente', 'totale gol', 'handicap']
                for element in clickable_elements:
                    if any(keyword in element.lower() for keyword in market_keywords):
                        return f"Found potential market button: {element}"
            
            return "No specific action found using heuristics"
            
        except Exception as e:
            logger.error(f"Error in fallback heuristics: {e}")
            return "Error in fallback heuristics"

    async def _execute_action(self, action: str) -> bool:
        """Execute a specific action on the page."""
        try:
            if not self.page:
                logger.error("Page is not initialized")
                return False
            
            # Parse the action
            if action.startswith("click:"):
                # Extract button text to click
                button_text = action[6:].strip()
                
                # Find and click the button
                clickable_elements = await self.page.query_selector_all('button, a, [role="button"]')
                
                for element in clickable_elements:
                    try:
                        if await element.is_visible():
                            text = await element.text_content()
                            if text and button_text.lower() in text.lower():
                                await element.click()
                                logger.info(f"Clicked button: {text.strip()}")
                                self._record_session_action("CLICK", text.strip())
                                return True
                    except Exception as e:
                        logger.warning(f"Error clicking element: {e}")
                        continue
                
                logger.warning(f"Could not find button to click: {button_text}")
                return False
                
            elif action.startswith("wait:"):
                # Extract wait time
                try:
                    wait_time = float(action[5:].strip())
                    await asyncio.sleep(wait_time)
                    logger.info(f"Waited for {wait_time} seconds")
                    return True
                except ValueError:
                    logger.error(f"Invalid wait time: {action}")
                    return False
                    
            else:
                logger.warning(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return False

    async def _try_manual_market_clicking(self):
        """Try manual market clicking as a fallback."""
        try:
            logger.info("Trying manual market clicking...")
            
            if not self.page:
                logger.error("Page is not initialized")
                return []
            
            # Get the page HTML
            page_html = await self.page.content()
            
            # Use LLM to find market button selectors
            user_goal = "find all clickable market buttons on this betting page"
            previous_actions = "Manual market clicking fallback"
            
            market_selectors = self._llm_find_selector(page_html, user_goal, previous_actions)
            
            if not market_selectors:
                logger.warning("LLM could not suggest market button selectors")
                return []
            
            # Find all market buttons using the selector
            market_buttons = []
            for market_selector in market_selectors:
                market_elements = await self.page.query_selector_all(market_selector)
                for element in market_elements:
                    try:
                        if await element.is_visible():
                            text = await element.text_content()
                            if text and text.strip():
                                # Exclude elements whose text is a number (int or float)
                                stripped_text = text.strip()
                                try:
                                    float_val = float(stripped_text.replace(',', '.'))
                                    continue
                                except ValueError:
                                    pass
                                market_buttons.append({
                                    'text': text.strip(),
                                    'element': element
                                })
                    except Exception as e:
                        logger.warning(f"Error processing market element: {e}")
                        continue
            
            logger.info(f"Found {len(market_buttons)} market buttons via manual clicking")
            return market_buttons
            
        except Exception as e:
            logger.error(f"Error in manual market clicking: {e}")
            self._record_session_action("ERROR_MANUAL_CLICKING", str(e))
            return []

    async def scrape_odds_markets(self, url: str, team: str, vs_team: str, competition: str) -> List[Dict[str, Any]]:
        """
        Odds market scraping:
        1. Navigate to URL
        2. Use LLM to find all market buttons on the page
        3. Click each market button to load odds data
        4. Extract odds data from the page after each click
        5. Return list of market data
        
        Args:
            url: The URL to scrape
            team: The home team name
            vs_team: The away team name  
            competition: The competition name
            
        Returns:
            List of market data dictionaries
            
        Raises:
            Exception: If scraping fails, this will terminate the script
        """
        try:
            logger.info(f"Starting odds market scraping for {team} vs {vs_team} in {competition}")
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
            
            # Wait for page to fully load
            await asyncio.sleep(3)
            
            # Step 1: Find all market buttons on the page
            logger.info("Step 1: Finding all market buttons on the page...")
            market_buttons = await self._get_market_buttons()
            logger.info(f"Found {len(market_buttons)} potential market buttons")
            
            if not market_buttons:
                logger.warning("No market buttons found on the page")
                return []
            
            # Step 2: Click each market button and extract data
            logger.info("Step 2: Processing market buttons...")
            all_markets_data = []
            
            for button_index, button in enumerate(market_buttons):
                try:
                    button_text = button.get('text', 'Unknown Market')
                    element = button.get('element')
                    
                    if not element or not await element.is_visible():
                        logger.warning(f"Skipping invisible button: {button_text}")
                        continue
                    
                    logger.info(f"Processing market button {button_index + 1}/{len(market_buttons)}: {button_text}")
                    
                    # Click the market button
                    await element.click()
                    logger.info(f"Clicked market button: {button_text}")
                    
                    # Wait for page to load new data
                    await self.page.wait_for_load_state('networkidle')
                    
                    # Capture the page HTML after clicking
                    page_html = await self.page.content()
                    
                    # Extract market data from the page HTML
                    market_data = await self._extract_market_data_from_html(page_html, button_text)
                    
                    if market_data:
                        all_markets_data.append(market_data)
                        logger.info(f"Successfully extracted data for market: {button_text}")
                    else:
                        logger.warning(f"No data found for market: {button_text}")
                        
                except Exception as e:
                    logger.error(f"Error processing market button {button_text}: {e}")
                    continue
            
            logger.info(f"Successfully collected data for {len(all_markets_data)} markets")
            return all_markets_data
            
        except Exception as e:
            error_msg = f"odds scraping failed: {str(e)}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            import sys
            sys.exit(1)
        finally:
            await self._close_browser()

    async def _extract_market_data_from_html(self, page_html: str, market_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract market data from page HTML using LLM.
        
        Args:
            page_html: The HTML content of the page
            market_name: The name of the market that was clicked
            
        Returns:
            Dictionary containing the extracted market data or None if no data found
        """
        try:
            if not page_html:
                return None
            
            # Clean HTML before sending to LLM
            cleaned_html = self._clean_html_for_llm(page_html)
            
            # Use the dedicated market data extraction agent
            if self.market_data_extraction_agent is None:
                logger.error("Market data extraction agent not initialized")
                return None
            
            # Build comprehensive summary of previously extracted data
            previously_extracted_summary = self._build_previously_extracted_odds_summary()
            
            # Compose the user message (chat prompt)
            chat_prompt = f"""I have clicked on the market button \"{market_name}\" and now need to extract all the betting data from the page.\n\nHTML Content:\n{cleaned_html}\n\nTASK: Extract all betting market data for \"{market_name}\" from this HTML content.{previously_extracted_summary}\n\nExtract the market data now:"""
            
            human_message = BaseMessage.make_user_message(
                role_name="MarketDataExtractor",
                content=chat_prompt
            )
            
            response = self.market_data_extraction_agent.step(human_message)
            result_text = response.msgs[0].content.strip()
            
            # Try to extract JSON from the response
            import json
            try:
                # Look for JSON in the response
                if '{' in result_text and '}' in result_text:
                    # Find the JSON part
                    start = result_text.find('{')
                    end = result_text.rfind('}') + 1
                    json_str = result_text[start:end]
                    
                    market_data = json.loads(json_str)
                    
                    # Validate the structure
                    if isinstance(market_data, dict) and market_data.get('market_name') == market_name:
                        logger.info(f"Successfully extracted market data for: {market_name}")
                        return market_data
                    else:
                        logger.warning(f"Invalid market data structure for: {market_name}")
                        return None
                else:
                    logger.warning(f"No JSON found in LLM response for market: {market_name}")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM response for market {market_name}: {e}")
                logger.debug(f"Raw LLM response: {result_text}")
                return None
            
        except Exception as e:
            logger.error(f"Error extracting market data from HTML for market {market_name}: {e}")
            return None

    def _build_previously_extracted_odds_summary(self) -> str:
        """Build a comprehensive summary of previously extracted odds data to prevent duplication."""
        if not self.previously_extracted_markets:
            return ""
        
        summary_parts = []
        summary_parts.append("\n\nPREVIOUSLY EXTRACTED ODDS DATA (DO NOT DUPLICATE):")
        
        for market_name, odds_data in self.previously_extracted_markets.items():
            if not odds_data:
                continue
                
            summary_parts.append(f"\n--- {market_name} ---")
            
            # Handle different data structures
            if isinstance(odds_data, dict):
                # Handle structured odds data
                if 'structure' in odds_data:
                    # This is structured data from market extraction
                    structure = odds_data.get('structure', 'unknown')
                    if structure == 'table':
                        headers = odds_data.get('headers', [])
                        rows = odds_data.get('rows', [])
                        summary_parts.append(f"Structure: Table with headers {headers}")
                        for i, row in enumerate(rows[:3]):  # Show first 3 rows
                            summary_parts.append(f"  Row {i+1}: {row}")
                        if len(rows) > 3:
                            summary_parts.append(f"  ... and {len(rows) - 3} more rows")
                    elif structure == 'odds_list':
                        odds_list = odds_data.get('odds', [])
                        summary_parts.append(f"Structure: Odds list with {len(odds_list)} items")
                        for i, odds_item in enumerate(odds_list[:5]):  # Show first 5 odds
                            summary_parts.append(f"  {i+1}. {odds_item}")
                        if len(odds_list) > 5:
                            summary_parts.append(f"  ... and {len(odds_list) - 5} more odds")
                    elif structure == 'text':
                        content = odds_data.get('content', '')
                        summary_parts.append(f"Structure: Text content: {content[:200]}{'...' if len(content) > 200 else ''}")
                else:
                    # This is simple odds data
                    summary_parts.append(f"Odds values: {odds_data}")
            elif isinstance(odds_data, list):
                # Handle list of odds data
                summary_parts.append(f"Odds array with {len(odds_data)} items:")
                for i, item in enumerate(odds_data[:3]):  # Show first 3 items
                    summary_parts.append(f"  {i+1}. {item}")
                if len(odds_data) > 3:
                    summary_parts.append(f"  ... and {len(odds_data) - 3} more items")
        
        summary_parts.append("\nCRITICAL INSTRUCTIONS:")
        summary_parts.append("- DO NOT extract any odds values that match the above data")
        summary_parts.append("- DO NOT return the same numerical values for different markets")
        summary_parts.append("- Each market should have unique odds data")
        summary_parts.append("- If you see similar odds values, they belong to a different market")
        summary_parts.append("- Focus only on NEW odds data that hasn't been extracted before")
        
        return "\n".join(summary_parts)

    def _clean_html_for_llm(self, html_content: str) -> str:
        """Clean HTML content by removing anything not useful for LLM selector finding odds market buttons. Only keep main content and clickable elements relevant to betting markets."""
        try:
            from bs4 import BeautifulSoup, Tag
            from bs4.element import NavigableString
            import re

            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script, style, link, meta, video, audio, iframe, embed, object, canvas, svg
            for tag in soup(['script', 'style', 'link', 'meta', 'video', 'audio', 'iframe', 'embed', 'object', 'canvas', 'svg']):
                tag.decompose()

            # Remove images
            for img in soup.find_all('img'):
                img.decompose()

            # Remove navigation bars, footers, headers, ads, popups, cookie banners, etc.
            nav_like = ['nav', 'footer', 'header', 'aside', 'form']
            for tag in nav_like:
                for el in soup.find_all(tag):
                    el.decompose()

            # Remove elements with inline style display:none or visibility:hidden
            for el in soup.find_all(style=True):
                if not isinstance(el, Tag):
                    continue
                style = el.get('style', '')
                if isinstance(style, str):
                    style = style.lower()
                    if 'display:none' in style or 'visibility:hidden' in style:
                        el.decompose()

            # Remove empty elements
            for el in soup.find_all():
                if isinstance(el, Tag) and not el.get_text(strip=True) and not el.find_all():
                    el.decompose()

            # Return the cleaned HTML (not just text)
            return str(soup)
        except Exception as e:
            import traceback
            logger.error(f"Error cleaning HTML: {e}\n{traceback.format_exc()}")
            return html_content

    def _llm_find_selector(self, html_content: str, user_goal: str, previous_actions: str) -> list[str]:
        """Use LLM to find the best CSS selectors for a given goal. Always returns a list of valid selectors (possibly empty)."""
        import time
        max_retries = 3
        retry_count = 0
        selector_agent = self._create_selector_agent()
        if selector_agent is None:
            return []
        html_length = len(html_content)
        logger.info(f"Finding selector for goal: {user_goal} (HTML length: {html_length} chars)")
        cleaned_html = self._clean_html_for_llm(html_content)
        previous_failed_selectors = []
        while retry_count < max_retries:
            try:
                if retry_count == 0:
                    prompt = f"""HTML Content:
{cleaned_html}

You are a web scraping expert. Find the BEST CSS selector for market buttons on this betting page.

User Goal: {user_goal}
Previous Actions: {previous_actions}

STRICT REQUIREMENTS:
- The selector MUST match only actual button elements (e.g., <button>), or elements with role=\"button\" or an onclick handler. Do NOT return selectors that are too generic.
- The selector MUST match only buttons that expand or reveal betting markets about odds (not competitions, not teams, not general navigation, not other categories).
- DO NOT return selectors for non-clickable containers or divs with text only.
- DO NOT match buttons for single bets, odds, or outcomes (like '1', 'X', '2', 'Over 2.5', etc).
- DO NOT match navigation, promo, or utility buttons.
- Prefer :has-text selectors ONLY if they are very strict (e.g., match a unique, clearly labeled clickable button element). Avoid generic :has-text selectors that could match multiple or non-clickable elements.
- If you cannot find a valid selector, return the string NONE.
- DO NOT return selectors that match more than 1 element on the page.
- Return ONLY the selector or NONE, no explanations, markdown, or formatting.

TASK: Find clickable elements (buttons, links, divs) that are market category buttons. These buttons when clicked will load or expand betting markets about odds (not competitions, not teams, not navigation, not other categories).

WHAT TO LOOK FOR:
- Market category buttons like \"Vincente\", \"Totale gol\", \"Handicap Asiatico\", \"Risultato Esatto\"
- Buttons that expand or load betting markets about odds
- Elements that contain betting market categories
- Clickable elements that show betting options

AVOID:
- Individual betting options (like \"1\", \"X\", \"2\", \"Over 2.5\")
- Promotional elements
- Navigation buttons
- Already expanded content
- Buttons for competitions, teams, or other non-odds categories

SELECTOR REQUIREMENTS:
1. Must be valid CSS/Playwright syntax
2. Should be specific enough to target only market buttons
3. Should work with querySelectorAll()
4. Should be robust against minor HTML changes

EXAMPLES OF GOOD SELECTORS:
- `button[data-market]` - buttons with data-market attribute
- `.market-button` - elements with market-button class
- `[role=\"button\"]:has-text(\"Market\")` - buttons containing \"Market\" text
- `.betting-markets button` - buttons inside betting-markets container

Return ONLY the CSS selector or NONE, no explanations or markdown formatting.

CSS Selector:"""
                elif retry_count == 1:
                    prompt = f"""HTML Content:
{cleaned_html}

You are a web scraping expert. Find a CSS selector for clickable elements on this betting page.

User Goal: {user_goal}
Previous Actions: {previous_actions}

TASK: Find any clickable elements that might be related to betting markets or odds.

WHAT TO LOOK FOR:
- Any clickable buttons, links, or divs
- Elements with onclick handlers
- Elements with role=\"button\"
- Elements that might expand or show betting data

SELECTOR REQUIREMENTS:
1. Must be valid CSS/Playwright syntax
2. Should target clickable elements
3. Should work with querySelectorAll()

Return ONLY the CSS selector, no explanations or markdown formatting.

CSS Selector:"""
                else:
                    prompt = f"""HTML Content:
{cleaned_html}

You are a web scraping expert. Find any CSS selector for clickable elements.

User Goal: {user_goal}
Previous Actions: {previous_actions}

TASK: Find any clickable elements on the page.

Return ONLY a valid CSS selector that will find clickable elements, no explanations.

CSS Selector:"""
                if previous_failed_selectors:
                    prompt += f"\n\nPREVIOUS INVALID SELECTORS (do NOT repeat): {previous_failed_selectors}"
                human_message = BaseMessage.make_user_message(
                    role_name="SelectorFinder",
                    content=prompt
                )
                try:
                    response = selector_agent.step(human_message)
                except Exception as e:
                    error_str = str(e).lower()
                    if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str:
                        logger.warning(f"Rate limit ({retry_count + 1}/{max_retries}): {e}")
                        api_key_manager.record_rate_limit_error(Models.flash_lite)
                        retry_count += 1
                        logger.info(f"Retrying with new API key after rate limit...")
                        time.sleep(2)
                        continue
                    else:
                        logger.error(f"Error in selector agent: {e}")
                        break
                result_text = response.msgs[0].content.strip()
                selector = result_text.strip()
                if selector.startswith('`') and selector.endswith('`'):
                    selector = selector[1:-1]
                selectors = [s.strip() for s in selector.split(',')] if ',' in selector else [selector]
                valid_selectors = [sel for sel in selectors if self._validate_selector(sel)]
                if valid_selectors:
                    logger.info(f"Successfully found selectors: {valid_selectors}")
                    return valid_selectors
                logger.warning(f"No valid selector found in: {selector}")
                previous_failed_selectors.extend([sel for sel in selectors if sel not in valid_selectors])
                retry_count += 1
                continue
            except Exception as e:
                logger.error(f"Error finding selector (attempt {retry_count + 1}): {e}")
                retry_count += 1
                continue
        logger.error(f"Failed to find valid selector after {max_retries} attempts")
        return []

    def _create_selector_agent(self) -> Optional[ChatAgent]:
        """Create a dedicated agent for selector finding."""
        model_type = Models.flash_lite
        try:
            # Set the correct API key for the platform before creating the model
            platform = api_key_manager._get_platform_from_model(model_type)
            api_keys = api_key_manager._get_api_keys_for_platform(platform)
            if api_keys:
                idx = api_key_manager.current_key_indices.get(platform, 0)
                os.environ[f"{platform.upper()}_API_KEY"] = api_keys[idx]

            # Create a model for the agent
            model = ModelFactory.create(
                model_platform=platform,
                model_type=model_type,
                api_key=os.environ.get(f"{platform.upper()}_API_KEY")
            )

            agent = ChatAgent(
                model=model,
                system_message=SELECTOR_AGENT_SYSTEM_PROMPT
            )

            logger.info("Selector agent created successfully")
            return agent

        except Exception as e:
            logger.error(f"Failed to create selector agent: {str(e)}")
            return None

    def _validate_selector(self, selector: str) -> bool:
        """Validate if a CSS selector is syntactically correct."""
        try:
            if not selector or not selector.strip():
                return False
            
            # Basic validation - check for common CSS selector patterns
            import re
            
            # Remove any markdown formatting
            clean_selector = selector.strip()
            if clean_selector.startswith('`') and clean_selector.endswith('`'):
                clean_selector = clean_selector[1:-1]
            
            # Check for basic CSS selector syntax
            # This is a validation - in practice, you might want more comprehensive checks
            if re.match(r'^[.#]?[a-zA-Z0-9_-]+(\[[^\]]*\])*(\.[a-zA-Z0-9_-]+)*(\:[a-zA-Z0-9_-]+(\([^)]*\))?)*$', clean_selector):
                return True
            
            # Check for attribute selectors
            if re.match(r'^\[[^\]]+\]$', clean_selector):
                return True
            
            # Check for complex selectors with spaces
            if ' ' in clean_selector:
                parts = clean_selector.split()
                return all(self._validate_selector(part) for part in parts)
            
            # Check for pseudo-selectors
            if ':' in clean_selector and not clean_selector.startswith(':'):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating selector: {e}")
            return False

    async def _find_element_with_selector(self, selector: str):
        """Find an element using a CSS selector."""
        try:
            if not self.page:
                logger.error("Page is not initialized")
                return None
            
            # Try to find the element
            element = await self.page.query_selector(selector)
            
            if element:
                logger.info(f"Found element with selector: {selector}")
                return element
            else:
                logger.warning(f"No element found with selector: {selector}")
                return None
                
        except Exception as e:
            logger.error(f"Error finding element with selector {selector}: {e}")
            return None


def create_scraping_agent(headless: bool = True, model_type: str = Models.flash_lite) -> 'WebScrapingAgent':
    """Create a WebScrapingAgent instance with the specified configuration."""
    return WebScrapingAgent(headless=headless, model_type=model_type)

async def fetch_url_html(url: str) -> str:
    """Fetch the HTML content of the given URL using Playwright (headless Chromium)."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        html = await page.content()
        await browser.close()
        return html