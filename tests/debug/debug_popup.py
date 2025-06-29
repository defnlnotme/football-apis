#!/usr/bin/env python3
"""
Debug script to understand why the popup isn't appearing after clicking the "all markets" button.
"""

import asyncio
import time
from playwright.async_api import async_playwright
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_popup_issue():
    """Debug the popup issue by manually testing the button click and popup detection."""
    
    url = "https://www.oddschecker.com/it/calcio/italia/serie-a/ac-milan-cremonese"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Set to False to see what's happening
        page = await browser.new_page()
        
        try:
            # Navigate to the page
            logger.info(f"Navigating to: {url}")
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=60000)  # Use domcontentloaded instead of networkidle
                logger.info("Successfully navigated to the page")
            except Exception as e:
                logger.warning(f"Navigation timeout, but continuing: {e}")
                # Try to continue anyway
            
            # Wait a bit for the page to fully load
            await asyncio.sleep(3)
            
            # Take a screenshot before clicking
            await page.screenshot(path="before_click.png")
            logger.info("Screenshot saved: before_click.png")
            
            # Get initial HTML
            initial_html = await page.content()
            logger.info(f"Initial HTML length: {len(initial_html)} characters")
            
            # Look for the "all markets" button
            logger.info("Looking for 'all markets' button...")
            
            # Try different selectors that might work
            button_selectors = [
                'button[data-testid="all-markets-button"]',
                'button:has-text("Tutti i Mercati")',
                'button:has-text("All Markets")',
                'button:has-text("Show More")',
                'button:has-text("More Markets")',
                '[data-testid*="markets"]',
                '[data-testid*="expand"]',
                'button[aria-label*="markets"]',
                'button[aria-label*="expand"]'
            ]
            
            button_found = None
            for selector in button_selectors:
                try:
                    button = await page.query_selector(selector)
                    if button and await button.is_visible():
                        logger.info(f"Found button with selector: {selector}")
                        button_found = button
                        break
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
            
            if not button_found:
                logger.error("Could not find 'all markets' button with any selector")
                
                # Let's look at all buttons on the page
                all_buttons = await page.query_selector_all('button')
                logger.info(f"Found {len(all_buttons)} buttons on the page")
                
                for i, button in enumerate(all_buttons[:10]):  # Check first 10 buttons
                    try:
                        text = await button.text_content()
                        is_visible = await button.is_visible()
                        logger.info(f"Button {i}: text='{text}', visible={is_visible}")
                    except Exception as e:
                        logger.debug(f"Error getting button {i} info: {e}")
                
                return
            
            # Get button text and attributes for debugging
            if button_found:
                button_text = await button_found.text_content()
                button_attributes = await button_found.evaluate('el => Object.fromEntries(Object.entries(el.attributes).map(([k,v]) => [k, v.value]))')
                logger.info(f"Button text: '{button_text}'")
                logger.info(f"Button attributes: {button_attributes}")
                
                # Click the button
                logger.info("Clicking the button...")
                await button_found.click()
            else:
                logger.error("No button found to click")
                return
            
            # Wait and monitor for changes
            logger.info("Monitoring for popup appearance...")
            start_time = time.time()
            max_wait = 15
            
            while time.time() - start_time < max_wait:
                try:
                    # Get current HTML
                    current_html = await page.content()
                    
                    # Check for significant changes
                    if len(current_html) > len(initial_html) + 100:
                        logger.info(f"HTML size increased: {len(initial_html)} -> {len(current_html)} characters")
                        
                        # Look for popup-like elements
                        popup_selectors = [
                            '[role="dialog"]',
                            '[class*="popup"]',
                            '[class*="modal"]',
                            '[class*="overlay"]',
                            '[class*="drawer"]',
                            '[id*="popup"]',
                            '[id*="modal"]',
                            '[data-testid*="popup"]',
                            '[data-testid*="modal"]'
                        ]
                        
                        for popup_selector in popup_selectors:
                            try:
                                popup_elements = await page.query_selector_all(popup_selector)
                                if popup_elements:
                                    logger.info(f"Found {len(popup_elements)} elements with selector: {popup_selector}")
                                    for i, popup in enumerate(popup_elements):
                                        try:
                                            is_visible = await popup.is_visible()
                                            text = await popup.text_content()
                                            logger.info(f"Popup {i}: visible={is_visible}, text='{text[:100]}...'")
                                        except Exception as e:
                                            logger.debug(f"Error getting popup {i} info: {e}")
                            except Exception as e:
                                logger.debug(f"Error with popup selector {popup_selector}: {e}")
                        
                        # Take a screenshot after changes
                        await page.screenshot(path="after_click.png")
                        logger.info("Screenshot saved: after_click.png")
                        
                        # Save the HTML for analysis
                        with open("after_click.html", "w", encoding="utf-8") as f:
                            f.write(current_html)
                        logger.info("HTML saved: after_click.html")
                        
                        break
                    
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error during monitoring: {e}")
                    await asyncio.sleep(0.5)
            
            # Final check - look for any market-related content
            logger.info("Final check for market content...")
            final_html = await page.content()
            
            # Look for market keywords in the HTML
            market_keywords = ['vincente', 'pareggio', 'sconfitta', 'totale', 'handicap', 'over', 'under', 'mercati', 'markets']
            for keyword in market_keywords:
                if keyword in final_html.lower():
                    logger.info(f"Found market keyword '{keyword}' in HTML")
            
            # Look for odds patterns (numbers with decimals)
            import re
            odds_pattern = re.compile(r'\b\d+\.\d+\b')
            odds_matches = odds_pattern.findall(final_html)
            if odds_matches:
                logger.info(f"Found {len(odds_matches)} odds values in HTML")
                logger.info(f"Sample odds: {odds_matches[:10]}")
            
            logger.info("Debug session completed")
            
        except Exception as e:
            logger.error(f"Error during debug session: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(debug_popup_issue()) 