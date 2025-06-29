#!/usr/bin/env python3
"""
Detailed debug script to investigate the modal element that's found but not visible.
"""

import asyncio
import time
from playwright.async_api import async_playwright
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_modal_issue():
    """Debug the modal visibility issue."""
    
    url = "https://www.oddschecker.com/it/calcio/italia/serie-a/ac-milan-cremonese"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        try:
            # Navigate to the page
            logger.info(f"Navigating to: {url}")
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=60000)
                logger.info("Successfully navigated to the page")
            except Exception as e:
                logger.warning(f"Navigation timeout, but continuing: {e}")
            
            await asyncio.sleep(3)
            
            # Find and click the button
            button = await page.query_selector('button[data-testid="all-markets-button"]')
            if not button:
                logger.error("Button not found")
                return
            
            logger.info("Clicking the button...")
            await button.click()
            
            # Wait a bit for the modal to appear
            await asyncio.sleep(2)
            
            # Look for modal elements
            modal_selectors = [
                '[id*="modal"]',
                '[class*="modal"]',
                '[role="dialog"]',
                '[class*="popup"]',
                '[class*="overlay"]',
                '[class*="drawer"]'
            ]
            
            for selector in modal_selectors:
                elements = await page.query_selector_all(selector)
                if elements:
                    logger.info(f"Found {len(elements)} elements with selector: {selector}")
                    
                    for i, element in enumerate(elements):
                        try:
                            # Get basic info
                            is_visible = await element.is_visible()
                            text = await element.text_content()
                            tag_name = await element.evaluate('el => el.tagName')
                            
                            # Get computed styles
                            styles = await element.evaluate('''el => {
                                const styles = window.getComputedStyle(el);
                                return {
                                    display: styles.display,
                                    visibility: styles.visibility,
                                    opacity: styles.opacity,
                                    position: styles.position,
                                    top: styles.top,
                                    left: styles.left,
                                    zIndex: styles.zIndex,
                                    width: styles.width,
                                    height: styles.height,
                                    transform: styles.transform,
                                    pointerEvents: styles.pointerEvents
                                };
                            }''')
                            
                            # Get bounding box
                            bbox = await element.bounding_box()
                            
                            logger.info(f"Element {i}:")
                            logger.info(f"  Tag: {tag_name}")
                            logger.info(f"  Visible: {is_visible}")
                            logger.info(f"  Text length: {len(text) if text else 0}")
                            logger.info(f"  Styles: {styles}")
                            logger.info(f"  Bounding box: {bbox}")
                            
                            # Check if it's the modal we're looking for
                            if text and len(text) > 100 and any(keyword in text.lower() for keyword in ['vincente', 'pareggio', 'mercati', 'markets']):
                                logger.info(f"  *** This looks like the markets modal! ***")
                                
                                # Try to make it visible by removing problematic styles
                                try:
                                    await element.evaluate('''el => {
                                        el.style.display = 'block';
                                        el.style.visibility = 'visible';
                                        el.style.opacity = '1';
                                        el.style.pointerEvents = 'auto';
                                    }''')
                                    logger.info("  Attempted to make modal visible")
                                    
                                    # Check if it's now visible
                                    is_visible_after = await element.is_visible()
                                    logger.info(f"  Visible after style fix: {is_visible_after}")
                                    
                                except Exception as e:
                                    logger.error(f"  Error trying to make modal visible: {e}")
                            
                        except Exception as e:
                            logger.error(f"Error analyzing element {i}: {e}")
            
            # Also check for any elements with market content that might be hidden
            logger.info("Looking for hidden elements with market content...")
            
            # Use JavaScript to find all elements with market keywords
            hidden_market_elements = await page.evaluate('''() => {
                const marketKeywords = ['vincente', 'pareggio', 'mercati', 'markets', 'totale', 'handicap'];
                const elements = [];
                
                function checkElement(el) {
                    const text = el.textContent || '';
                    const hasMarketContent = marketKeywords.some(keyword => 
                        text.toLowerCase().includes(keyword)
                    );
                    
                    if (hasMarketContent && text.length > 50) {
                        const styles = window.getComputedStyle(el);
                        const isHidden = styles.display === 'none' || 
                                       styles.visibility === 'hidden' || 
                                       parseFloat(styles.opacity) === 0;
                        
                        if (isHidden) {
                            elements.push({
                                tagName: el.tagName,
                                text: text.substring(0, 200),
                                styles: {
                                    display: styles.display,
                                    visibility: styles.visibility,
                                    opacity: styles.opacity,
                                    position: styles.position,
                                    zIndex: styles.zIndex
                                }
                            });
                        }
                    }
                    
                    // Check children
                    for (const child of el.children) {
                        checkElement(child);
                    }
                }
                
                checkElement(document.body);
                return elements;
            }''')
            
            if hidden_market_elements:
                logger.info(f"Found {len(hidden_market_elements)} hidden elements with market content:")
                for i, elem in enumerate(hidden_market_elements[:5]):  # Show first 5
                    logger.info(f"  Hidden element {i}: {elem}")
            
            logger.info("Debug session completed")
            
        except Exception as e:
            logger.error(f"Error during debug session: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(debug_modal_issue()) 