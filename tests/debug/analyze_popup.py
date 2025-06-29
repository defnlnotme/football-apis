#!/usr/bin/env python3
"""
Analyze the popup issue and understand why the modal has zero height.
"""

import asyncio
import time
from playwright.async_api import async_playwright
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def analyze_popup_issue():
    """Analyze the popup issue in detail."""
    
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
            
            # Wait for the modal to appear
            await asyncio.sleep(3)
            
            # Look for the modal element that was found in the debug
            modal = await page.query_selector('[id*="modal"]')
            if modal:
                logger.info("Found modal element")
                
                # Get detailed information about the modal
                modal_info = await modal.evaluate('''el => {
                    const styles = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return {
                        tagName: el.tagName,
                        id: el.id,
                        className: el.className,
                        innerHTML: el.innerHTML.substring(0, 500),
                        styles: {
                            display: styles.display,
                            visibility: styles.visibility,
                            opacity: styles.opacity,
                            position: styles.position,
                            top: styles.top,
                            left: styles.left,
                            width: styles.width,
                            height: styles.height,
                            zIndex: styles.zIndex,
                            transform: styles.transform,
                            pointerEvents: styles.pointerEvents,
                            overflow: styles.overflow
                        },
                        rect: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height,
                            top: rect.top,
                            left: rect.left,
                            bottom: rect.bottom,
                            right: rect.right
                        }
                    };
                }''')
                
                logger.info(f"Modal info: {modal_info}")
                
                # Check if there are any child elements
                children = await modal.query_selector_all('*')
                logger.info(f"Modal has {len(children)} child elements")
                
                # Look for any elements with market content
                market_elements = await page.evaluate('''() => {
                    const marketKeywords = ['vincente', 'pareggio', 'mercati', 'markets', 'totale', 'handicap'];
                    const elements = [];
                    
                    function checkElement(el) {
                        const text = el.textContent || '';
                        const hasMarketContent = marketKeywords.some(keyword => 
                            text.toLowerCase().includes(keyword)
                        );
                        
                        if (hasMarketContent && text.length > 20) {
                            const styles = window.getComputedStyle(el);
                            elements.push({
                                tagName: el.tagName,
                                text: text.substring(0, 100),
                                styles: {
                                    display: styles.display,
                                    visibility: styles.visibility,
                                    opacity: styles.opacity,
                                    position: styles.position,
                                    width: styles.width,
                                    height: styles.height
                                }
                            });
                        }
                        
                        // Check children
                        for (const child of el.children) {
                            checkElement(child);
                        }
                    }
                    
                    checkElement(document.body);
                    return elements;
                }''')
                
                if market_elements:
                    logger.info(f"Found {len(market_elements)} elements with market content:")
                    for i, elem in enumerate(market_elements[:10]):  # Show first 10
                        logger.info(f"  Market element {i}: {elem}")
                
                # Try to wait for the modal to load content
                logger.info("Waiting for modal content to load...")
                for i in range(10):
                    await asyncio.sleep(1)
                    
                    # Check if modal height has changed
                    current_height = await modal.evaluate('el => window.getComputedStyle(el).height')
                    logger.info(f"Modal height after {i+1}s: {current_height}")
                    
                    # Check if any content has appeared
                    content = await modal.evaluate('el => el.textContent')
                    if content and len(content.strip()) > 0:
                        logger.info(f"Modal now has content: {content[:200]}...")
                        break
                
                # Try to force the modal to be visible
                logger.info("Attempting to make modal visible...")
                await modal.evaluate('''el => {
                    el.style.display = 'block';
                    el.style.visibility = 'visible';
                    el.style.opacity = '1';
                    el.style.height = 'auto';
                    el.style.minHeight = '200px';
                    el.style.pointerEvents = 'auto';
                    el.style.position = 'fixed';
                    el.style.top = '50px';
                    el.style.left = '50px';
                    el.style.zIndex = '9999';
                }''')
                
                # Wait a bit and check again
                await asyncio.sleep(2)
                
                final_info = await modal.evaluate('''el => {
                    const styles = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return {
                        textContent: el.textContent.substring(0, 500),
                        styles: {
                            display: styles.display,
                            visibility: styles.visibility,
                            opacity: styles.opacity,
                            height: styles.height,
                            width: styles.width
                        },
                        rect: {
                            width: rect.width,
                            height: rect.height
                        }
                    };
                }''')
                
                logger.info(f"Final modal info after style changes: {final_info}")
                
            else:
                logger.error("Modal element not found")
            
            logger.info("Analysis completed")
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(analyze_popup_issue()) 