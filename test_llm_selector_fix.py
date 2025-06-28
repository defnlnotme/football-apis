#!/usr/bin/env python3
"""
Test script to verify that the LLM no longer returns example selectors from the prompt.
"""

import asyncio
import logging
from agent import create_scraping_agent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def test_llm_selector_fix():
    """Test that the LLM doesn't return example selectors."""
    try:
        print("Creating scraping agent...")
        agent = create_scraping_agent(headless=True)
        
        print("Testing LLM selector finding with sample HTML...")
        
        # Sample HTML that doesn't contain the example selectors
        sample_html = """
        <html>
        <body>
            <div class="betting-container">
                <button class="expand-markets-btn">Show All Markets</button>
                <div class="markets-section">
                    <button class="market-btn">Winner</button>
                    <button class="market-btn">Total Goals</button>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Test the LLM selector finding
        selector = agent._llm_find_selector(
            sample_html,
            "find the 'Show All Markets' button",
            "navigated to page"
        )
        
        print(f"LLM returned selector: {selector}")
        
        # Check if it's one of the example selectors
        example_selectors = [
            'button[data-testid="all-markets"]',
            '#markets-popup button',
            'button[aria-label="Winner market"]',
            'div[class*="market-categories"] button',
            'button:has-text("Vincente")',
            '[data-market-type="winner"]'
        ]
        
        if selector in example_selectors:
            print("❌ FAILED: LLM returned example selector from prompt")
            return False
        elif selector is None:
            print("⚠️  LLM returned None (which is acceptable)")
            return True
        else:
            print("✅ SUCCESS: LLM returned a real selector from the HTML")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing LLM selector fix...")
    success = asyncio.run(test_llm_selector_fix())
    if success:
        print("✅ Test passed!")
    else:
        print("❌ Test failed!") 