#!/usr/bin/env python3

from agent import WebScrapingAgent

def test_selector_validation():
    print("Testing Selector Validation...")
    
    # Create agent instance (without browser)
    agent = WebScrapingAgent(headless=True)
    
    # Test valid selectors
    valid_selectors = [
        'button[data-testid="all-markets"]',
        '#markets-popup button',
        'button[aria-label="Winner market"]',
        'div[class*="market-categories"] button',
        'button:has-text("Vincente")',
        '[data-market-type="winner"]',
        'button.market-button.primary',
        'div:nth-child(2) button'
    ]
    
    print("\nTesting VALID selectors:")
    for selector in valid_selectors:
        is_valid = agent._validate_selector(selector)
        print(f"  {selector}: {'✓' if is_valid else '✗'}")
    
    # Test invalid selectors
    invalid_selectors = [
        'button',  # too generic
        'div',     # too generic
        'span',    # too generic
        'button:contains("text")',  # invalid syntax
        'CLICK_BUTTON:text',        # action command
        '.class',                   # too generic
        'button[class*="btn"]',     # too generic pattern
        '',                         # empty
        '   ',                      # whitespace only
    ]
    
    print("\nTesting INVALID selectors:")
    for selector in invalid_selectors:
        is_valid = agent._validate_selector(selector)
        print(f"  {selector}: {'✗' if not is_valid else '✓'}")
    
    print("\nSelector validation test completed!")

if __name__ == "__main__":
    test_selector_validation() 