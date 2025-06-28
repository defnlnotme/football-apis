#!/usr/bin/env python3
"""
Test script for the new diff-based agent behavior.
"""

import asyncio
import logging
from agent import create_scraping_agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_agent_diff_behavior():
    """Test the new diff-based agent behavior."""
    
    # Create agent
    agent = create_scraping_agent(headless=False)  # Set to False for debugging
    
    try:
        # Test URL (you can replace with an actual betting site URL)
        test_url = "https://www.oddschecker.com/football/italy/serie-a/inter-milan-v-juventus/winner"
        
        logger.info("Testing new diff-based agent behavior...")
        logger.info(f"URL: {test_url}")
        
        # Test the scraping
        markets = await agent.scrape_odds_markets(
            url=test_url,
            team="Inter Milan",
            vs_team="Juventus", 
            competition="Serie A"
        )
        
        logger.info(f"Test completed. Collected {len(markets)} markets.")
        
        # Print some details about collected markets
        for i, market in enumerate(markets[:3]):  # Show first 3 markets
            logger.info(f"Market {i+1}:")
            logger.info(f"  Odds: {market.get('odds', [])}")
            logger.info(f"  Source: {market.get('source', 'unknown')}")
            logger.info(f"  Text: {market.get('text', '')[:100]}...")
            logger.info("---")
        
        return markets
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return []
    finally:
        await agent._close_browser()

if __name__ == "__main__":
    asyncio.run(test_agent_diff_behavior()) 