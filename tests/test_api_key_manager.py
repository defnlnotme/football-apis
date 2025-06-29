#!/usr/bin/env python3
"""
Test script for the API key manager functionality.
"""

import asyncio
import logging
from agent import api_key_manager, create_scraping_agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_key_manager():
    """Test the API key manager functionality."""
    
    logger.info("Testing API key manager...")
    
    # Test loading API keys
    logger.info("Testing API key loading...")
    api_keys = api_key_manager._get_api_keys_for_platform('gemini')
    logger.info(f"Found {len(api_keys)} Gemini API keys")
    
    # Test current key retrieval
    current_key = api_key_manager.get_current_api_key('gemini')
    logger.info(f"Current Gemini API key: {current_key[:10]}..." if current_key else "No current key")
    
    # Test rate limit error recording
    logger.info("Testing rate limit error recording...")
    api_key_manager.record_rate_limit_error('gemini-2.5-flash-lite-preview-06-17')
    logger.info("Rate limit error recorded")
    
    # Test key rotation
    logger.info("Testing API key rotation...")
    api_key_manager._rotate_api_key('gemini')
    new_key = api_key_manager.get_current_api_key('gemini')
    logger.info(f"New Gemini API key: {new_key[:10]}..." if new_key else "No new key")
    
    logger.info("API key manager test completed.")

async def test_agent_with_rate_limit_handling():
    """Test the agent with rate limit handling."""
    
    logger.info("Testing agent with rate limit handling...")
    
    # Create agent
    agent = create_scraping_agent(headless=True)
    
    try:
        # Test URL (this should trigger rate limit handling if needed)
        test_url = "https://www.oddschecker.com/football/italy/serie-a/inter-milan-v-juventus/winner"
        
        logger.info(f"Testing agent with URL: {test_url}")
        
        # Test the scraping (this will use the API key manager internally)
        markets = await agent.scrape_odds_markets(
            url=test_url,
            team="Inter Milan",
            vs_team="Juventus", 
            competition="Serie A"
        )
        
        logger.info(f"Agent test completed. Collected {len(markets)} markets.")
        return markets
        
    except Exception as e:
        logger.error(f"Agent test failed: {e}")
        return []
    finally:
        await agent._close_browser()

if __name__ == "__main__":
    # Test API key manager
    test_api_key_manager()
    
    # Test agent with rate limit handling
    asyncio.run(test_agent_with_rate_limit_handling()) 