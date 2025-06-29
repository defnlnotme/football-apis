#!/usr/bin/env python3
"""
Test script for the simplified market type identification approach.
"""

import asyncio
import logging
import json
from agent import create_scraping_agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_simplified_market_types():
    """Test the simplified market type identification approach."""
    
    # Create agent
    agent = create_scraping_agent(headless=False)  # Set to False for debugging
    
    try:
        # Test URL (you can replace with an actual betting site URL)
        test_url = "https://www.oddschecker.com/football/italy/serie-a/inter-milan-v-juventus/winner"
        
        logger.info("Testing simplified market type identification...")
        logger.info(f"URL: {test_url}")
        
        # Test the scraping
        markets = await agent.scrape_odds_markets(
            url=test_url,
            team="Inter Milan",
            vs_team="Juventus", 
            competition="Serie A"
        )
        
        logger.info(f"Test completed. Collected {len(markets)} markets.")
        
        # Analyze the market types found
        market_types = {}
        for market in markets:
            market_type = market.get('market_type', 'unknown')
            market_name = market.get('market_name', 'Unknown Market')
            structure = market.get('structure', 'unknown')
            
            if market_type not in market_types:
                market_types[market_type] = []
            
            market_types[market_type].append({
                'name': market_name,
                'structure': structure
            })
        
        # Print analysis
        logger.info("\n=== MARKET TYPE ANALYSIS ===")
        logger.info(f"Total unique market types found: {len(market_types)}")
        
        for market_type, markets_list in market_types.items():
            logger.info(f"\nMarket Type: {market_type}")
            logger.info(f"  Count: {len(markets_list)}")
            for market in markets_list:
                logger.info(f"  - {market['name']} ({market['structure']})")
        
        # Save the structured data to a JSON file for inspection
        output_file = "simplified_market_types_test_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'markets': markets,
                'market_types_analysis': market_types,
                'total_markets': len(markets),
                'unique_market_types': len(market_types)
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nDetailed results saved to: {output_file}")
        
        return markets, market_types
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return [], {}
    finally:
        await agent._close_browser()

def test_market_type_extraction():
    """Test the market type extraction logic with sample text."""
    
    # Create a test instance to test the methods
    agent = create_scraping_agent(headless=True)
    
    test_cases = [
        "Match Winner - Inter Milan vs Juventus",
        "Total Goals Over/Under 2.5",
        "Asian Handicap -1.5",
        "Exact Score 2-1",
        "First Half Winner",
        "Both Teams to Score",
        "Corner Kicks Over 9.5",
        "Yellow Cards Over 4.5",
        "Player to Score First",
        "Clean Sheet - Yes",
        "Some Random Market We Don't Know About",
        "Double Chance Home or Draw",
        "Half Time/Full Time Result"
    ]
    
    logger.info("\n=== MARKET TYPE EXTRACTION TEST ===")
    
    for test_text in test_cases:
        market_type = agent._identify_market_type_from_text(test_text)
        logger.info(f"Text: '{test_text}'")
        logger.info(f"  Extracted Type: {market_type}")
        logger.info("---")

if __name__ == "__main__":
    # Test the market type extraction logic first
    test_market_type_extraction()
    
    # Run the full test
    markets, market_types = asyncio.run(test_simplified_market_types())
    
    if markets:
        logger.info(f"\n✅ Successfully collected {len(markets)} markets with {len(market_types)} unique market types")
    else:
        logger.warning("❌ No markets collected") 