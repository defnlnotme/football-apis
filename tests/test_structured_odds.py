#!/usr/bin/env python3
"""
Test script for the new structured odds extraction functionality.
"""

import asyncio
import logging
import json
from agent import create_scraping_agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_structured_odds_extraction():
    """Test the new structured odds extraction functionality."""
    
    # Create agent
    agent = create_scraping_agent(headless=False)  # Set to False for debugging
    
    try:
        # Test URL (you can replace with an actual betting site URL)
        test_url = "https://www.oddschecker.com/football/italy/serie-a/inter-milan-v-juventus/winner"
        
        logger.info("Testing new structured odds extraction...")
        logger.info(f"URL: {test_url}")
        
        # Test the scraping
        markets = await agent.scrape_odds_markets(
            url=test_url,
            team="Inter Milan",
            vs_team="Juventus", 
            competition="Serie A"
        )
        
        logger.info(f"Test completed. Collected {len(markets)} markets.")
        
        # Print structured data details
        for i, market in enumerate(markets):
            logger.info(f"\nMarket {i+1}:")
            logger.info(f"  Market Type: {market.get('market_type', 'unknown')}")
            logger.info(f"  Market Name: {market.get('market_name', 'unknown')}")
            logger.info(f"  Structure: {market.get('structure', 'unknown')}")
            
            if market.get('structure') == 'table':
                logger.info(f"  Headers: {market.get('headers', [])}")
                logger.info(f"  Rows: {len(market.get('rows', []))}")
                for j, row in enumerate(market.get('rows', [])[:3]):  # Show first 3 rows
                    logger.info(f"    Row {j+1}: {row}")
            
            elif market.get('structure') == 'list':
                odds_list = market.get('odds', [])
                logger.info(f"  Odds Count: {len(odds_list)}")
                for j, odds_item in enumerate(odds_list[:3]):  # Show first 3 odds
                    logger.info(f"    Odds {j+1}: {odds_item.get('condition', 'unknown')} - {odds_item.get('odds', 'unknown')} ({odds_item.get('bookmaker', 'unknown')})")
            
            elif market.get('structure') == 'text':
                content = market.get('content', '')
                logger.info(f"  Content Preview: {content[:100]}...")
            
            logger.info("---")
        
        # Save the structured data to a JSON file for inspection
        output_file = "structured_odds_test_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(markets, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Structured odds data saved to: {output_file}")
        
        return markets
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return []
    finally:
        await agent._close_browser()

def analyze_structured_data(markets):
    """Analyze the structured data to verify it's in the correct format."""
    
    logger.info("\n=== STRUCTURED DATA ANALYSIS ===")
    
    table_markets = [m for m in markets if m.get('structure') == 'table']
    list_markets = [m for m in markets if m.get('structure') == 'list']
    text_markets = [m for m in markets if m.get('structure') == 'text']
    
    logger.info(f"Table structure markets: {len(table_markets)}")
    logger.info(f"List structure markets: {len(list_markets)}")
    logger.info(f"Text structure markets: {len(text_markets)}")
    
    # Analyze table markets
    for i, market in enumerate(table_markets):
        logger.info(f"\nTable Market {i+1}: {market.get('market_type', 'unknown')}")
        headers = market.get('headers', [])
        rows = market.get('rows', [])
        
        if headers:
            logger.info(f"  Headers: {headers}")
            logger.info(f"  Columns: {len(headers)}")
        
        if rows:
            logger.info(f"  Rows: {len(rows)}")
            logger.info(f"  Sample row: {rows[0] if rows else 'No rows'}")
    
    # Analyze list markets
    for i, market in enumerate(list_markets):
        logger.info(f"\nList Market {i+1}: {market.get('market_type', 'unknown')}")
        odds_list = market.get('odds', [])
        logger.info(f"  Odds entries: {len(odds_list)}")
        
        if odds_list:
            sample_odds = odds_list[0]
            logger.info(f"  Sample odds: {sample_odds}")
    
    logger.info("\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    # Run the test
    markets = asyncio.run(test_structured_odds_extraction())
    
    # Analyze the results
    if markets:
        analyze_structured_data(markets)
    else:
        logger.warning("No markets collected to analyze") 