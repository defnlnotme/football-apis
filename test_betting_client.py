import os
import sys
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from football_apis.clients.betting_odds import BettingOddsClient

def main():
    # Initialize the client with the API key from apikeys.txt
    api_key = "36d28d6506d794715a812401e41f7e6b"
    logger.info(f"Initializing BettingOddsClient with API key: {api_key[:5]}...{api_key[-3:]}")
    
    try:
        client = BettingOddsClient(api_key=api_key, cache_enabled=False)
        logger.info("Client initialized successfully")
        
        logger.info("Testing connection to The Odds API...")
        if client.test_connection():
            logger.info("✅ Successfully connected to The Odds API")
        
            # Test getting available sports
            logger.info("Fetching available sports...")
            sports = client.get_sports()
            logger.info(f"Found {len(sports)} sports")
            if sports:
                logger.info("First 5 sports:")
                for sport in sports[:5]:
                    logger.info(f"- {sport.get('title', 'N/A')} (key: {sport.get('key', 'N/A')})")
            else:
                logger.warning("No sports data returned")
            
            # Test getting odds for soccer EPL
            logger.info("Fetching odds for English Premier League...")
            try:
                odds = client.get_odds(
                    sport_key="soccer_epl",
                    regions="eu",
                    markets="h2h",
                    odds_format="decimal"
                )
                
                if odds and isinstance(odds, list):
                    logger.info(f"Found {len(odds)} upcoming matches")
                    for i, match in enumerate(odds[:3], 1):  # Show first 3 matches
                        home_team = match.get('home_team', 'Unknown')
                        away_team = match.get('away_team', 'Unknown')
                        commence_time = match.get('commence_time', 'Unknown')
                        logger.info(f"\nMatch {i}: {home_team} vs {away_team}")
                        logger.info(f"Time: {commence_time}")
                        
                        # Show bookmaker odds
                        bookmakers = match.get('bookmakers', [])[:2]  # First 2 bookmakers
                        if not bookmakers:
                            logger.info("  No bookmaker data available")
                            
                        for bookmaker in bookmakers:
                            logger.info(f"\n  {bookmaker.get('title', 'Unknown Bookmaker')}:")
                            for market in bookmaker.get('markets', []):
                                if market.get('key') == 'h2h':
                                    for outcome in market.get('outcomes', []):
                                        logger.info(f"    {outcome.get('name', 'N/A')}: {outcome.get('price', 'N/A')}")
                else:
                    logger.warning("No odds data available for EPL at the moment or invalid response format")
                    if isinstance(odds, dict):
                        logger.warning(f"Response content: {odds}")
                    
            except Exception as e:
                logger.error(f"❌ Error fetching odds: {str(e)}", exc_info=True)
        else:
            logger.error("❌ Failed to connect to The Odds API")
    except Exception as e:
        logger.error(f"❌ Error initializing client: {str(e)}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        raise
