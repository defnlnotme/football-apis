from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import requests
from ..base_client import RateLimitedAPIClient
import logging

logger = logging.getLogger(__name__)

class TheOddsApiClient(RateLimitedAPIClient):
    """Client for accessing betting odds data."""
    
    def __init__(self, api_key: Optional[str] = None, cache_enabled: bool = True, cache_ttl: int = 3600):
        """
        Initialize the TheOddsApiClient.
        
        Args:
            api_key: API key for The Odds API (https://the-odds-api.com/)
            cache_enabled: Whether to enable response caching
            cache_ttl: Time-to-live for cache entries in seconds (default: 1 hour)
        """
        super().__init__(
            api_key=api_key,
            base_url="https://api.the-odds-api.com/v4",
            cache_enabled=cache_enabled,
            cache_ttl=cache_ttl
        )
        
        # Ensure session is properly configured
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def _add_auth(self):
        """Add authentication to the request."""
        if self.api_key:
            self.session.params = {"apiKey": self.api_key}
    
    def test_connection(self) -> bool:
        """Test the API connection."""
        try:
            response = self.get("/sports")
            return isinstance(response, list) and len(response) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_sports(self, all_available: bool = False) -> List[Dict[str, Any]]:
        """
        Get a list of available sports and their keys.
        
        Args:
            all_available: If True, return all available sports, not just active ones
            
        Returns:
            List of sports with their details
        """
        params = {}
        if all_available:
            params["all"] = "true"
            
        response = self.get("/sports", params=params)
        return response if isinstance(response, list) else []
    
    def get_odds(
        self,
        sport_key: str = "soccer_epl",
        regions: str = "eu",
        markets: str = "h2h,spreads,totals",
        date_format: str = "iso",
        odds_format: str = "decimal",
        bookmakers: Optional[str] = None,
        commence_time_from: Optional[Union[str, datetime]] = None,
        commence_time_to: Optional[Union[str, datetime]] = None,
        event_ids: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get odds for upcoming events.
        
        Args:
            sport_key: The sport key (default: soccer_epl)
            regions: Regions to include (comma-separated, e.g., "eu,us,uk")
            markets: Markets to include (comma-separated, e.g., "h2h,spreads,totals")
            date_format: Format for dates (iso or unix)
            odds_format: Format for odds (decimal, american, hongkong, malay, indonesian)
            bookmakers: Comma-separated list of bookmakers to include
            commence_time_from: Filter events starting after this time
            commence_time_to: Filter events starting before this time
            event_ids: Comma-separated game ids to filter the response
            
        Returns:
            List of events with their odds
        """
        params = {
            "regions": regions,
            "markets": markets,
            "dateFormat": date_format,
            "oddsFormat": odds_format,
        }
        
        if bookmakers:
            params["bookmakers"] = bookmakers
            
        if commence_time_from:
            if isinstance(commence_time_from, datetime):
                commence_time_from = commence_time_from.isoformat()
            params["commenceTimeFrom"] = commence_time_from
            
        if commence_time_to:
            if isinstance(commence_time_to, datetime):
                commence_time_to = commence_time_to.isoformat()
            params["commenceTimeTo"] = commence_time_to
            
        if event_ids:
            params["eventIds"] = event_ids
        
        endpoint = f"/sports/{sport_key}/odds"
        response = self.get(endpoint, params=params)
        
        if not isinstance(response, list):
            logger.warning(f"Unexpected response format: {response}")
            return []
            
        return response
    
    def get_scores(
        self,
        sport_key: str = "soccer_epl",
        days_from: int = 3,
        date_format: str = "iso"
    ) -> List[Dict[str, Any]]:
        """
        Get scores for recently completed events.
        
        Args:
            sport_key: The sport key (default: soccer_epl)
            days_from: Number of days to look back for completed events (max 3 for free tier)
            date_format: Format for dates (iso or unix)
            
        Returns:
            List of events with their scores
        """
        # Limit days_from to 3 for free tier
        days_from = min(max(days_from, 1), 3)
        
        params = {
            "daysFrom": days_from,
            "dateFormat": date_format
        }
        
        endpoint = f"/sports/{sport_key}/scores"
        response = self.get(endpoint, params=params)
        
        if not isinstance(response, list):
            logger.warning(f"Unexpected response format: {response}")
            return []
            
        return response
    
    def get_historical_odds(
        self,
        sport_key: str,
        date: Union[str, datetime],
        regions: str = "eu",
        markets: str = "h2h,spreads,totals",
        date_format: str = "iso",
        odds_format: str = "decimal"
    ) -> Dict[str, Any]:
        """
        Get historical odds for events that started at a specific time.
        
        Note: This endpoint might not be available in the free tier.
        
        Args:
            sport_key: The sport key
            date: The timestamp of the data snapshot to be returned (ISO8601 format)
            regions: Regions to include (comma-separated)
            markets: Markets to include (comma-separated)
            date_format: Format for dates (iso or unix)
            odds_format: Format for odds
            
        Returns:
            Historical odds data for the events
        """
        if isinstance(date, datetime):
            date = date.isoformat()
        
        params = {
            "regions": regions,
            "markets": markets,
            "dateFormat": date_format,
            "oddsFormat": odds_format,
            "date": date
        }
        
        endpoint = f"/historical/sports/{sport_key}/odds"
        response = self.get(endpoint, params=params)
        return response if isinstance(response, dict) else {}
    
    def get_events(
        self,
        sport_key: str = "soccer_epl",
        date_format: str = "iso"
    ) -> List[Dict[str, Any]]:
        """
        Get a list of upcoming events.
        
        Args:
            sport_key: The sport key (default: soccer_epl)
            date_format: Format for dates (iso or unix)
            
        Returns:
            List of upcoming events
        """
        params = {
            "dateFormat": date_format,
        }
        
        endpoint = f"/sports/{sport_key}/events"
        response = self.get(endpoint, params=params)
        
        if not isinstance(response, list):
            logger.warning(f"Unexpected response format: {response}")
            return []
            
        return response
    
    def get_historical_odds_archive(
        self,
        sport_key: str,
        date: Union[str, datetime],
        regions: str = "eu",
        markets: str = "h2h,spreads,totals",
        date_format: str = "iso",
        odds_format: str = "decimal"
    ) -> Dict[str, Any]:
        """
        Get historical odds for events that started at a specific time.
        
        Note: This endpoint might not be available in the free tier.
        
        Args:
            sport_key: The sport key
            date: The timestamp of the data snapshot to be returned (ISO8601 format)
            regions: Regions to include (comma-separated)
            markets: Markets to include (comma-separated)
            date_format: Format for dates (iso or unix)
            odds_format: Format for odds
            
        Returns:
            Historical odds data for the events
        """
        if isinstance(date, datetime):
            date = date.isoformat()
        
        params = {
            "regions": regions,
            "markets": markets,
            "dateFormat": date_format,
            "oddsFormat": odds_format,
            "date": date
        }
        
        endpoint = f"/sports/{sport_key}/odds-history"
        response = self.get(endpoint, params=params)
        return response if isinstance(response, dict) else {}
