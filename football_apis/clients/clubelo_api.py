from typing import Dict, List, Optional, Union, Any, cast
from datetime import datetime, timedelta
import requests
import csv
import io
from ..base_client import RateLimitedAPIClient
import logging

logger = logging.getLogger(__name__)

class ClubEloClient(RateLimitedAPIClient):
    """Client for accessing team strength ratings (Elo, etc.)."""
    
    def __init__(self, cache_enabled: bool = True, cache_ttl: int = 86400):
        """
        Initialize the ClubEloClient.
        
        Args:
            cache_enabled: Whether to enable response caching
            cache_ttl: Time-to-live for cache entries in seconds (default: 24 hours)
        """
        super().__init__(
            base_url="http://api.clubelo.com",
            cache_enabled=cache_enabled,
            cache_ttl=cache_ttl
        )
        
        # Ensure session is properly configured
        self.session.headers.update({
            'Content-Type': 'text/csv',
            'Accept': 'text/csv'
        })
    
    def _parse_elo_rating(self, rating: Dict[str, Any], is_current: bool = False) -> Dict[str, Any]:
        """
        Parse a single Elo rating entry into a standardized format.
        
        Args:
            rating: Dictionary containing raw Elo rating data
            is_current: Whether this is the current/latest rating
            
        Returns:
            Dictionary with standardized Elo rating data
        """
        def safe_int_parse(value: Optional[str]) -> Optional[int]:
            if value is None or value == "None":
                return None
            try:
                return int(value)
            except (ValueError, TypeError):
                return None

        return {
            "team_name": rating.get("club"),
            "country": rating.get("country"),
            "level": safe_int_parse(rating.get("level")),
            "elo": safe_int_parse(rating.get("elo")),
            "rank": safe_int_parse(rating.get("rank")),
            "from": rating.get("from"),
            "to": rating.get("to"),
            "is_current": is_current
        }
    
    def parse_response(self, response: requests.Response) -> Union[Dict[str, Any], List[Dict[str, Any]], str]:
        """
        Parses CSV response from the ClubElo API into a list of dictionaries.
        """
        content = response.text
        if not content.strip():
            return []
        
        try:
            # Use StringIO to treat the string content as a file
            csvfile = io.StringIO(content)
            reader = csv.DictReader(csvfile)
            # Convert all field names to lowercase for consistency
            return [{k.lower(): v for k, v in row.items()} for row in reader]
        except Exception as e:
            logger.error(f"Error parsing ClubElo CSV response: {e}")
            return content
    
    def test_connection(self) -> bool:
        """Test the API connection."""
        try:
            response = self.session.get(self.base_url)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_team_elo(
        self, 
        team_name: Optional[str] = None, 
        date: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """
        Get Elo ratings for a specific team.
        
        This endpoint provides the ranking history for a club. For the exact spelling of a club's name, 
        check the ranking. Values before 1960 should be considered provisional.
        
        Args:
            team_name: Name of the team (e.g., "Barcelona"). Must match the exact spelling from the ranking.
            date: Optional date in YYYY-MM-DD format to filter ratings
            
        Returns:
            List of dictionaries containing Elo rating data, or None if not found.
            Each rating includes historical data with from/to dates.
        """
        if not team_name:
            raise ValueError("team_name must be provided")

        endpoint = f"/{team_name}"

        try:
            response_list = cast(List[Dict[str, Any]], self.get_cached(endpoint))
            if not response_list:
                return None

            # Sort ratings by date (from oldest to newest)
            sorted_ratings = sorted(
                response_list,
                key=lambda x: datetime.strptime(cast(str, x.get("from", "")), "%Y-%m-%d").date()
            )

            # If a specific date is provided, filter ratings up to that date
            if date:
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
                filtered_ratings = [
                    rating for rating in sorted_ratings
                    if rating.get("from") and datetime.strptime(cast(str, rating.get("from")), "%Y-%m-%d").date() <= target_date
                ]
            else:
                filtered_ratings = sorted_ratings

            # Parse all ratings, marking the last one as current
            parsed_ratings = []
            for i, rating in enumerate(filtered_ratings):
                is_current = (i == len(filtered_ratings) - 1)
                parsed_ratings.append(self._parse_elo_rating(rating, is_current=is_current))

            return parsed_ratings

        except Exception as e:
            logger.error(f"Failed to get team Elo: {str(e)}")
            return None
    
    def get_top_teams(
        self,
        date: Optional[str] = None,
        limit: Optional[int] = None,
        country: Optional[str] = None,
        min_elo: Optional[int] = None
    ) -> List[Dict]:
        """
        Get top teams by Elo rating for a specific date.

        This endpoint provides the full ranking for each day since 1939. Values before 1960 
        should be considered provisional.

        Args:
            date: Date in YYYY-MM-DD format (default: latest available).
            limit: Maximum number of teams to return.
            country: Filter by country code (e.g., "ENG", "ESP").
            min_elo: Minimum Elo rating.

        Returns:
            List of top teams with their Elo ratings.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        endpoint = f"/{date}"

        try:
            response_list = cast(List[Dict[str, Any]], self.get_cached(endpoint))
            if not response_list:
                return []

            teams = []
            for team in response_list:
                if not team.get("club"):
                    continue

                if country and (team.get("country", '') or '').lower() != country.lower():
                    continue

                elo = float(cast(str, team.get("elo"))) if team.get("elo") is not None else 0
                if min_elo is not None and elo < min_elo:
                    continue

                parsed_rating = self._parse_elo_rating(team, is_current=True)
                teams.append(parsed_rating)
                
                if limit is not None and len(teams) >= limit:
                    break
            return teams

        except Exception as e:
            logger.error(f"Failed to get top teams: {str(e)}")
            return []
    
    def get_fixtures(self) -> List[Dict]:
        """
        Get calculated probabilities for all upcoming matches.

        This endpoint provides the calculated probabilities for all upcoming matches. Probabilities are given for:
        - All goal differences between -5 and +5
        - Aggregated probability for all goal differences smaller than -5 or bigger than +5
        - Each exact result with 6 or less total goals

        For traditional match odds (1X2):
        - Sum all negative goal differences for an away win
        - Sum all positive goal differences for a home win
        - Draw is goal difference = 0

        Returns:
            List of dictionaries with fixture probabilities.
        """
        endpoint = "/Fixtures"
        try:
            response_list = cast(List[Dict[str, Any]], self.get_cached(endpoint))
            if not response_list:
                return []
            return response_list
        except Exception as e:
            logger.error(f"Failed to get fixtures: {str(e)}")
            return []
