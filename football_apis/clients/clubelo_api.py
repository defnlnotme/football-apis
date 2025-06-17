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
    ) -> Optional[Dict]:
        """
        Get Elo rating for a specific team.
        
        Args:
            team_name: Name of the team (e.g., "Barcelona")
            date: Date in YYYY-MM-DD format (default: latest available)
            
        Returns:
            Dictionary containing Elo rating data or None if not found
        """
        if not team_name:
            raise ValueError("team_name must be provided")

        # Clubelo API returns historical data for a team directly via /CLUBNAME
        # The `get_cached` method will call `get`, which will use `parse_response`
        # to get a list of dictionaries from the CSV.
        endpoint = f"/{team_name}"

        try:
            # response here will be a list of dicts (parsed from CSV)
            response_list = cast(List[Dict[str, Any]], self.get_cached(endpoint))
            if not response_list:
                return None

            # If a specific date is provided, find the closest historical rating
            if date:
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
                closest_rating = None

                for rating in response_list:
                    # Use "from" date for comparison as "to" might be in the future
                    rating_from_date = datetime.strptime(cast(str, rating.get("from")), "%Y-%m-%d").date() if rating.get("from") else None
                    if rating_from_date and rating_from_date <= target_date:
                        if closest_rating is None or (rating.get("from") and closest_rating.get("from") and datetime.strptime(cast(str, rating.get("from")), "%Y-%m-%d").date() > datetime.strptime(cast(str, closest_rating.get("from")), "%Y-%m-%d").date()):
                            closest_rating = rating

                if closest_rating:
                    return {
                        "team_name": closest_rating.get("club"),
                        "country": closest_rating.get("country"),
                        "level": int(cast(str, closest_rating.get("level"))) if closest_rating.get("level") is not None else None,
                        "elo": int(cast(str, closest_rating.get("elo"))) if closest_rating.get("elo") is not None else None,
                        "rank": int(cast(str, closest_rating.get("rank"))) if closest_rating.get("rank") is not None else None,
                        "from": closest_rating.get("from"),
                        "to": closest_rating.get("to"),
                        "is_current": False
                    }

            # If no date specified or no historical data found, return the latest rating
            latest_rating = response_list[-1] # Assuming the last entry is the most recent
            return {
                "team_name": latest_rating.get("club"),
                "country": latest_rating.get("country"),
                "level": int(cast(str, latest_rating.get("level"))) if latest_rating.get("level") is not None else None,
                "elo": int(cast(str, latest_rating.get("elo"))) if latest_rating.get("elo") is not None else None,
                "rank": int(cast(str, latest_rating.get("rank"))) if latest_rating.get("rank") is not None else None,
                "from": latest_rating.get("from"),
                "to": latest_rating.get("to"),
                "is_current": True
            }

        except Exception as e:
            logger.error(f"Failed to get team Elo: {str(e)}")
            return None
    
    def get_top_teams(
        self,
        date: Optional[str] = None,
        limit: int = 20,
        country: Optional[str] = None,
        min_elo: Optional[int] = None
    ) -> List[Dict]:
        """
        Get top teams by Elo rating for a specific date.

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
            # response here will be a list of dicts (parsed from CSV)
            response_list = cast(List[Dict[str, Any]], self.get_cached(endpoint))
            if not response_list:
                return []

            teams = []
            for team in response_list:
                # Ensure 'id' and 'club' (team name) exist
                if not team.get("id") or not team.get("club"):
                    continue

                if country and (team.get("country", '') or '').lower() != country.lower():
                    continue

                elo = int(cast(str, team.get("elo"))) if team.get("elo") is not None else 0
                if min_elo is not None and elo < min_elo:
                    continue

                teams.append({
                    "rank": int(cast(str, team.get("rank"))) if team.get("rank") is not None else None,
                    "team_id": int(cast(str, team.get("id"))) if team.get("id") is not None else None,
                    "team_name": team.get("club"),
                    "country": team.get("country"),
                    "elo": elo,
                    "level": int(cast(str, team.get("level"))) if team.get("level") is not None else None
                })

                if len(teams) >= limit:
                    break

            return teams

        except Exception as e:
            logger.error(f"Failed to get top teams: {str(e)}")
            return []
    
    def get_team_form(
        self, 
        team_id: int, 
        matches: int = 5,
        competition: Optional[str] = None
    ) -> Dict:
        """
        Get recent form (win/loss/draw) for a team.
        
        Args:
            team_id: ClubElo team ID
            matches: Number of recent matches to consider
            competition: Optional filter by competition name
            
        Returns:
            Dictionary with win/loss/draw counts and goal difference
        """
        # This method is more complex as it requires match data. 
        # ClubElo API primarily provides Elo ratings, not detailed match results.
        # This method might need to fetch data from another source or be reconsidered.
        logger.warning("get_team_form is not fully implemented with ClubElo API. It provides Elo ratings, not detailed match results.")
        return {"error": "Method not fully implemented for ClubElo API."}

    def get_fixtures(self) -> List[Dict]:
        """
        Get calculated probabilities for all upcoming matches.

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
