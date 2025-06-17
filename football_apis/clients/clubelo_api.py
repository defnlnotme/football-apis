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
    
    def __init__(self, api_key: Optional[str] = None, cache_enabled: bool = True, cache_ttl: int = 86400):
        """
        Initialize the ClubEloClient.
        
        Args:
            api_key: API key for authentication (if required)
            cache_enabled: Whether to enable response caching
            cache_ttl: Time-to-live for cache entries in seconds (default: 24 hours)
        """
        super().__init__(
            api_key=api_key,
            base_url="http://api.clubelo.com",
            cache_enabled=cache_enabled,
            cache_ttl=cache_ttl
        )
        
        # Ensure session is properly configured
        self.session.headers.update({
            'X-Auth-Token': self.api_key or '',
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
    
    def _add_auth(self):
        """Add authentication to the request."""
        if self.api_key:
            self.session.headers.update({"X-Auth-Token": self.api_key})
    
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
        team_id: Optional[int] = None,
        date: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get Elo rating for a specific team.
        
        Args:
            team_name: Name of the team (e.g., "Barcelona")
            team_id: ClubElo team ID (alternative to team_name)
            date: Date in YYYY-MM-DD format (default: latest available)
            
        Returns:
            Dictionary containing Elo rating data or None if not found
        """
        if not team_name and not team_id:
            raise ValueError("Either team_name or team_id must be provided")
            
        if team_name:
            # Search for team by name
            teams = self.search_teams(query=team_name)
            if not teams:
                logger.warning(f"No team found with name: {team_name}")
                return None
                
            # Get the first match (assuming search_teams returns matches sorted by relevance or Elo)
            team = teams[0]
            team_id = int(cast(str, team.get("id"))) if team.get("id") is not None else None
            if team_id is None:
                return None
        
        # Clubelo API returns historical data for a team directly via /{team_id}
        # The `get_cached` method will call `get`, which will use `parse_response`
        # to get a list of dictionaries from the CSV.
        if team_id is None:
            return None # Should not happen if previous checks are correct, but for type safety
        endpoint = f"/{str(team_id)}"
        
        try:
            # response here will be a list of dicts (parsed from CSV)
            response_list = cast(List[Dict[str, Any]], self.get_cached(endpoint))
            if not response_list:
                return None
            
            # Find the most recent rating for the team
            latest_rating = None
            for rating in response_list:
                if latest_rating is None or (rating.get("to") and latest_rating.get("to") and datetime.strptime(cast(str, rating.get("to")), "%Y-%m-%d") > datetime.strptime(cast(str, latest_rating.get("to")), "%Y-%m-%d")):
                    latest_rating = rating
            
            if not latest_rating:
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
                        "team_id": team_id,
                        "team_name": closest_rating.get("club"),
                        "country": closest_rating.get("country"),
                        "level": int(cast(str, closest_rating.get("level"))) if closest_rating.get("level") is not None else None,
                        "elo": int(cast(str, closest_rating.get("elo"))) if closest_rating.get("elo") is not None else None,
                        "rank": int(cast(str, closest_rating.get("rank"))) if closest_rating.get("rank") is not None else None,
                        "from": closest_rating.get("from"),
                        "to": closest_rating.get("to"),
                        "is_current": False
                    }
            
            # Return current rating if no date specified or no historical data found
            return {
                "team_id": team_id,
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
    
    def search_teams(self, query: str) -> List[Dict]:
        """
        Search for teams by name.
        
        Args:
            query: Search query (team name or part of it)
            
        Returns:
            List of matching teams with their IDs and names
        """
        endpoint = "/search"
        params = {"q": query}
        
        try:
            # response here will be a list of dicts (parsed from CSV)
            response_list = cast(List[Dict[str, Any]], self.get_cached(endpoint, params=params))
            if not response_list:
                return []
                
            # Filter teams by query (case-insensitive, partial match)
            # ClubElo's /search endpoint actually returns an exact match or similar, but the data needs to be extracted.
            # The `parse_response` will convert to lowercase keys. The 'club' key holds the team name.
            return [
                {
                    "id": int(cast(str, team.get("id"))) if team.get("id") is not None else None,
                    "name": team.get("club"),
                    "country": team.get("country"),
                    "elo": int(cast(str, team.get("elo"))) if team.get("elo") is not None else None,
                    "rank": int(cast(str, team.get("rank"))) if team.get("rank") is not None else None
                }
                for team in response_list
                if team.get("id") and team.get("club") and query.lower() in (team.get("club") or '').lower()
            ]
            
        except Exception as e:
            logger.error(f"Team search failed: {str(e)}")
            return []
    
    def get_top_teams(
        self, 
        limit: int = 20, 
        country: Optional[str] = None, 
        min_elo: Optional[int] = None
    ) -> List[Dict]:
        """
        Get top teams by Elo rating.
        
        Args:
            limit: Maximum number of teams to return
            country: Filter by country code (e.g., "ENG", "ESP")
            min_elo: Minimum Elo rating
            
        Returns:
            List of top teams with their Elo ratings
        """
        endpoint = "/ranking"
        
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
    
    def get_historical_elos(
        self, 
        team_id: int, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get historical Elo ratings for a team over a date range.
        
        Args:
            team_id: ClubElo team ID
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (default: today)
            
        Returns:
            List of historical Elo ratings with dates
        """
        endpoint = f"/{team_id}"
        
        try:
            # response here will be a list of dicts (parsed from CSV)
            response_list = cast(List[Dict[str, Any]], self.get_cached(endpoint))
            if not response_list:
                return []
                
            history = []
            for rating in response_list:
                # ClubElo API provides 'from' and 'to' dates in the CSV for historical data
                # The 'club' key represents the team name
                history.append({
                    "date_from": rating.get("from"),
                    "date_to": rating.get("to"),
                    "elo": int(cast(str, rating.get("elo"))) if rating.get("elo") is not None else None,
                    "rank": int(cast(str, rating.get("rank"))) if rating.get("rank") is not None else None,
                    "club": rating.get("club")
                })

            # Filter by date range if specified
            filtered_history = []
            start_dt = datetime.strptime(cast(str, start_date), "%Y-%m-%d").date() if start_date else None
            end_dt_str = cast(str, end_date) if end_date is not None else datetime.now().strftime("%Y-%m-%d")
            end_dt = datetime.strptime(end_dt_str, "%Y-%m-%d").date()
            
            for entry in history:
                entry_date_from = datetime.strptime(cast(str, entry.get("date_from")), "%Y-%m-%d").date() if entry.get("date_from") else None
                entry_date_to = datetime.strptime(cast(str, entry.get("date_to")), "%Y-%m-%d").date() if entry.get("date_to") else None
                
                if (start_dt is None or (entry_date_to and start_dt and entry_date_to >= start_dt)) and \
                   (end_dt is None or (entry_date_from and end_dt and entry_date_from <= end_dt)):
                    filtered_history.append(entry)

            return filtered_history
            
        except Exception as e:
            logger.error(f"Failed to get historical Elos: {str(e)}")
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
