from typing import Dict, List, Optional, Union, Any, cast
from datetime import datetime, timedelta
import requests
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
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
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
        team_name: str = None, 
        team_id: int = None,
        date: str = None
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
            teams = self.search_teams(team_name)
            if not teams:
                logger.warning(f"No team found with name: {team_name}")
                return None
                
            # Get the first match
            team = teams[0]
            team_id = team.get("id")
            if not team_id:
                return None
        
        # Get team details including current Elo
        endpoint = f"/{team_id}"
        
        try:
            response = self.get_cached(endpoint)
            if not response:
                return None
                
            # If a specific date is provided, find the closest historical rating
            if date:
                historical = response.get("history", [])
                if not historical:
                    return None
                    
                # Find the most recent rating on or before the specified date
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
                closest_rating = None
                
                for rating in historical:
                    rating_date = datetime.strptime(rating.get("from"), "%Y-%m-%d").date()
                    if rating_date > target_date:
                        continue
                    if closest_rating is None or rating_date > datetime.strptime(closest_rating.get("from"), "%Y-%m-%d").date():
                        closest_rating = rating
                
                if closest_rating:
                    return {
                        "team_id": team_id,
                        "team_name": response.get("team"),
                        "country": response.get("country"),
                        "level": response.get("level"),
                        "elo": closest_rating.get("elo"),
                        "rank": closest_rating.get("rank"),
                        "from": closest_rating.get("from"),
                        "to": closest_rating.get("to"),
                        "is_current": False
                    }
            
            # Return current rating if no date specified or no historical data found
            return {
                "team_id": team_id,
                "team_name": response.get("team"),
                "country": response.get("country"),
                "level": response.get("level"),
                "elo": response.get("elo"),
                "rank": response.get("rank"),
                "from": response.get("from"),
                "to": response.get("to"),
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
            response = self.get_cached(endpoint, params=params)
            if not isinstance(response, list):
                return []
                
            return [
                {
                    "id": team.get("id"),
                    "name": team.get("team"),
                    "country": team.get("country"),
                    "elo": team.get("elo"),
                    "rank": team.get("rank")
                }
                for team in response
                if team.get("id") and team.get("team")
            ]
            
        except Exception as e:
            logger.error(f"Team search failed: {str(e)}")
            return []
    
    def get_top_teams(
        self, 
        limit: int = 20, 
        country: str = None, 
        min_elo: int = None
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
            response = self.get_cached(endpoint)
            if not isinstance(response, list):
                return []
                
            teams = []
            for team in response:
                if not team.get("id") or not team.get("team"):
                    continue
                    
                if country and team.get("country") != country:
                    continue
                    
                if min_elo is not None and team.get("elo", 0) < min_elo:
                    continue
                    
                teams.append({
                    "rank": team.get("rank"),
                    "team_id": team.get("id"),
                    "team_name": team.get("team"),
                    "country": team.get("country"),
                    "elo": team.get("elo"),
                    "level": team.get("level")
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
        start_date: str = None, 
        end_date: str = None
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
            response = self.get_cached(endpoint)
            if not response or not isinstance(response.get("history"), list):
                return []
                
            history = response["history"]
            
            # Filter by date range if specified
            if start_date or end_date:
                start = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else datetime.min.date()
                end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else datetime.max.date()
                
                filtered = []
                for entry in history:
                    entry_date = datetime.strptime(entry.get("from"), "%Y-%m-%d").date()
                    if start <= entry_date <= end:
                        filtered.append(entry)
                history = filtered
                
            return [
                {
                    "date": entry.get("from"),
                    "elo": entry.get("elo"),
                    "rank": entry.get("rank"),
                    "matches": entry.get("matches")
                }
                for entry in history
            ]
            
        except Exception as e:
            logger.error(f"Failed to get historical Elos: {str(e)}")
            return []
    
    def get_team_form(
        self, 
        team_id: int, 
        matches: int = 5,
        competition: str = None
    ) -> Dict:
        """
        Get recent form for a team based on Elo changes.
        
        Args:
            team_id: ClubElo team ID
            matches: Number of recent matches to include
            competition: Filter by competition (if available)
            
        Returns:
            Dictionary with form information
        """
        endpoint = f"/{team_id}"
        
        try:
            response = self.get_cached(endpoint)
            if not response or not isinstance(response.get("matches"), list):
                return {"team_id": team_id, "recent_matches": [], "form_rating": 0}
                
            matches_data = response["matches"]
            
            # Filter by competition if specified
            if competition:
                matches_data = [
                    m for m in matches_data 
                    if m.get("competition") and competition.lower() in m.get("competition", "").lower()
                ]
            
            # Get most recent matches
            recent_matches = sorted(
                matches_data,
                key=lambda x: x.get("date", ""),
                reverse=True
            )[:matches]
            
            # Calculate form rating (points per match: 3 for win, 1 for draw, 0 for loss)
            points = 0
            for match in recent_matches:
                result = match.get("result")
                if result == "W":
                    points += 3
                elif result == "D":
                    points += 1
            
            form_rating = (points / (matches * 3)) * 100  # As percentage
            
            return {
                "team_id": team_id,
                "team_name": response.get("team"),
                "recent_matches": recent_matches,
                "form_rating": round(form_rating, 1),
                "matches_played": len(recent_matches),
                "wins": sum(1 for m in recent_matches if m.get("result") == "W"),
                "draws": sum(1 for m in recent_matches if m.get("result") == "D"),
                "losses": sum(1 for m in recent_matches if m.get("result") == "L")
            }
            
        except Exception as e:
            logger.error(f"Failed to get team form: {str(e)}")
            return {"team_id": team_id, "recent_matches": [], "form_rating": 0}
