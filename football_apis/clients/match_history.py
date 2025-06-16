from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from ..base_client import BaseAPIClient, CachingMixin, RateLimitMixin
import logging

logger = logging.getLogger(__name__)

class MatchHistoryClient(BaseAPIClient, CachingMixin, RateLimitMixin):
    """Client for accessing match history and head-to-head data."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the MatchHistoryClient.
        
        Args:
            api_key: API key for authentication (if required)
        """
        super().__init__(
            api_key=api_key,
            base_url="https://api.football-data.org/v4"
        )
        self.cache_ttl = 3600  # 1 hour cache TTL for match data
    
    def test_connection(self) -> bool:
        """Test the API connection."""
        try:
            response = self.get("/")
            return "matches" in response.get("_links", {})
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_team_matches(
        self, 
        team_id: int, 
        limit: int = 8,
        competitions: Optional[List[str]] = None,
        status: str = "FINISHED"
    ) -> List[Dict]:
        """
        Get recent matches for a specific team.
        
        Args:
            team_id: ID of the team
            limit: Maximum number of matches to return (default: 8)
            competitions: Filter by competition codes (e.g., ["PL", "CL"])
            status: Filter by match status (e.g., "FINISHED", "SCHEDULED")
            
        Returns:
            List of match dictionaries
        """
        params = {
            "limit": limit,
            "status": status
        }
        
        if competitions:
            params["competitions"] = ",".join(competitions)
        
        endpoint = f"/teams/{team_id}/matches"
        response = self.get_cached(endpoint, params=params)
        return response.get("matches", [])
    
    def get_head_to_head(
        self, 
        team1_id: int, 
        team2_id: int, 
        limit: int = 10,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict:
        """
        Get head-to-head statistics between two teams.
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            limit: Maximum number of matches to return (default: 10)
            date_from: Filter matches after this date (YYYY-MM-DD)
            date_to: Filter matches before this date (YYYY-MM-DD)
            
        Returns:
            Dictionary containing head-to-head statistics and matches
        """
        # Default to matches from the last 3 years
        if date_from is None:
            date_from = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")
        if date_to is None:
            date_to = datetime.now().strftime("%Y-%m-%d")
        
        # Get matches for team1
        team1_matches = self.get_team_matches(
            team_id=team1_id,
            limit=100,  # Get more matches to find H2H
            status="FINISHED"
        )
        
        # Filter for matches against team2
        h2h_matches = [
            match for match in team1_matches 
            if (match["homeTeam"]["id"] == team2_id or 
                 match["awayTeam"]["id"] == team2_id)
        ][:limit]
        
        # Calculate basic statistics
        team1_wins = 0
        team2_wins = 0
        draws = 0
        
        for match in h2h_matches:
            if match["score"]["winner"] == "HOME_TEAM":
                if match["homeTeam"]["id"] == team1_id:
                    team1_wins += 1
                else:
                    team2_wins += 1
            elif match["score"]["winner"] == "AWAY_TEAM":
                if match["awayTeam"]["id"] == team1_id:
                    team1_wins += 1
                else:
                    team2_wins += 1
            else:
                draws += 1
        
        return {
            "team1_id": team1_id,
            "team2_id": team2_id,
            "total_matches": len(h2h_matches),
            "team1_wins": team1_wins,
            "team2_wins": team2_wins,
            "draws": draws,
            "matches": h2h_matches,
            "date_range": {
                "from": date_from,
                "to": date_to
            }
        }
    
    def get_team_info(self, team_id: int) -> Optional[Dict]:
        """
        Get information about a specific team.
        
        Args:
            team_id: ID of the team
            
        Returns:
            Team information dictionary or None if not found
        """
        try:
            return self.get(f"/teams/{team_id}")
        except Exception as e:
            logger.error(f"Failed to get team info for ID {team_id}: {str(e)}")
            return None
    
    def search_teams(self, name: str) -> List[Dict]:
        """
        Search for teams by name.
        
        Args:
            name: Team name to search for
            
        Returns:
            List of matching teams
        """
        try:
            response = self.get("/teams", params={"name": name})
            return response.get("teams", [])
        except Exception as e:
            logger.error(f"Team search failed for '{name}': {str(e)}")
            return []
    
    def get_competition_matches(
        self, 
        competition_id: int, 
        season: Optional[int] = None,
        matchday: Optional[int] = None
    ) -> List[Dict]:
        """
        Get matches for a specific competition.
        
        Args:
            competition_id: ID of the competition
            season: Season year (e.g., 2023 for 2023/2024 season)
            matchday: Specific matchday
            
        Returns:
            List of matches
        """
        endpoint = f"/competitions/{competition_id}/matches"
        params = {}
        
        if season is not None:
            params["season"] = season
        if matchday is not None:
            params["matchday"] = matchday
            
        response = self.get_cached(endpoint, params=params)
        return response.get("matches", [])
    
    def get_match_details(self, match_id: int) -> Optional[Dict]:
        """
        Get detailed information about a specific match.
        
        Args:
            match_id: ID of the match
            
        Returns:
            Match details or None if not found
        """
        try:
            return self.get(f"/matches/{match_id}")
        except Exception as e:
            logger.error(f"Failed to get match details for ID {match_id}: {str(e)}")
            return None
