from typing import Dict, List, Optional, Union
import requests
from datetime import datetime, timedelta
from ..base_client import BaseAPIClient, CachingMixin, RateLimitMixin
import logging

logger = logging.getLogger(__name__)

class PerformanceStatsClient(CachingMixin, RateLimitMixin, BaseAPIClient):
    """Client for accessing team and player performance statistics."""
    
    def __init__(self, api_key: str = None, cache_enabled: bool = True, cache_ttl: int = 86400):
        """
        Initialize the PerformanceStatsClient.
        
        Args:
            api_key: API key for authentication (if required)
            cache_enabled: Whether to enable response caching
            cache_ttl: Time-to-live for cache entries in seconds (default: 24 hours)
        """
        # Initialize BaseAPIClient first
        BaseAPIClient.__init__(
            self,
            api_key=api_key,
            base_url="https://api.football-data.org/v4"
        )
        
        # Initialize RateLimitMixin
        RateLimitMixin.__init__(self)
        
        # Initialize CachingMixin with provided parameters
        CachingMixin.__init__(self, cache_enabled=cache_enabled, cache_ttl=cache_ttl)
        
        # Ensure session is properly configured
        self.session.headers.update({
            'X-Auth-Token': self.api_key or '',
            'Content-Type': 'application/json'
        })
    
    def test_connection(self) -> bool:
        """Test the API connection."""
        try:
            response = self.get("/")
            return "teams" in response.get("_links", {})
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_team_statistics(
        self, 
        team_id: int, 
        season: Optional[int] = None,
        competition_id: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Get performance statistics for a specific team.
        
        Args:
            team_id: ID of the team
            season: Season year (e.g., 2023 for 2023/2024 season)
            competition_id: ID of the competition (optional)
            
        Returns:
            Dictionary containing team statistics
        """
        if season is None:
            season = datetime.now().year
            
        endpoint = f"/teams/{team_id}"
        params = {"season": season}
        
        if competition_id:
            params["competitions"] = competition_id
            
        team_data = self.get_cached(endpoint, params=params)
        
        if not team_data:
            return {}
            
        # Extract relevant statistics
        stats = {
            "team_id": team_id,
            "team_name": team_data.get("name"),
            "season": season,
            "competition_id": competition_id,
            "stats": {}
        }
        
        # Get team statistics from the squad endpoint
        try:
            matches = self.get_team_matches(team_id, season=season, competition_id=competition_id)
            
            # Calculate basic statistics
            total_matches = len(matches)
            if total_matches == 0:
                return stats
                
            # Initialize counters
            goals_for = 0
            goals_against = 0
            shots_total = 0
            shots_on_target = 0
            clean_sheets = 0
            xg_for = 0.0
            xg_against = 0.0
            
            for match in matches:
                # Determine if team is home or away
                is_home = match["homeTeam"]["id"] == team_id
                
                # Get match stats
                home_stats = match.get("homeTeamStats", {})
                away_stats = match.get("awayTeamStats", {})
                
                team_stats = home_stats if is_home else away_stats
                opponent_stats = away_stats if is_home else home_stats
                
                # Aggregate stats
                goals_for += team_stats.get("goals", 0)
                goals_against += opponent_stats.get("goals", 0)
                shots_total += team_stats.get("shots", {}).get("total", 0)
                shots_on_target += team_stats.get("shots", {}).get("on", 0)
                
                # xG (expected goals) if available
                if "expectedGoals" in team_stats:
                    xg_for += float(team_stats["expectedGoals"] or 0)
                if "expectedGoals" in opponent_stats:
                    xg_against += float(opponent_stats["expectedGoals"] or 0)
                
                # Clean sheets
                if (is_home and match["score"]["fullTime"]["homeTeam"] == 0) or \
                   (not is_home and match["score"]["fullTime"]["awayTeam"] == 0):
                    clean_sheets += 1
            
            # Calculate averages
            stats["stats"].update({
                "matches_played": total_matches,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "goals_per_match": round(goals_for / total_matches, 2) if total_matches > 0 else 0,
                "goals_against_per_match": round(goals_against / total_matches, 2) if total_matches > 0 else 0,
                "clean_sheets": clean_sheets,
                "clean_sheet_percentage": round((clean_sheets / total_matches) * 100, 2) if total_matches > 0 else 0,
                "shots_per_match": round(shots_total / total_matches, 2) if total_matches > 0 else 0,
                "shots_on_target_per_match": round(shots_on_target / total_matches, 2) if total_matches > 0 else 0,
                "xg_per_match": round(xg_for / total_matches, 2) if total_matches > 0 else 0,
                "xg_against_per_match": round(xg_against / total_matches, 2) if total_matches > 0 else 0,
            })
            
        except Exception as e:
            logger.error(f"Error calculating team statistics: {str(e)}")
        
        return stats
    
    def get_team_matches(
        self, 
        team_id: int, 
        season: Optional[int] = None,
        competition_id: Optional[int] = None,
        status: str = "FINISHED"
    ) -> List[Dict]:
        """
        Get matches for a specific team.
        
        Args:
            team_id: ID of the team
            season: Season year (e.g., 2023 for 2023/2024 season)
            competition_id: ID of the competition (optional)
            status: Match status (e.g., "FINISHED", "SCHEDULED")
            
        Returns:
            List of matches
        """
        if season is None:
            season = datetime.now().year
            
        endpoint = f"/teams/{team_id}/matches"
        params = {
            "season": season,
            "status": status
        }
        
        if competition_id:
            params["competitions"] = competition_id
            
        try:
            response = self.get_cached(endpoint, params=params)
            return response.get("matches", [])
        except Exception as e:
            logger.error(f"Failed to get team matches: {str(e)}")
            return []
    
    def get_team_standings(
        self, 
        team_id: int, 
        competition_id: int, 
        season: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Get league standings/position for a team in a specific competition.
        
        Args:
            team_id: ID of the team
            competition_id: ID of the competition
            season: Season year (e.g., 2023 for 2023/2024 season)
            
        Returns:
            Dictionary containing team's standing information or None if not found
        """
        if season is None:
            season = datetime.now().year
            
        endpoint = f"/competitions/{competition_id}/standings"
        params = {"season": season}
        
        try:
            response = self.get_cached(endpoint, params=params)
            
            # Find the team in the standings
            for standing in response.get("standings", []):
                if standing["type"] == "TOTAL":
                    for table in standing.get("table", []):
                        if table["team"]["id"] == team_id:
                            return {
                                "position": table["position"],
                                "played_games": table["playedGames"],
                                "won": table["won"],
                                "draw": table["draw"],
                                "lost": table["lost"],
                                "points": table["points"],
                                "goals_for": table["goalsFor"],
                                "goals_against": table["goalsAgainst"],
                                "goal_difference": table["goalDifference"],
                                "form": table.get("form", ""),
                                "competition": response.get("competition", {}).get("name"),
                                "season": season
                            }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get team standings: {str(e)}")
            return None
    
    def get_player_statistics(
        self, 
        player_id: int, 
        season: Optional[int] = None,
        competition_id: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Get performance statistics for a specific player.
        
        Note: This is a placeholder as the free tier of Football-Data.org doesn't provide detailed player stats.
        In a production environment, you would use a different API like API-Football or Wyscout.
        
        Args:
            player_id: ID of the player
            season: Season year (e.g., 2023 for 2023/2024 season)
            competition_id: ID of the competition (optional)
            
        Returns:
            Dictionary containing player statistics
        """
        # This is a simplified version. In a real implementation, you would use a different API
        # that provides detailed player statistics.
        
        logger.warning("Detailed player statistics are not available in the free tier. "
                     "Consider using a different API like API-Football or Wyscout.")
        
        return {
            "player_id": player_id,
            "season": season,
            "competition_id": competition_id,
            "stats": {}
        }
