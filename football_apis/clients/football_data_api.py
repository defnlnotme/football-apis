import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time

from football_apis.base_client import BaseAPIClient, CachedAPIClient, RateLimitedAPIClient

logger = logging.getLogger(__name__)

class FootballDataClient(RateLimitedAPIClient):
    """
    Client for accessing team, player, match, and area data from the Football-Data API v4.
    
    This client provides methods to fetch team statistics, match history, standings, player stats, competition data, area/country info, and more.
    Many methods use caching for efficiency. Some endpoints may be restricted based on your API subscription (403 errors).
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_enabled: bool = True, cache_ttl: int = 3600):
        """
        Initialize the FootballDataClient.
        Args:
            api_key: API key for Football-Data API
            cache_enabled: Whether to enable caching
            cache_ttl: Cache TTL in seconds
        """
        super().__init__(
            api_key=api_key,
            base_url="http://api.football-data.org/v4",
            cache_enabled=cache_enabled,
            cache_ttl=cache_ttl
        )
    
    def test_connection(self) -> bool:
        """Test the connection to the Football-Data API."""
        try:
            response = self.get("/competitions")
            return isinstance(response, dict) and "competitions" in response
        except Exception as e:
            logger.error(f"Failed to connect to Football-Data API: {str(e)}")
            return False

    # --- Performance Stats Methods ---
    def get_team_statistics(self, team_id: int) -> Dict[str, Any]:
        """
        Get team details (no direct statistics endpoint in Football-Data.org v4 API).
        Note: The v4 API does not provide a direct team statistics endpoint. This method returns team details instead.
        Args:
            team_id: ID of the team
        Returns:
            Dictionary with team details or error message.
        """
        response = self.get(f"/teams/{team_id}")
        return response if isinstance(response, dict) else {"error": "Invalid response format"}

    def get_team_matches(self, team_id: int, season: Optional[int] = None, competition_id: Optional[int] = None, status: str = "FINISHED", limit: Optional[int] = None, competitions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get team matches for a specific season and competition, or recent matches for a team.
        Args:
            team_id: ID of the team
            season: Optional season year
            competition_id: Optional competition ID
            status: Match status (default 'FINISHED')
            limit: Optional max number of matches
            competitions: Optional list of competition codes
        Returns:
            Dictionary with matches or error message.
        """
        params: Dict[str, Union[str, int]] = {'status': status}
        if season:
            params['season'] = str(season)
        if competition_id:
            params['competition'] = str(competition_id)
        if limit is not None:
            params['limit'] = limit
        if competitions:
            params['competitions'] = ','.join(competitions)
        response = self.get(f"/teams/{team_id}/matches", params=params)
        return response if isinstance(response, dict) else {"error": "Invalid response format"}

    def get_team_standings(self, team_id: int, competition_id: int, season: Optional[int] = None) -> Dict[str, Any]:
        """
        Get team standings in a competition.
        Args:
            team_id: ID of the team
            competition_id: ID of the competition
            season: Optional season year
        Returns:
            Dictionary with standings or error message.
        """
        params: Dict[str, Union[str, int]] = {}
        if season:
            params['season'] = str(season)
        response = self.get(f"/competitions/{competition_id}/standings", params=params)
        return response if isinstance(response, dict) else {"error": "Invalid response format"}

    def get_player_statistics(self, player_id: int, season: Optional[int] = None, competition_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get player statistics for a specific season and competition.
        Args:
            player_id: ID of the player
            season: Optional season year
            competition_id: Optional competition ID
        Returns:
            Dictionary with player statistics or error message.
        """
        params: Dict[str, Union[str, int]] = {}
        if season:
            params['season'] = str(season)
        if competition_id:
            params['competition'] = str(competition_id)
        response = self.get(f"/players/{player_id}/statistics", params=params)
        return response if isinstance(response, dict) else {"error": "Invalid response format"}

    # --- Match History Methods ---
    def get_head_to_head(self, team1_id: int, team2_id: int, limit: int = 10, date_from: Optional[str] = None, date_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Get head-to-head matches between two teams.
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            limit: Max number of matches (default 10)
            date_from: Optional start date (YYYY-MM-DD)
            date_to: Optional end date (YYYY-MM-DD)
        Returns:
            Dictionary with head-to-head matches or error message.
        """
        params: Dict[str, Union[str, int]] = {'limit': limit}
        if date_from:
            params['dateFrom'] = date_from
        if date_to:
            params['dateTo'] = date_to
        response = self.get(f"/teams/{team1_id}/matches", params=params)
        return response if isinstance(response, dict) else {"error": "Invalid response format"}

    def get_team_info(self, team_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a team.
        Args:
            team_id: ID of the team
        Returns:
            Dictionary with team info or error message.
        """
        response = self.get(f"/teams/{team_id}")
        return response if isinstance(response, dict) else {"error": "Invalid response format"}

    def search_teams(self, name: Optional[str] = None, season: Optional[int] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> Dict[str, Any]:
        """
        Search for teams by name or get a list of teams. If offset=-1, fetches all teams by paginating through the API.
        Args:
            name: Optional team name to search
            season: Optional season year
            limit: Optional max number of teams
            offset: Optional offset for pagination (use -1 to fetch all)
        Returns:
            Dictionary with teams or error message.
        """
        if offset == -1:
            # Fetch all teams by paginating
            all_teams = []
            current_offset = 0
            page_limit = limit if limit else 50  # Use provided limit or default to 50 per page
            while True:
                page_params: Dict[str, Union[str, int]] = {}
                if name:
                    page_params['name'] = name
                if season:
                    page_params['season'] = str(season)
                page_params['limit'] = page_limit
                page_params['offset'] = current_offset
                try:
                    response, from_cache = self.get_cached("/teams", params=page_params, return_cache_status=True)
                except Exception as e:
                    logger.error(f"Error fetching teams at offset {current_offset}: {e}")
                    break
                if not (isinstance(response, dict) and 'teams' in response):
                    break
                teams = response['teams']
                all_teams.extend(teams)
                if len(teams) < page_limit:
                    break
                current_offset += len(teams)
            return {'teams': all_teams}
        else:
            params: Dict[str, Union[str, int]] = {}
            if name:
                params['name'] = name
            if season:
                params['season'] = str(season)
            if limit:
                params['limit'] = limit
            if offset:
                params['offset'] = offset
            response = self.get_cached("/teams", params=params)
            return response if isinstance(response, dict) else {"error": "Invalid response format"}

    def get_competition_matches(self, competition_id: int, season: Optional[int] = None, matchday: Optional[int] = None) -> Dict[str, Any]:
        """
        Get matches for a specific competition.
        Args:
            competition_id: ID of the competition
            season: Optional season year
            matchday: Optional matchday number
        Returns:
            Dictionary with matches or error message.
        """
        params: Dict[str, Union[str, int]] = {}
        if season:
            params['season'] = str(season)
        if matchday:
            params['matchday'] = str(matchday)
        response = self.get(f"/competitions/{competition_id}/matches", params=params)
        return response if isinstance(response, dict) else {"error": "Invalid response format"}

    def get_match_details(self, match_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific match.
        Args:
            match_id: ID of the match
        Returns:
            Dictionary with match details or error message.
        Note:
            Some matches may be restricted and return 403 errors depending on your API subscription.
        """
        response = self.get(f"/matches/{match_id}")
        return response if isinstance(response, dict) else {"error": "Invalid response format"} 

    # --- Person (Player/Staff) Methods ---
    def get_person(self, person_id: int) -> Dict[str, Any]:
        """
        Get details for a specific person (player, staff, referee, etc).
        Args:
            person_id: ID of the person
        Returns:
            Dictionary with person details or error message.
        """
        response = self.get(f"/persons/{person_id}")
        return response if isinstance(response, dict) else {"error": "Invalid response format"}

    def get_person_matches(self, person_id: int, lineup: Optional[str] = None, e: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None, competitions: Optional[str] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> Dict[str, Any]:
        """
        Get matches for a specific person (player, staff, referee, etc).
        Supports filters: lineup, e, dateFrom, dateTo, competitions, limit, offset.
        Args:
            person_id: ID of the person
            lineup: Optional lineup filter (STARTING, BENCH)
            e: Optional event type (GOAL, ASSIST, SUB_IN, SUB_OUT)
            date_from: Optional start date
            date_to: Optional end date
            competitions: Optional competitions filter
            limit: Optional max number of matches
            offset: Optional offset for pagination
        Returns:
            Dictionary with matches or error message.
        """
        params: Dict[str, Union[str, int]] = {}
        if lineup:
            params["lineup"] = lineup
        if e:
            params["e"] = e
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to
        if competitions:
            params["competitions"] = competitions
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        response = self.get(f"/persons/{person_id}/matches", params=params)
        return response if isinstance(response, dict) else {"error": "Invalid response format"}

    # --- Area (Country/Region) Methods ---
    def get_areas(self) -> Dict[str, Any]:
        """
        Get a list of all areas (countries, regions, etc), using cache for efficiency.
        Returns:
            Dictionary with list of areas or error message.
        """
        response = self.get_cached("/areas")
        return response if isinstance(response, dict) else {"error": "Invalid response format"}

    def get_area(self, area_id: int) -> Dict[str, Any]:
        """
        Get details for a specific area by its ID. Tries to find the area in cached get_areas first for efficiency.
        Args:
            area_id: ID of the area
        Returns:
            Dictionary with area details or error message.
        """
        areas_response = self.get_areas()
        if isinstance(areas_response, dict) and "areas" in areas_response:
            for area in areas_response["areas"]:
                if area.get("id") == area_id:
                    return area
        # Fallback to API call if not found in cache
        response = self.get(f"/areas/{area_id}")
        return response if isinstance(response, dict) else {"error": "Invalid response format"}