"""Tests for the MatchHistoryClient."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

class TestMatchHistoryClient:
    """Test cases for MatchHistoryClient."""

    def test_initialization(self, match_history_client):
        """Test client initialization with API key."""
        assert match_history_client is not None
        assert match_history_client.base_url == "https://api.football-data.org/v4"
        assert match_history_client.api_key == "test_api_key"
        assert "X-Auth-Token" in match_history_client.session.headers
        assert match_history_client.session.headers["X-Auth-Token"] == "test_api_key"

    def test_test_connection_success(self, match_history_client, mock_requests_get):
        """Test successful connection to the API."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {"_links": {"matches": "https://api.football-data.org/v4/matches"}}
        mock_requests_get.return_value = mock_response

        assert match_history_client.test_connection() is True
        mock_requests_get.assert_called_once_with("GET", "https://api.football-data.org/v4/", params=None, json=None, headers={})

    def test_test_connection_failure(self, match_history_client, mock_requests_get):
        """Test failed connection to the API."""
        # Mock the response to raise an exception
        mock_requests_get.side_effect = Exception("Connection error")
        assert match_history_client.test_connection() is False

    def test_get_team_matches(self, match_history_client, mock_requests_get):
        """Test getting team matches."""
        # Mock response data
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "matches": [
                {"id": 1, "homeTeam": {"name": "Team A"}, "awayTeam": {"name": "Team B"}},
                {"id": 2, "homeTeam": {"name": "Team A"}, "awayTeam": {"name": "Team C"}}
            ]
        }
        mock_requests_get.return_value = mock_response

        # Call the method
        matches = match_history_client.get_team_matches(team_id=1, limit=2)
        
        # Assertions
        assert len(matches) == 2
        assert matches[0]["homeTeam"]["name"] == "Team A"
        mock_requests_get.assert_called_once_with(
            "GET", 
            "https://api.football-data.org/v4/teams/1/matches", 
            params={"limit": 2, "status": "FINISHED"}, 
            json=None, 
            headers={}
        )

    def test_get_head_to_head(self, match_history_client, mock_requests_get):
        """Test getting head-to-head statistics."""
        # Mock response for team matches
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "matches": [
                {
                    "id": 1, 
                    "homeTeam": {"id": 1, "name": "Team A"}, 
                    "awayTeam": {"id": 2, "name": "Team B"},
                    "score": {"winner": "HOME_TEAM"}
                },
                {
                    "id": 2, 
                    "homeTeam": {"id": 2, "name": "Team B"}, 
                    "awayTeam": {"id": 1, "name": "Team A"},
                    "score": {"winner": "DRAW"}
                }
            ]
        }
        mock_requests_get.return_value = mock_response

        # Call the method
        h2h = match_history_client.get_head_to_head(team1_id=1, team2_id=2)
        
        # Assertions
        assert h2h["team1_id"] == 1
        assert h2h["team2_id"] == 2
        assert h2h["team1_wins"] == 1
        assert h2h["draws"] == 1
        assert h2h["team2_wins"] == 0
        assert len(h2h["matches"]) == 2

    def test_get_team_info(self, match_history_client, mock_requests_get):
        """Test getting team information."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": 1,
            "name": "Test Team",
            "shortName": "TT",
            "tla": "TST"
        }
        mock_requests_get.return_value = mock_response

        # Call the method
        team_info = match_history_client.get_team_info(team_id=1)
        
        # Assertions
        assert team_info["name"] == "Test Team"
        assert team_info["tla"] == "TST"
        mock_requests_get.assert_called_once_with(
            "GET", 
            "https://api.football-data.org/v4/teams/1", 
            params=None, 
            json=None, 
            headers={}
        )

    def test_search_teams(self, match_history_client, mock_requests_get):
        """Test searching for teams."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "teams": [
                {"id": 1, "name": "Test Team 1"},
                {"id": 2, "name": "Test Team 2"}
            ]
        }
        mock_requests_get.return_value = mock_response

        # Call the method
        results = match_history_client.search_teams("Test")
        
        # Assertions
        assert len(results) == 2
        assert results[0]["name"] == "Test Team 1"
        mock_requests_get.assert_called_once_with(
            "GET", 
            "https://api.football-data.org/v4/teams", 
            params={"name": "Test"}, 
            json=None, 
            headers={}
        )

    def test_get_competition_matches(self, match_history_client, mock_requests_get):
        """Test getting competition matches."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "matches": [
                {"id": 1, "homeTeam": {"name": "Team A"}, "awayTeam": {"name": "Team B"}},
                {"id": 2, "homeTeam": {"name": "Team C"}, "awayTeam": {"name": "Team D"}}
            ]
        }
        mock_requests_get.return_value = mock_response

        # Call the method
        matches = match_history_client.get_competition_matches(
            competition_id=2021, 
            season=2023,
            matchday=5
        )
        
        # Assertions
        assert len(matches) == 2
        assert matches[0]["homeTeam"]["name"] == "Team A"
        mock_requests_get.assert_called_once_with(
            "GET", 
            "https://api.football-data.org/v4/competitions/2021/matches", 
            params={"season": 2023, "matchday": 5}, 
            json=None, 
            headers={}
        )

    def test_get_match_details(self, match_history_client, mock_requests_get):
        """Test getting match details."""
        # Mock response
        match_data = {
            "id": 1,
            "homeTeam": {"name": "Team A"},
            "awayTeam": {"name": "Team B"},
            "score": {"winner": "HOME_TEAM"}
        }
        mock_response = MagicMock()
        mock_response.json.return_value = match_data
        mock_requests_get.return_value = mock_response

        # Call the method
        result = match_history_client.get_match_details(match_id=1)
        
        # Assertions
        assert result == match_data
        mock_requests_get.assert_called_once_with(
            "GET", 
            "https://api.football-data.org/v4/matches/1", 
            params=None, 
            json=None, 
            headers={}
        )

    def test_error_handling(self, match_history_client, mock_requests_get):
        """Test error handling for API requests."""
        # Mock a failed request
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_requests_get.return_value = mock_response

        # Call the method and verify it handles the error
        result = match_history_client.get_team_info(team_id=999)
        assert result is None
