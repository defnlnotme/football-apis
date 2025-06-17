"""Tests for the MatchHistoryClient."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from football_apis.clients.football_data_api import FootballDataMatchHistoryClient

class TestFootballDataMatchHistoryClient:
    """Test cases for MatchHistoryClient."""

    @pytest.fixture
    def client(self):
        return FootballDataMatchHistoryClient(api_key="test_key")

    def test_initialization(self, client):
        """Test client initialization with API key."""
        assert client is not None
        assert client.base_url == "https://api.football-data.org/v4"
        assert client.api_key == "test_key"
        assert "X-Auth-Token" in client.session.headers
        assert client.session.headers["X-Auth-Token"] == "test_key"

    def test_test_connection_success(self, client, mock_requests_get):
        """Test successful connection to the API."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {"_links": {"matches": "https://api.football-data.org/v4/matches"}}
        mock_requests_get.return_value = mock_response

        assert client.test_connection() is True
        mock_requests_get.assert_called_once_with("GET", "https://api.football-data.org/v4/", params=None, json=None, headers={})

    def test_test_connection_failure(self, client, mock_requests_get):
        """Test failed connection to the API."""
        # Mock the response to raise an exception
        mock_requests_get.side_effect = Exception("Connection error")
        assert client.test_connection() is False

    def test_get_team_matches(self, client):
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = {"matches": [{"id": 1}]}
            result = client.get_team_matches(team_id=1, limit=5)
            assert result == {"matches": [{"id": 1}]}
            mock_get.assert_called_once_with("/teams/1/matches", params={"limit": 5, "status": "FINISHED"})

    def test_get_head_to_head(self, client):
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = {"matches": [{"id": 1}]}
            result = client.get_head_to_head(team1_id=1, team2_id=2, limit=10)
            assert result == {"matches": [{"id": 1}]}
            mock_get.assert_called_once_with("/teams/1/matches", params={"limit": 10})

    def test_get_team_info(self, client):
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = {"id": 1, "name": "Team A"}
            result = client.get_team_info(team_id=1)
            assert result == {"id": 1, "name": "Team A"}
            mock_get.assert_called_once_with("/teams/1")

    def test_search_teams(self, client):
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = {"teams": [{"id": 1, "name": "Team A"}]}
            result = client.search_teams(name="Team A")
            assert result == {"teams": [{"id": 1, "name": "Team A"}]}
            mock_get.assert_called_once_with("/teams", params={"name": "Team A"})

    def test_get_competition_matches(self, client):
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = {"matches": [{"id": 1}]}
            result = client.get_competition_matches(competition_id=2021, season=2023)
            assert result == {"matches": [{"id": 1}]}
            mock_get.assert_called_once_with("/competitions/2021/matches", params={"season": "2023"})

    def test_get_match_details(self, client):
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = {"id": 1, "homeTeam": {"name": "Team A"}, "awayTeam": {"name": "Team B"}}
            result = client.get_match_details(match_id=1)
            assert result == {"id": 1, "homeTeam": {"name": "Team A"}, "awayTeam": {"name": "Team B"}}
            mock_get.assert_called_once_with("/matches/1")

    def test_error_handling(self, client, mock_requests_get):
        """Test error handling for API requests."""
        # Mock a failed request
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_requests_get.return_value = mock_response

        # Call the method and verify it handles the error
        result = client.get_team_info(team_id=999)
        assert result is None
