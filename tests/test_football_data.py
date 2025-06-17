"""Tests for the FootballDataClient."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from football_apis.clients.football_data_api import FootballDataClient

class TestFootballDataClient:
    @pytest.fixture
    def client(self):
        return FootballDataClient(api_key="test_key")
    
    def test_get_team_statistics(self, client):
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = {"statistics": {"played": 10}}
            result = client.get_team_statistics(team_id=1, season=2023)
            assert result == {"statistics": {"played": 10}}
            mock_get.assert_called_once_with("/teams/1/statistics", params={"season": "2023"})
    
    def test_get_team_matches(self, client):
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = {"matches": [{"id": 1}]}
            result = client.get_team_matches(team_id=1, season=2023)
            assert result == {"matches": [{"id": 1}]}
            mock_get.assert_called_once_with("/teams/1/matches", params={"season": "2023", "status": "FINISHED"})
    
    def test_get_team_standings(self, client):
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = {"standings": [{"position": 1}]}
            result = client.get_team_standings(team_id=1, competition_id=2021, season=2023)
            assert result == {"standings": [{"position": 1}]}
            mock_get.assert_called_once_with("/competitions/2021/standings", params={"season": "2023"})
    
    def test_get_player_statistics(self, client):
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = {"statistics": {"goals": 5}}
            result = client.get_player_statistics(player_id=1, season=2023)
            assert result == {"statistics": {"goals": 5}}
            mock_get.assert_called_once_with("/players/1/statistics", params={"season": "2023"})
    
    def test_test_connection(self, client):
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = {"competitions": []}
            assert client.test_connection() is True
            mock_get.assert_called_once_with("/competitions")
            
            mock_get.side_effect = Exception("API Error")
            assert client.test_connection() is False

    def test_get_head_to_head(self, client):
        with patch.object(client, 'get') as mock_get:
            mock_get.return_value = {"matches": [{"id": 1}]}
            result = client.get_head_to_head(team1_id=1, team2_id=2, limit=10)
            assert result == {"matches": [{"id": 1}]}
            mock_get.assert_called_once_with("/teams/1/matches", params={'limit': 10})

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
        with pytest.raises(Exception, match="API Error"):
            client.get_team_info(team_id=999)
