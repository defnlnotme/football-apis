"""Tests for the PerformanceStatsClient."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from football_apis.clients.football_data_api import FootballDataPerformanceStatsClient

class TestFootballDataPerformanceStatsClient:
    @pytest.fixture
    def client(self):
        return FootballDataPerformanceStatsClient(api_key="test_key")
    
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
