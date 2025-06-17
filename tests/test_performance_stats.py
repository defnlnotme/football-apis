"""Tests for the PerformanceStatsClient."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

class TestPerformanceStatsClient:
    """Test cases for PerformanceStatsClient."""

    def test_initialization(self, performance_stats_client):
        """Test client initialization with API key."""
        assert performance_stats_client is not None
        assert performance_stats_client.base_url == "https://api.football-data.org/v4"
        assert performance_stats_client.api_key == "test_api_key"
        assert "X-Auth-Token" in performance_stats_client.session.headers

    def test_test_connection_success(self, performance_stats_client, mock_requests_get):
        """Test successful connection to the API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"_links": {"teams": "https://api.football-data.org/v4/teams"}}
        mock_requests_get.return_value = mock_response

        assert performance_stats_client.test_connection() is True
        mock_requests_get.assert_called_once()

    def test_get_team_statistics(self, performance_stats_client, mock_requests_get):
        """Test getting team statistics."""
        # Mock response for team data
        team_data = {
            "id": 57,
            "name": "Arsenal",
            "squad": [],
            "matches": [
                {
                    "homeTeam": {"id": 57, "name": "Arsenal"},
                    "awayTeam": {"id": 61, "name": "Chelsea"},
                    "homeTeamStats": {"goals": 2, "shots": {"total": 10, "on": 5}, "expectedGoals": 1.8},
                    "awayTeamStats": {"goals": 1, "shots": {"total": 8, "on": 3}, "expectedGoals": 0.9},
                    "score": {"winner": "HOME_TEAM", "fullTime": {"homeTeam": 2, "awayTeam": 1}}
                }
            ]
        }
        
        mock_response = MagicMock()
        mock_response.json.return_value = team_data
        mock_requests_get.return_value = mock_response

        # Call the method
        stats = performance_stats_client.get_team_statistics(team_id=57, season=2023)
        
        # Assertions
        assert stats["team_id"] == 57
        assert stats["team_name"] == "Arsenal"
        assert stats["stats"]["matches_played"] == 1
        assert stats["stats"]["goals_for"] == 2
        assert stats["stats"]["goals_against"] == 1
        assert stats["stats"]["clean_sheets"] == 0
        assert stats["stats"]["xg_per_match"] == 1.8

    def test_get_team_matches(self, performance_stats_client, mock_requests_get):
        """Test getting team matches."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "matches": [
                {"id": 1, "homeTeam": {"name": "Team A"}, "awayTeam": {"name": "Team B"}},
                {"id": 2, "homeTeam": {"name": "Team C"}, "awayTeam": {"name": "Team A"}}
            ]
        }
        mock_requests_get.return_value = mock_response

        matches = performance_stats_client.get_team_matches(team_id=1, season=2023)
        assert len(matches) == 2
        assert matches[0]["homeTeam"]["name"] == "Team A"

    def test_get_team_standings(self, performance_stats_client, mock_requests_get):
        """Test getting team standings."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "standings": [
                {
                    "type": "TOTAL",
                    "table": [
                        {
                            "position": 1,
                            "team": {"id": 57, "name": "Arsenal"},
                            "playedGames": 38,
                            "won": 26,
                            "draw": 6,
                            "lost": 6,
                            "points": 84,
                            "goalsFor": 88,
                            "goalsAgainst": 43,
                            "goalDifference": 45,
                            "form": "WWLWD"
                        }
                    ]
                }
            ],
            "competition": {"name": "Premier League"},
            "season": {"currentMatchday": 38}
        }
        mock_requests_get.return_value = mock_response

        standings = performance_stats_client.get_team_standings(
            team_id=57, 
            competition_id=2021, 
            season=2023
        )
        
        assert standings["position"] == 1
        assert standings["team_name"] == "Arsenal"
        assert standings["points"] == 84
        assert standings["competition"] == "Premier League"

    def test_get_player_statistics(self, performance_stats_client):
        """Test getting player statistics (placeholder implementation)."""
        # This is a placeholder test since the actual implementation is a stub
        stats = performance_stats_client.get_player_statistics(player_id=1, season=2023)
        assert stats["player_id"] == 1
        assert stats["season"] == 2023
        assert stats["stats"] == {}

    def test_error_handling(self, performance_stats_client, mock_requests_get):
        """Test error handling for API requests."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_requests_get.return_value = mock_response

        result = performance_stats_client.get_team_statistics(team_id=999, season=2023)
        assert result == {}
