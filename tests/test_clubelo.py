"""Tests for the ClubEloClient."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

class TestClubEloClient:
    """Test cases for ClubEloClient."""

    def test_initialization(self, clubelo_client):
        """Test client initialization without API key."""
        assert clubelo_client is not None
        assert clubelo_client.base_url == "http://api.clubelo.com"
        assert clubelo_client.cache_ttl == 86400  # 24 hours

    @patch('football_apis.clients.clubelo_api.ClubEloClient.get_cached')
    def test_get_team_elo(self, mock_get_cached, clubelo_client):
        """Test getting team Elo rating."""
        mock_get_cached.return_value = [
            {'rank': '1', 'club': 'Manchester City', 'country': 'ENG', 'level': '1', 'elo': '2000', 'from': '2023-01-01', 'to': '2023-03-31'},
            {'rank': '1', 'club': 'Manchester City', 'country': 'ENG', 'level': '1', 'elo': '2050', 'from': '2023-04-01', 'to': '2023-06-30'},
            {'rank': '1', 'club': 'Manchester City', 'country': 'ENG', 'level': '1', 'elo': '2073', 'from': '2023-07-01', 'to': '2023-12-31'}
        ]

        ratings = clubelo_client.get_team_elo(team_name="Manchester City", date="2023-05-15")
        assert len(ratings) == 2  # Should get ratings up to May 15
        assert ratings[0]["team_name"] == "Manchester City"
        assert ratings[0]["elo"] == 2000
        assert ratings[0]["is_current"] is False
        assert ratings[1]["elo"] == 2050
        assert ratings[1]["is_current"] is True

        ratings = clubelo_client.get_team_elo(team_name="Manchester City")
        assert len(ratings) == 3  # Should get all ratings
        assert ratings[0]["team_name"] == "Manchester City"
        assert ratings[0]["elo"] == 2000
        assert ratings[0]["is_current"] is False
        assert ratings[2]["elo"] == 2073
        assert ratings[2]["is_current"] is True

        mock_get_cached.return_value = []
        ratings = clubelo_client.get_team_elo(team_name="NonExistent Team")
        assert ratings is None

    @patch('football_apis.clients.clubelo_api.ClubEloClient.get_cached')
    def test_get_top_teams(self, mock_get_cached, clubelo_client):
        """Test getting top teams with optional date and filters."""
        mock_get_cached.return_value = [
            {'rank': '1', 'club': 'Manchester City', 'country': 'ENG', 'level': '1', 'elo': '2073', 'from': '2023-01-01', 'to': '2023-12-31', 'id': '1'},
            {'rank': '2', 'club': 'Liverpool', 'country': 'ENG', 'level': '1', 'elo': '2050', 'from': '2023-01-01', 'to': '2023-12-31', 'id': '2'},
            {'rank': '5', 'club': 'Chelsea', 'country': 'ENG', 'level': '1', 'elo': '1980', 'from': '2023-01-01', 'to': '2023-12-31', 'id': '5'},
            {'rank': '20', 'club': 'Real Madrid', 'country': 'ESP', 'level': '1', 'elo': '1950', 'from': '2023-01-01', 'to': '2023-12-31', 'id': '20'}
        ]

        top_teams = clubelo_client.get_top_teams()
        assert len(top_teams) == 4
        assert top_teams[0]["team_name"] == "Manchester City"
        assert top_teams[0]["is_current"] is True

        top_teams_limited = clubelo_client.get_top_teams(limit=2)
        assert len(top_teams_limited) == 2
        assert top_teams_limited[0]["team_name"] == "Manchester City"
        assert top_teams_limited[1]["team_name"] == "Liverpool"

        eng_teams = clubelo_client.get_top_teams(country="eng")
        assert len(eng_teams) == 3
        assert all(t["country"] == "ENG" for t in eng_teams)

        strong_teams = clubelo_client.get_top_teams(min_elo=2000)
        assert len(strong_teams) == 2
        assert strong_teams[0]["team_name"] == "Manchester City"
        assert strong_teams[1]["team_name"] == "Liverpool"

        top_teams_date = clubelo_client.get_top_teams(date="2023-01-15")
        assert len(top_teams_date) == 4

    @patch('football_apis.clients.clubelo_api.ClubEloClient.get_cached')
    def test_get_fixtures(self, mock_get_cached, clubelo_client):
        """Test getting upcoming fixtures."""
        mock_get_cached.return_value = [
            {"date": "2024-03-10", "team1": "Home Team", "team2": "Away Team", "prob1": "0.5", "probX": "0.3", "prob2": "0.2"},
            {"date": "2024-03-11", "team1": "Another Home", "team2": "Another Away", "prob1": "0.6", "probX": "0.2", "prob2": "0.2"}
        ]

        fixtures = clubelo_client.get_fixtures()
        assert len(fixtures) == 2
        assert fixtures[0]["team1"] == "Home Team"
        assert fixtures[1]["prob1"] == "0.6"

        mock_get_cached.return_value = []
        fixtures_empty = clubelo_client.get_fixtures()
        assert fixtures_empty == []

    @patch('football_apis.clients.clubelo_api.ClubEloClient.get_cached')
    def test_error_handling(self, mock_get_cached, clubelo_client):
        """Test error handling for API requests."""
        mock_get_cached.side_effect = Exception("API Error")

        team_elo = clubelo_client.get_team_elo(team_name="Any Team")
        assert team_elo is None

        top_teams = clubelo_client.get_top_teams()
        assert top_teams == []

        fixtures = clubelo_client.get_fixtures()
        assert fixtures == []
