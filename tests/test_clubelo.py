"""Tests for the ClubEloClient."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

class TestClubEloClient:
    """Test cases for ClubEloClient."""

    def test_initialization(self, clubelo_client):
        """Test client initialization with API key."""
        assert clubelo_client is not None
        assert clubelo_client.base_url == "http://api.clubelo.com"
        assert clubelo_client.api_key == "test_api_key"
        assert clubelo_client.cache_ttl == 86400  # 24 hours

    @patch('football_apis.clients.clubelo_api.ClubEloClient.get_cached')
    @patch('football_apis.clients.clubelo_api.ClubEloClient.search_teams')
    def test_get_team_elo(self, mock_search_teams, mock_get_cached, clubelo_client):
        """Test getting team Elo rating."""
        # Mock search_teams to return a list of dictionaries with expected keys
        mock_search_teams.return_value = [{'id': '1', 'club': 'Manchester City', 'country': 'ENG', 'level': '1', 'elo': '2073', 'from': '2023-01-01', 'to': '2023-12-31', 'rank': '1'}]
        
        # Mock get_cached to return a list of dictionaries with expected keys
        mock_get_cached.return_value = [
            {'rank': '1', 'club': 'Manchester City', 'country': 'ENG', 'level': '1', 'elo': '2073', 'from': '2023-01-01', 'to': '2023-12-31'},
            {'rank': '2', 'club': 'Liverpool', 'country': 'ENG', 'level': '1', 'elo': '2050', 'from': '2023-01-01', 'to': '2023-12-31'}
        ]

        # Test with team name and date
        team = clubelo_client.get_team_elo(team_name="Manchester City", date="2023-12-31")
        assert team["team_name"] == "Manchester City"
        assert team["elo"] == 2073
        assert team["is_current"] is False # Because a specific date was provided
        
        # Test with team ID and date
        team = clubelo_client.get_team_elo(team_id=1, date="2023-12-31") 
        assert team["team_name"] == "Manchester City"
        assert team["elo"] == 2073
        assert team["is_current"] is False

        # Test with no date (should return current Elo)
        mock_get_cached.return_value = [
            {'rank': '1', 'club': 'Manchester City', 'country': 'ENG', 'level': '1', 'elo': '2073', 'from': '2023-01-01', 'to': '2023-12-31'},
        ]
        team = clubelo_client.get_team_elo(team_id=1)
        assert team["team_name"] == "Manchester City"
        assert team["elo"] == 2073
        assert team["is_current"] is True

    @patch('football_apis.clients.clubelo_api.ClubEloClient.get_cached')
    def test_search_teams(self, mock_get_cached, clubelo_client):
        """Test searching for teams."""
        mock_get_cached.return_value = [
            {'rank': '1', 'club': 'Manchester City', 'country': 'ENG', 'level': '1', 'elo': '2073', 'from': '2023-01-01', 'to': '2023-12-31', 'id': '1'},
            {'rank': '2', 'club': 'Liverpool', 'country': 'ENG', 'level': '1', 'elo': '2050', 'from': '2023-01-01', 'to': '2023-12-31', 'id': '2'},
            {'rank': '5', 'club': 'Chelsea', 'country': 'ENG', 'level': '1', 'elo': '1980', 'from': '2023-01-01', 'to': '2023-12-31', 'id': '5'}
        ]

        results = clubelo_client.search_teams("che")
        assert len(results) == 1
        assert results[0]["name"] == "Chelsea"
        assert results[0]["country"] == "ENG"
        assert results[0]["elo"] == 1980

    @patch('football_apis.clients.clubelo_api.ClubEloClient.get_cached')
    def test_get_top_teams(self, mock_get_cached, clubelo_client):
        """Test getting top teams."""
        mock_get_cached.return_value = [
            {'rank': '1', 'club': 'Manchester City', 'country': 'ENG', 'level': '1', 'elo': '2073', 'from': '2023-01-01', 'to': '2023-12-31', 'id': '1'},
            {'rank': '2', 'club': 'Liverpool', 'country': 'ENG', 'level': '1', 'elo': '2050', 'from': '2023-01-01', 'to': '2023-12-31', 'id': '2'},
            {'rank': '5', 'club': 'Chelsea', 'country': 'ENG', 'level': '1', 'elo': '1980', 'from': '2023-01-01', 'to': '2023-12-31', 'id': '5'},
            {'rank': '20', 'club': 'Real Madrid', 'country': 'ESP', 'level': '1', 'elo': '1950', 'from': '2023-01-01', 'to': '2023-12-31', 'id': '20'}
        ]

        top_teams = clubelo_client.get_top_teams(limit=2)
        assert len(top_teams) == 2
        assert top_teams[0]["team_name"] == "Manchester City"
        assert top_teams[1]["team_name"] == "Liverpool"
        
        eng_teams = clubelo_client.get_top_teams(country="eng")
        assert len(eng_teams) == 3
        assert all(t["country"] == "ENG" for t in eng_teams)
        
        strong_teams = clubelo_client.get_top_teams(min_elo=2000)
        assert len(strong_teams) == 2

    @patch('football_apis.clients.clubelo_api.ClubEloClient.get_cached')
    def test_get_historical_elos(self, mock_get_cached, clubelo_client):
        """Test getting historical Elo ratings."""
        mock_get_cached.return_value = [
            {'rank': '1', 'club': 'Manchester City', 'country': 'ENG', 'level': '1', 'elo': '2073', 'from': '2023-01-01', 'to': '2023-03-31', 'id': '1'},
            {'rank': '1', 'club': 'Manchester City', 'country': 'ENG', 'level': '1', 'elo': '2065', 'from': '2023-04-01', 'to': '2023-06-30', 'id': '1'},
            {'rank': '1', 'club': 'Manchester City', 'country': 'ENG', 'level': '1', 'elo': '2080', 'from': '2023-07-01', 'to': '2023-12-31', 'id': '1'}
        ]

        history = clubelo_client.get_historical_elos(
            team_id=1, 
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        assert len(history) == 3
        assert history[0]["elo"] == 2073
        assert history[2]["elo"] == 2080
        
        history = clubelo_client.get_historical_elos(
            team_id=1, 
            start_date="2023-04-01",
            end_date="2023-06-30"
        )
        assert len(history) == 1
        assert history[0]["elo"] == 2065

    @patch('football_apis.clients.clubelo_api.ClubEloClient.get_cached')
    def test_error_handling(self, mock_get_cached, clubelo_client):
        """Test error handling for API requests."""
        mock_get_cached.side_effect = Exception("API Error")

        team_elo = clubelo_client.get_team_elo(team_id=999, date="2023-01-01")
        assert team_elo is None
