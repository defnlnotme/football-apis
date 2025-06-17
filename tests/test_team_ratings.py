"""Tests for the TeamRatingsClient."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open

class TestTeamRatingsClient:
    """Test cases for TeamRatingsClient."""

    def test_initialization(self, team_ratings_client):
        """Test client initialization with API key."""
        assert team_ratings_client is not None
        assert team_ratings_client.base_url == "http://api.clubelo.com"
        assert team_ratings_client.api_key == "test_api_key"
        assert team_ratings_client.cache_ttl == 86400  # 24 hours

    @patch('builtins.open', new_callable=mock_open, read_data="""Rank,Club,Country,Level,Elo,From,To
1,Manchester City,ENG,1,2073,2023-01-01,2023-12-31
2,Liverpool,ENG,1,2050,2023-01-01,2023-12-31
""")
    def test_get_rankings(self, mock_file, team_ratings_client, mock_requests_get):
        """Test getting team rankings."""
        # Mock the requests.get response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """Rank,Club,Country,Level,Elo,From,To
1,Manchester City,ENG,1,2073,2023-01-01,2023-12-31
2,Liverpool,ENG,1,2050,2023-01-01,2023-12-31
"""
        mock_requests_get.return_value = mock_response

        # Call the method
        rankings = team_ratings_client.get_rankings()
        
        # Assertions
        assert len(rankings) == 2
        assert rankings[0]["rank"] == 1
        assert rankings[0]["name"] == "Manchester City"
        assert rankings[0]["elo"] == 2073
        assert rankings[1]["name"] == "Liverpool"
        
        # Verify the request was made correctly
        mock_requests_get.assert_called_once_with("http://api.clubelo.com/")

    @patch('builtins.open', new_callable=mock_open, read_data="""Rank,Club,Country,Level,Elo,From,To
1,Manchester City,ENG,1,2073,2023-01-01,2023-12-31
2,Liverpool,ENG,1,2050,2023-01-01,2023-12-31
""")
    def test_get_team_elo(self, mock_file, team_ratings_client, mock_requests_get):
        """Test getting team Elo rating."""
        # Mock the requests.get response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """Rank,Club,Country,Level,Elo,From,To
1,Manchester City,ENG,1,2073,2023-01-01,2023-12-31
2,Liverpool,ENG,1,2050,2023-01-01,2023-12-31
"""
        mock_requests_get.return_value = mock_response

        # Test with team name
        team = team_ratings_client.get_team_elo(team_name="Manchester City")
        assert team["name"] == "Manchester City"
        assert team["elo"] == 2073
        
        # Test with team ID (ClubElo ID)
        team = team_ratings_client.get_team_elo(team_id="MCI")
        assert team["name"] == "Manchester City"

    @patch('builtins.open', new_callable=mock_open, read_data="""Rank,Club,Country,Level,Elo,From,To
1,Manchester City,ENG,1,2073,2023-01-01,2023-12-31
2,Liverpool,ENG,1,2050,2023-01-01,2023-12-31
5,Chelsea,ENG,1,1980,2023-01-01,2023-12-31
""")
    def test_search_teams(self, mock_file, team_ratings_client, mock_requests_get):
        """Test searching for teams."""
        # Mock the requests.get response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """Rank,Club,Country,Level,Elo,From,To
1,Manchester City,ENG,1,2073,2023-01-01,2023-12-31
2,Liverpool,ENG,1,2050,2023-01-01,2023-12-31
5,Chelsea,ENG,1,1980,2023-01-01,2023-12-31
"""
        mock_requests_get.return_value = mock_response

        # Search for teams
        results = team_ratings_client.search_teams("che")
        
        # Should find Chelsea
        assert len(results) == 1
        assert results[0]["name"] == "Chelsea"
        assert results[0]["country"] == "ENG"
        assert results[0]["elo"] == 1980

    @patch('builtins.open', new_callable=mock_open, read_data="""Rank,Club,Country,Level,Elo,From,To
1,Manchester City,ENG,1,2073,2023-01-01,2023-12-31
2,Liverpool,ENG,1,2050,2023-01-01,2023-12-31
5,Chelsea,ENG,1,1980,2023-01-01,2023-12-31
20,Real Madrid,ESP,1,1950,2023-01-01,2023-12-31
""")
    def test_get_top_teams(self, mock_file, team_ratings_client, mock_requests_get):
        """Test getting top teams."""
        # Mock the requests.get response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """Rank,Club,Country,Level,Elo,From,To
1,Manchester City,ENG,1,2073,2023-01-01,2023-12-31
2,Liverpool,ENG,1,2050,2023-01-01,2023-12-31
5,Chelsea,ENG,1,1980,2023-01-01,2023-12-31
20,Real Madrid,ESP,1,1950,2023-01-01,2023-12-31
"""
        mock_requests_get.return_value = mock_response

        # Get top 2 teams
        top_teams = team_ratings_client.get_top_teams(limit=2)
        assert len(top_teams) == 2
        assert top_teams[0]["name"] == "Manchester City"
        assert top_teams[1]["name"] == "Liverpool"
        
        # Filter by country
        eng_teams = team_ratings_client.get_top_teams(country="ENG")
        assert len(eng_teams) == 3  # 3 English teams in the test data
        assert all(t["country"] == "ENG" for t in eng_teams)
        
        # Filter by min Elo
        strong_teams = team_ratings_client.get_top_teams(min_elo=2000)
        assert len(strong_teams) == 2  # Only Man City and Liverpool have Elo > 2000

    @patch('builtins.open', new_callable=mock_open, read_data="""Rank,Club,Country,Level,Elo,From,To
1,Manchester City,ENG,1,2073,2023-01-01,2023-03-31
1,Manchester City,ENG,1,2065,2023-04-01,2023-06-30
1,Manchester City,ENG,1,2080,2023-07-01,2023-12-31
""")
    def test_get_historical_elos(self, mock_file, team_ratings_client, mock_requests_get):
        """Test getting historical Elo ratings."""
        # Mock the requests.get response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """Rank,Club,Country,Level,Elo,From,To
1,Manchester City,ENG,1,2073,2023-01-01,2023-03-31
1,Manchester City,ENG,1,2065,2023-04-01,2023-06-30
1,Manchester City,ENG,1,2080,2023-07-01,2023-12-31
"""
        mock_requests_get.return_value = mock_response

        # Get historical Elos
        history = team_ratings_client.get_historical_elos(
            team_id="MCI",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Should return all 3 entries
        assert len(history) == 3
        assert history[0]["elo"] == 2073
        assert history[2]["elo"] == 2080
        
        # Test with date range
        history = team_ratings_client.get_historical_elos(
            team_id="MCI",
            start_date="2023-04-01",
            end_date="2023-06-30"
        )
        assert len(history) == 1
        assert history[0]["elo"] == 2065

    def test_error_handling(self, team_ratings_client, mock_requests_get):
        """Test error handling for API requests."""
        # Mock a failed request
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_requests_get.return_value = mock_response

        # Should handle the error and return empty list
        rankings = team_ratings_client.get_rankings()
        assert rankings == []
