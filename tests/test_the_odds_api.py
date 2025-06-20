"""Tests for the TheOddsApiClient."""
import pytest
from unittest.mock import MagicMock, patch, ANY
import requests

class TestTheOddsApiClient:
    """Test cases for TheOddsApiClient."""

    def test_initialization(self, the_odds_api_client):
        """Test client initialization with API key."""
        assert the_odds_api_client is not None
        assert the_odds_api_client.base_url == "https://api.the-odds-api.com/v4"
        assert the_odds_api_client.api_key == "test_api_key"
        assert the_odds_api_client.session.params == {}

    def test_test_connection_success(self, the_odds_api_client, mock_requests_get):
        """Test successful connection to the API."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"key": "soccer_epl", "title": "EPL"}]
        mock_requests_get.return_value = mock_response

        assert the_odds_api_client.test_connection() is True
        mock_requests_get.assert_called_once_with(
            method="GET",
            url="https://api.the-odds-api.com/v4/sports",
            params={"apiKey": "test_api_key"},
            json=None,
            headers=ANY
        )

    def test_get_sports(self, the_odds_api_client, mock_requests_get):
        """Test getting available sports."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"key": "soccer_epl", "title": "Premier League"},
            {"key": "soccer_laliga", "title": "La Liga"}
        ]
        mock_requests_get.return_value = mock_response

        sports = the_odds_api_client.get_sports()
        assert len(sports) == 2
        assert sports[0]["key"] == "soccer_epl"
        assert sports[0]["title"] == "Premier League"

    def test_get_odds(self, the_odds_api_client, mock_requests_get):
        """Test getting betting odds."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": "test123",
                "sport_key": "soccer_epl",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "bookmakers": [
                    {
                        "key": "betfair",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Arsenal", "price": 2.1},
                                    {"name": "Draw", "price": 3.4},
                                    {"name": "Chelsea", "price": 3.6}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
        mock_requests_get.return_value = mock_response

        odds = the_odds_api_client.get_odds(
            sport_key="soccer_epl",
            regions="eu",
            markets="h2h"
        )
        
        assert len(odds) == 1
        assert odds[0]["home_team"] == "Arsenal"
        assert odds[0]["away_team"] == "Chelsea"
        assert len(odds[0]["bookmakers"][0]["markets"][0]["outcomes"]) == 3

    def test_get_scores(self, the_odds_api_client, mock_requests_get):
        """Test getting scores."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": "test123",
                "sport_key": "soccer_epl",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "scores": {"Arsenal": 2, "Chelsea": 1},
                "completed": True
            }
        ]
        mock_requests_get.return_value = mock_response

        scores = the_odds_api_client.get_scores(sport_key="soccer_epl", days_from=1)
        
        assert len(scores) == 1
        assert scores[0]["home_team"] == "Arsenal"
        assert scores[0]["scores"]["Arsenal"] == 2
        assert scores[0]["completed"] is True

    def test_get_historical_odds(self, the_odds_api_client, mock_requests_get):
        """Test getting historical odds."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "test123",
                    "sport_key": "soccer_epl",
                    "commence_time": "2023-01-01T15:00:00Z",
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "bookmakers": []
                }
            ]
        }
        mock_requests_get.return_value = mock_response

        historical = the_odds_api_client.get_historical_odds(
            sport_key="soccer_epl",
            date="2023-01-01T15:00:00Z"
        )
        
        assert "data" in historical
        assert historical["data"][0]["home_team"] == "Arsenal"

    def test_error_handling(self, the_odds_api_client, mock_requests_get):
        """Test error handling for API requests."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_requests_get.return_value = mock_response

        with pytest.raises(Exception, match="API Error"):
            the_odds_api_client.get_sports()

    def test_rate_limiting(self, the_odds_api_client, mock_requests_get):
        """Test rate limiting handling."""
        # First response is a 429 (rate limited)
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {
            'x-ratelimit-requests-remaining': '0',
            'x-ratelimit-requests-reset': '60'
        }
        rate_limit_response.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Rate Limited")
        rate_limit_response.request = MagicMock()
        rate_limit_response.request.method = "GET"
        rate_limit_response.request.path_url = "/sports"
        rate_limit_response.request.params = {"apiKey": "test_api_key"}
        rate_limit_response.request.json = None
        rate_limit_response.request.headers = {}

        # Second response is successful
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = [{"key": "soccer_epl", "title": "EPL"}]
        success_response.raise_for_status.return_value = None

        # Set up side effects
        mock_requests_get.side_effect = [rate_limit_response, success_response]

        # This should handle the rate limit and retry
        sports = the_odds_api_client.get_sports()

        assert len(sports) == 1
        assert sports[0]["key"] == "soccer_epl"
        assert mock_requests_get.call_count == 2
