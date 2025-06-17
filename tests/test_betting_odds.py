"""Tests for the BettingOddsClient."""
import pytest
from unittest.mock import MagicMock, patch

class TestBettingOddsClient:
    """Test cases for BettingOddsClient."""

    def test_initialization(self, betting_odds_client):
        """Test client initialization with API key."""
        assert betting_odds_client is not None
        assert betting_odds_client.base_url == "https://api.the-odds-api.com/v4"
        assert betting_odds_client.api_key == "test_api_key"
        assert "apiKey" in betting_odds_client.session.params
        assert betting_odds_client.session.params["apiKey"] == "test_api_key"

    def test_test_connection_success(self, betting_odds_client, mock_requests_get):
        """Test successful connection to the API."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"key": "soccer_epl", "title": "EPL"}]
        mock_requests_get.return_value = mock_response

        assert betting_odds_client.test_connection() is True
        mock_requests_get.assert_called_once_with(
            "GET",
            "https://api.the-odds-api.com/v4/sports",
            params={"apiKey": "test_api_key"},
            json=None,
            headers={}
        )

    def test_get_sports(self, betting_odds_client, mock_requests_get):
        """Test getting available sports."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"key": "soccer_epl", "title": "Premier League"},
            {"key": "soccer_laliga", "title": "La Liga"}
        ]
        mock_requests_get.return_value = mock_response

        sports = betting_odds_client.get_sports()
        assert len(sports) == 2
        assert sports[0]["key"] == "soccer_epl"
        assert sports[0]["title"] == "Premier League"

    def test_get_odds(self, betting_odds_client, mock_requests_get):
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

        odds = betting_odds_client.get_odds(
            sport_key="soccer_epl",
            regions="eu",
            markets="h2h"
        )
        
        assert len(odds) == 1
        assert odds[0]["home_team"] == "Arsenal"
        assert odds[0]["away_team"] == "Chelsea"
        assert len(odds[0]["bookmakers"][0]["markets"][0]["outcomes"]) == 3

    def test_get_scores(self, betting_odds_client, mock_requests_get):
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

        scores = betting_odds_client.get_scores(sport_key="soccer_epl", days_from=1)
        
        assert len(scores) == 1
        assert scores[0]["home_team"] == "Arsenal"
        assert scores[0]["scores"]["Arsenal"] == 2
        assert scores[0]["completed"] is True

    def test_get_historical_odds(self, betting_odds_client, mock_requests_get):
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

        historical = betting_odds_client.get_historical_odds(
            sport_key="soccer_epl",
            event_id="test123"
        )
        
        assert "data" in historical
        assert historical["data"][0]["home_team"] == "Arsenal"

    def test_error_handling(self, betting_odds_client, mock_requests_get):
        """Test error handling for API requests."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_requests_get.return_value = mock_response

        result = betting_odds_client.get_sports()
        assert result == []

    def test_rate_limiting(self, betting_odds_client, mock_requests_get):
        """Test rate limiting handling."""
        # First response is a 429 (rate limited)
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {
            'x-ratelimit-requests-remaining': '0',
            'x-ratelimit-requests-reset': '60'
        }
        
        # Second response is successful
        success_response = MagicMock()
        success_response.json.return_value = [{"key": "soccer_epl", "title": "EPL"}]
        
        # Set up side effects
        mock_requests_get.side_effect = [rate_limit_response, success_response]
        
        # This should handle the rate limit and retry
        sports = betting_odds_client.get_sports()
        
        assert len(sports) == 1
        assert sports[0]["key"] == "soccer_epl"
        assert mock_requests_get.call_count == 2
