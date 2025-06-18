import os
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import json
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load test data
def load_test_data(filename):
    """Load test data from a JSON file."""
    test_data_dir = Path(__file__).parent / "data"
    filepath = test_data_dir / filename
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Fixtures
@pytest.fixture
def mock_requests_get():
    """Fixture for mocking requests.get"""
    with patch('requests.Session.request') as mock_request:
        yield mock_request

@pytest.fixture
def mock_requests_post():
    """Fixture for mocking requests.post"""
    with patch('requests.Session.post') as mock_post:
        yield mock_post

@pytest.fixture
def mock_requests_session():
    """Fixture for mocking requests.Session"""
    with patch('requests.Session') as mock_session:
        mock_session.return_value = Mock()
        yield mock_session

@pytest.fixture
def mock_response_200():
    """Fixture for a successful 200 response"""
    def _create_mock_response(json_data=None, status_code=200):
        mock_resp = Mock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data or {}
        mock_resp.text = json.dumps(json_data) if json_data else ''
        return mock_resp
    return _create_mock_response

@pytest.fixture
def mock_response_404():
    """Fixture for a 404 response"""
    mock_resp = Mock()
    mock_resp.status_code = 404
    mock_resp.json.return_value = {"message": "Not found"}
    mock_resp.text = '{"message": "Not found"}'
    return mock_resp

@pytest.fixture
def mock_response_429():
    """Fixture for a rate limit response"""
    mock_resp = Mock()
    mock_resp.status_code = 429
    mock_resp.json.return_value = {"message": "Too many requests"}
    mock_resp.headers = {
        'X-RateLimit-Remaining': '0',
        'X-RateLimit-Reset': str(int((datetime.now() + timedelta(seconds=60)).timestamp()))
    }
    return mock_resp

@pytest.fixture
def mock_response_500():
    """Fixture for a server error response"""
    mock_resp = Mock()
    mock_resp.status_code = 500
    mock_resp.text = 'Internal Server Error'
    return mock_resp

# Client fixtures
@pytest.fixture
def the_odds_api_client():
    """Fixture for TheOddsApiClient with test API key"""
    from football_apis.clients.the_odds_api import TheOddsApiClient
    return TheOddsApiClient(api_key="test_api_key")

@pytest.fixture
def clubelo_client():
    """Fixture for ClubEloClient"""
    from football_apis.clients.clubelo_api import ClubEloClient
    return ClubEloClient()

@pytest.fixture
def football_data_client():
    """Fixture for FootballDataClient with test API key"""
    from football_apis.clients.football_data_api import FootballDataClient
    return FootballDataClient(api_key="test_api_key")

# Test data fixtures
@pytest.fixture
def sample_match_data():
    """Sample match data for testing"""
    return {
        "matches": [
            {
                "id": 1,
                "competition": {"name": "Premier League"},
                "homeTeam": {"id": 57, "name": "Arsenal"},
                "awayTeam": {"id": 61, "name": "Chelsea"},
                "score": {"winner": "HOME_TEAM"},
                "status": "FINISHED"
            }
        ]
    }

@pytest.fixture
def sample_team_data():
    """Sample team data for testing"""
    return {
        "id": 57,
        "name": "Arsenal",
        "shortName": "Arsenal",
        "tla": "ARS",
        "crest": "https://crests.football-data.org/57.png",
        "founded": 1886,
        "venue": "Emirates Stadium"
    }

@pytest.fixture
def sample_odds_data():
    """Sample odds data for testing"""
    return {
        "home_win": 2.1,
        "draw": 3.4,
        "away_win": 3.6
    }
