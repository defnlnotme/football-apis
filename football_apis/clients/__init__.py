"""
Football API Clients

This module contains the client implementations for various football data APIs.
"""

from .the_odds_api import TheOddsApiClient
from .clubelo_api import ClubEloClient
from .football_data_api import FootballDataPerformanceStatsClient, FootballDataMatchHistoryClient

__all__ = [
    'TheOddsApiClient',
    'ClubEloClient',
    'FootballDataPerformanceStatsClient',
    'FootballDataMatchHistoryClient'
]
