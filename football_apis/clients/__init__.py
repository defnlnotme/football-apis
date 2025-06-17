"""
Football API Clients

This module contains the client implementations for various football data APIs.
"""

from .the_odds_api import TheOddsApiClient
from .clubelo_api import ClubEloClient

__all__ = [
    'TheOddsApiClient',
    'ClubEloClient'
]
