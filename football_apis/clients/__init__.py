"""
Football API Clients

This module contains the client implementations for various football data APIs.
"""

from .match_history import MatchHistoryClient
from .performance_stats import PerformanceStatsClient
from .betting_odds import BettingOddsClient
from .team_ratings import TeamRatingsClient

__all__ = [
    'MatchHistoryClient',
    'PerformanceStatsClient',
    'BettingOddsClient',
    'TeamRatingsClient',
]
