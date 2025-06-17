"""
Football APIs - A Python package for accessing various football (soccer) data APIs.

This package provides a unified interface to multiple football data sources,
including match history, team statistics, betting odds, and team ratings.

Available clients:
- MatchHistoryClient: Access match history and head-to-head statistics
- PerformanceStatsClient: Get team and player performance metrics
- BettingOddsClient: Retrieve betting odds from various bookmakers
- TeamRatingsClient: Access team Elo ratings and strength metrics
"""

from .clients.match_history import MatchHistoryClient
from .clients.performance_stats import PerformanceStatsClient
from .clients.betting_odds import BettingOddsClient
from .clients.team_ratings import TeamRatingsClient

__version__ = "0.1.0"

__all__ = [
    "MatchHistoryClient",
    "PerformanceStatsClient",
    "BettingOddsClient",
    "TeamRatingsClient",
]
