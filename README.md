# Football APIs

A Python package for accessing various football (soccer) data APIs in a unified way.

## Features

- **Match History**: Get historical match data and head-to-head statistics
- **Performance Stats**: Access team and player performance metrics
- **Betting Odds**: Retrieve betting odds from various bookmakers
- **Team Ratings**: Get Elo ratings and team strength metrics

## Installation

```bash
pip install -e .
```

## Usage

```python
from football_apis import MatchHistoryClient, PerformanceStatsClient, BettingOddsClient, TeamRatingsClient

# Initialize clients
match_client = MatchHistoryClient(api_key="your_api_key")
performance_client = PerformanceStatsClient(api_key="your_api_key")
betting_client = BettingOddsClient(api_key="your_api_key")
ratings_client = TeamRatingsClient(api_key="your_api_key")

# Example: Get match history
matches = match_client.get_team_matches(team_id=57)  # Arsenal
```

## Available Clients

### MatchHistoryClient
- `get_team_matches(team_id, limit=10)`
- `get_head_to_head(team1_id, team2_id)`
- `get_team_info(team_id)`
- `search_teams(query)`
- `get_competition_matches(competition_id, season, matchday=None)`
- `get_match_details(match_id)`

### PerformanceStatsClient
- `get_team_statistics(team_id, season=2023)`
- `get_team_matches(team_id, season=2023, limit=10)`
- `get_team_standings(team_id, competition_id, season=2023)`
- `get_player_statistics(player_id, season=2023)`

### BettingOddsClient
- `get_sports()`
- `get_odds(sport_key, regions="eu", markets="h2h"`)
- `get_scores(sport_key, days_from=1)`
- `get_historical_odds(sport_key, event_id)`

### TeamRatingsClient
- `get_rankings()`
- `get_team_elo(team_id=None, team_name=None)`
- `search_teams(query)`
- `get_top_teams(limit=10, country=None, min_elo=0)`
- `get_historical_elos(team_id, start_date=None, end_date=None)`

## Configuration

Set your API keys as environment variables or pass them directly to the client constructors.

## License

MIT
