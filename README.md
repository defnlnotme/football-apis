# Football APIs

A Python package for accessing various football (soccer) data APIs in a unified way.

## Features

- **Match History & Stats**: Get historical match data, head-to-head statistics, team/player performance, standings, and more via [Football-Data.org](https://www.football-data.org/)
- **Betting Odds**: Retrieve live and historical betting odds from [The Odds API](https://the-odds-api.com/)
- **Team Ratings**: Get Elo ratings and team strength metrics from [ClubElo](http://clubelo.com/)
- **Unified Interface**: Consistent API, caching, and rate limiting across all clients

## Installation

```bash
pip install -e .
```

## Client Overview

The package exposes three main clients:

- `FootballDataClient`: For match history, team/player stats, competitions, and standings (Football-Data.org)
- `TheOddsApiClient`: For betting odds, sports, and scores (The Odds API)
- `ClubEloClient`: For Elo ratings and team rankings (ClubElo, no API key required)

Import them from the main package:

```python
from football_apis import FootballDataClient, TheOddsApiClient, ClubEloClient
```

## API Keys & Configuration

- **FootballDataClient**: Requires an API key from [Football-Data.org](https://www.football-data.org/client/register)
- **TheOddsApiClient**: Requires an API key from [The Odds API](https://the-odds-api.com/)
- **ClubEloClient**: No API key required

You can provide API keys in two ways:

1. **Directly in the constructor:**
   ```python
   client = FootballDataClient(api_key="YOUR_API_KEY")
   ```
2. **Environment variable:**
   - For FootballDataClient: `FOOTBALLDATACLIENT_API_KEY`
   - For TheOddsApiClient: `THEODDSAPICLIENT_API_KEY`

   Example:
   ```bash
   export FOOTBALLDATACLIENT_API_KEY=your_key_here
   export THEODDSAPICLIENT_API_KEY=your_key_here
   ```

## Usage Examples

### 1. FootballDataClient (Football-Data.org)

```python
from football_apis import FootballDataClient

client = FootballDataClient(api_key="YOUR_API_KEY")

# Get team details
team = client.get_team_info(team_id=57)  # Arsenal FC

# Get recent matches for a team
matches = client.get_team_matches(team_id=57, limit=5)

# Get head-to-head matches
h2h = client.get_head_to_head(team1_id=57, team2_id=61)

# Get competition standings
standings = client.get_team_standings(team_id=57, competition_id=2021)

# Search for teams
teams = client.search_teams(name="Arsenal")
```

### 2. TheOddsApiClient (The Odds API)

```python
from football_apis import TheOddsApiClient

client = TheOddsApiClient(api_key="YOUR_API_KEY")

# List available sports
sports = client.get_sports()

# Get odds for English Premier League
odds = client.get_odds(sport_key="soccer_epl", regions="eu", markets="h2h")

# Get recent scores
scores = client.get_scores(sport_key="soccer_epl", days_from=2)
```

### 3. ClubEloClient (ClubElo)

```python
from football_apis import ClubEloClient

client = ClubEloClient()

# Get Elo ratings for a team
elo_history = client.get_team_elo(team_name="Barcelona")

# Get top teams by Elo rating
top_teams = client.get_top_teams(limit=10, country="ESP")
```

## Advanced Configuration

All clients support caching and rate limiting:

```python
client = FootballDataClient(api_key="...", cache_enabled=True, cache_ttl=3600)  # cache_ttl in seconds
```

- `cache_enabled`: Enable/disable response caching (default: True)
- `cache_ttl`: Cache time-to-live in seconds (default varies by client)

## Error Handling & Troubleshooting

- All methods return Python dicts/lists. Check for error keys in responses.
- For API errors (e.g., 403, 429), check your API key and rate limits.
- Use `client.test_connection()` to verify connectivity.
- Enable logging for more details:
  ```python
  import logging
  logging.basicConfig(level=logging.INFO)
  ```

## Contributing

Pull requests and issues are welcome! Please add tests for new features.

## License

MIT
