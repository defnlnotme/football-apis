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

## Command Line Interface

The package includes a comprehensive CLI tool (`client.py`) for fetching data from all endpoints:

```bash
# List available parameters for betting odds
python client.py the-odds-api --list-params

# Fetch data from all endpoints
python client.py all

# Fetch only betting odds and team ratings
python client.py the-odds-api clubelo

# Get help for a specific endpoint
python client.py the-odds-api --info

# Get betting odds with specific parameters
python client.py the-odds-api --sport soccer_epl --region eu --odds-format decimal

# Get team ratings for a specific team
python client.py clubelo --team Barcelona --date 2024-01-01

# Get performance stats for a team in a specific season
python client.py football-data --team-id 81 --season 2023 --competition-id 2021
```

For more examples and detailed usage, run:
```bash
python client.py --help
```

---

## Web Scraping with LLM Data Extraction

In addition to the API clients, this package includes a powerful web scraping tool (`scrape.py`) that can extract HTML content from football websites and use CAMEL agents to intelligently identify and extract structured data.

### Scraping Features

- **Multi-site scraping**: Scrape from 8 popular football websites
- **LLM-powered data extraction**: Use CAMEL agents to extract structured data from scraped content
- **Flexible site selection**: Choose individual sites or scrape all at once
- **Robust error handling**: Retry logic and rate limit handling
- **Structured output**: Save both HTML and JSON extracted data
- **Beautiful console output**: Colored and formatted results
- **Extensible architecture**: Easy to add new sites and extraction types

### Available Sites

- **worldfootball**: World Football - Football statistics and information
- **transfermarkt**: Transfermarkt - Football transfer market and player statistics
- **espn**: ESPN Soccer - Latest football news and scores
- **bbc_sport**: BBC Sport Football - UK football news and live scores
- **goal**: Goal.com - International football news and live scores
- **skysports**: Sky Sports Football - Premier League and football coverage
- **fifa**: FIFA - Official website of international football
- **uefa**: UEFA - European football governing body

### Scraping Usage

#### Basic Scraping

List all available sites:
```bash
python scrape.py --list
```

Scrape a single site:
```bash
python scrape.py --site worldfootball
```

Scrape all sites:
```bash
python scrape.py --site all
```

#### Data Extraction

Scrape and extract data from a single site:
```bash
python scrape.py --site worldfootball --extract-competitions
```

Scrape and extract data from all sites:
```bash
python scrape.py --site all --extract-competitions
```

### Scraping Output Files

#### HTML Files
- Format: `{site_name}_{timestamp}.html`
- Contains the full HTML content of the scraped page

#### Extracted Data Files
- Format: `{site_name}_competitions_{timestamp}.json`
- Contains structured data extracted by the LLM

#### Example Data Structure

```json
{
  "competitions": [
    {
      "name": "Premier League",
      "type": "league",
      "country": "England",
      "season": "2023/24",
      "url": "https://example.com/premier-league",
      "description": "Top tier English football league"
    }
  ],
  "summary": {
    "total_competitions": 15,
    "categories": {
      "leagues": 8,
      "tournaments": 3,
      "cups": 2,
      "international": 1,
      "regional": 1,
      "youth": 0,
      "womens": 0
    }
  }
}
```

### Data Extraction Types

The current implementation includes:
- **Competition extraction**: Identifies and categorizes football competitions, tournaments, and leagues
- **Categorization**: Automatically categorizes data into relevant types (leagues, tournaments, cups, etc.)

The system is designed to be easily extensible for other types of data extraction such as:
- Player information and statistics
- Team details and rankings
- Match schedules and results
- News and articles
- Transfer information
- Historical data

### Scraping Requirements

- Python 3.8+
- CAMEL-AI framework
- Gemini API key (set in `.envrc`):
  ```
  GEMINI_API_KEY=your_gemini_api_key_here
  ```

### Scraping Example Output

```
Scraping: worldfootball
Description: World Football - Football statistics and information
URL: https://www.worldfootball.net/
Data extraction: Enabled

✓ HTML extracted and saved to: worldfootball_20241201_143022.html
✓ Content length: 45678 characters

Extracting data from worldfootball...
✓ Data extracted and saved to: worldfootball_competitions_20241201_143022.json
✓ Found 25 competitions

Data categories:
  - leagues: 12
  - tournaments: 8
  - cups: 3
  - international: 2

=== Data Found on worldfootball ===
Total: 25

LEAGUES (12):
  • Premier League
    Country: England
    Season: 2023/24
    Description: Top tier English football league

  • La Liga
    Country: Spain
    Season: 2023/24
    Description: Spanish top division

TOURNAMENTS (8):
  • UEFA Champions League
    Country: Europe
    Season: 2023/24
    Description: European club championship
```

### Extending the Scraper

The scraping system is designed to be easily extensible:

1. **Add new sites**: Update the `SITE_URLS` dictionary in `scrape.py`
2. **Add new extraction types**: Create new CAMEL agents with specialized prompts
3. **Customize data structures**: Modify the JSON output format for different data types
4. **Add new analysis**: Implement additional processing for extracted data

The modular architecture allows you to:
- Add new football websites to the scraping list
- Create specialized LLM agents for different types of data extraction
- Customize the output format for different use cases
- Integrate with other data processing pipelines

## Contributing

Pull requests and issues are welcome! Please add tests for new features.

## License

MIT
