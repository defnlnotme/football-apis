import os
import sys
import json
import logging
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from football_apis.clients.the_odds_api import TheOddsApiClient
from football_apis.clients.clubelo_api import ClubEloClient
from football_apis.clients.football_data_api import FootballDataPerformanceStatsClient, FootballDataMatchHistoryClient

def get_api_key(service_name: str) -> str:
    """
    Read API key for a specific service from apikeys.json
    
    Args:
        service_name (str): Name of the service in apikeys.json
        
    Returns:
        str: The API key for the specified service
        
    Raises:
        FileNotFoundError: If apikeys.json is not found
        json.JSONDecodeError: If apikeys.json has invalid JSON format
        KeyError: If the service or API key is not found in the JSON
    """
    try:
        with open('apikeys.json', 'r') as f:
            api_keys = json.load(f)
            return api_keys[service_name]['api_key']
    except FileNotFoundError:
        logger.error("❌ apikeys.json file not found")
        raise
    except json.JSONDecodeError:
        logger.error("❌ Invalid JSON format in apikeys.json")
        raise
    except KeyError:
        logger.error(f"❌ API key not found for service '{service_name}' in apikeys.json")
        raise

async def fetch_betting_odds(output_dir: Path, args: argparse.Namespace) -> None:
    """Fetch betting odds data asynchronously"""
    try:
        api_key = get_api_key('the-odds-api')
        logger.info(f"Initializing TheOddsApiClient with API key: {api_key[:5]}...{api_key[-3:]}")
        
        client = TheOddsApiClient(api_key=api_key, cache_enabled=False)
        logger.info("Client initialized successfully")
        
        # Get available sports
        sports = client.get_sports()
        if not sports:
            logger.warning("No sports data returned")
            return
            
        # Prepare output data
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "sports": sports,
            "results": {}
        }
        
        # Call each specified method
        for method_name in args.betting_method:
            logger.info(f"Calling method: {method_name}")
            method = getattr(client, method_name)
            
            try:
                if method_name == 'get_odds':
                    data = method(
                        sport_key=args.sport or "soccer_italy_serie_a",
                        regions=args.region or "eu",
                        markets=args.markets or "h2h",
                        odds_format=args.odds_format or "decimal",
                        bookmakers=args.bookmakers,
                        commence_time_from=args.date_from,
                        commence_time_to=args.date_to
                    )
                elif method_name == 'get_sports':
                    data = method(all_available=args.all_available if hasattr(args, 'all_available') else False)
                elif method_name == 'get_scores':
                    data = method(
                        sport_key=args.sport or "soccer_italy_serie_a",
                        days_from=args.days_from if hasattr(args, 'days_from') else 3,
                        date_format=args.date_format or "iso"
                    )
                elif method_name == 'get_events':
                    data = method(
                        sport_key=args.sport or "soccer_italy_serie_a",
                        date_format=args.date_format or "iso",
                        bookmakers=args.bookmakers
                    )
                elif method_name == 'get_historical_odds':
                    if not args.event_id:
                        logger.error("❌ event_id is required for get_historical_odds")
                        continue
                    data = method(
                        sport_key=args.sport or "soccer_italy_serie_a",
                        event_id=args.event_id,
                        regions=args.region or "eu",
                        markets=args.markets or "h2h,spreads,totals",
                        date_format=args.date_format or "iso",
                        odds_format=args.odds_format or "decimal",
                        bookmakers=args.bookmakers
                    )
                elif method_name == 'get_historical_odds_archive':
                    if not args.commence_time:
                        logger.error("❌ commence_time is required for get_historical_odds_archive")
                        continue
                    data = method(
                        sport_key=args.sport or "soccer_italy_serie_a",
                        commence_time=args.commence_time,
                        regions=args.region or "eu",
                        markets=args.markets or "h2h,spreads,totals",
                        date_format=args.date_format or "iso",
                        odds_format=args.odds_format or "decimal",
                        bookmakers=args.bookmakers
                    )
                
                output_data["results"][method_name] = data
                logger.info(f"✅ Successfully fetched data from {method_name}")
                
            except Exception as e:
                logger.error(f"❌ Error in {method_name}: {str(e)}")
                output_data["results"][method_name] = {"error": str(e)}
        
        # Save to file
        output_file = output_dir / f"betting_odds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"✅ Saved betting odds data to {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Error in betting odds fetch: {str(e)}", exc_info=True)

async def fetch_team_ratings(output_dir: Path, args: argparse.Namespace) -> None:
    """Fetch team ratings data asynchronously"""
    try:
        client = ClubEloClient()
        logger.info("Fetching team ratings data...")
        
        # Prepare output data
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        # Call each specified method
        for method_name in args.ratings_method:
            logger.info(f"Calling method: {method_name}")
            method = getattr(client, method_name)
            
            try:
                if method_name == 'get_team_elo':
                    data = method(
                        team_name=args.team,
                        team_id=args.team_id,
                        date=args.date
                    )
                elif method_name == 'search_teams':
                    if not args.team:
                        logger.error("❌ team name is required for search_teams")
                        continue
                    data = method(query=args.team)
                elif method_name == 'get_top_teams':
                    data = method(
                        limit=args.limit if hasattr(args, 'limit') else 20,
                        country=args.country,
                        min_elo=args.min_elo
                    )
                elif method_name == 'get_historical_elos':
                    if not args.team_id:
                        logger.error("❌ team_id is required for get_historical_elos")
                        continue
                    data = method(
                        team_id=args.team_id,
                        start_date=args.date_from,
                        end_date=args.date_to
                    )
                elif method_name == 'get_team_form':
                    if not args.team_id:
                        logger.error("❌ team_id is required for get_team_form")
                        continue
                    data = method(
                        team_id=args.team_id,
                        matches=args.matches or 5,
                        competition=args.competition
                    )
                
                output_data["results"][method_name] = data
                logger.info(f"✅ Successfully fetched data from {method_name}")
                
            except Exception as e:
                logger.error(f"❌ Error in {method_name}: {str(e)}")
                output_data["results"][method_name] = {"error": str(e)}
        
        # Save to file
        output_file = output_dir / f"team_ratings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"✅ Saved team ratings data to {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Error in team ratings fetch: {str(e)}", exc_info=True)

async def fetch_performance_stats(output_dir: Path, args: argparse.Namespace) -> None:
    """Fetch performance stats data asynchronously"""
    try:
        client = FootballDataPerformanceStatsClient()
        logger.info("Fetching performance stats data...")
        
        # Prepare output data
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        # Call each specified method
        for method_name in args.stats_method:
            logger.info(f"Calling method: {method_name}")
            method = getattr(client, method_name)
            
            try:
                if method_name == 'get_team_statistics':
                    if not args.team_id:
                        logger.error("❌ team_id is required for get_team_statistics")
                        continue
                    data = method(
                        team_id=args.team_id,
                        season=args.season,
                        competition_id=args.competition_id
                    )
                elif method_name == 'get_team_matches':
                    if not args.team_id:
                        logger.error("❌ team_id is required for get_team_matches")
                        continue
                    data = method(
                        team_id=args.team_id,
                        season=args.season,
                        competition_id=args.competition_id,
                        status=args.status or "FINISHED"
                    )
                elif method_name == 'get_team_standings':
                    if not args.team_id or not args.competition_id:
                        logger.error("❌ team_id and competition_id are required for get_team_standings")
                        continue
                    data = method(
                        team_id=args.team_id,
                        competition_id=args.competition_id,
                        season=args.season
                    )
                elif method_name == 'get_player_statistics':
                    if not args.player_id:
                        logger.error("❌ player_id is required for get_player_statistics")
                        continue
                    data = method(
                        player_id=args.player_id,
                        season=args.season,
                        competition_id=args.competition_id
                    )
                
                output_data["results"][method_name] = data
                logger.info(f"✅ Successfully fetched data from {method_name}")
                
            except Exception as e:
                logger.error(f"❌ Error in {method_name}: {str(e)}")
                output_data["results"][method_name] = {"error": str(e)}
        
        # Save to file
        output_file = output_dir / f"performance_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"✅ Saved performance stats data to {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Error in performance stats fetch: {str(e)}", exc_info=True)

async def fetch_match_history(output_dir: Path, args: argparse.Namespace) -> None:
    """Fetch match history data asynchronously"""
    try:
        client = FootballDataMatchHistoryClient()
        logger.info("Fetching match history data...")
        
        # Prepare output data
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        # Call each specified method
        for method_name in args.history_method:
            logger.info(f"Calling method: {method_name}")
            method = getattr(client, method_name)
            
            try:
                if method_name == 'get_team_matches':
                    if not args.team1:
                        logger.error("❌ team1 is required for get_team_matches")
                        continue
                    data = method(
                        team_id=args.team1,
                        limit=args.limit or 8,
                        competitions=args.competitions.split(',') if args.competitions else None,
                        status=args.status or "FINISHED"
                    )
                elif method_name == 'get_head_to_head':
                    if not args.team1 or not args.team2:
                        logger.error("❌ team1 and team2 are required for get_head_to_head")
                        continue
                    data = method(
                        team1_id=args.team1,
                        team2_id=args.team2,
                        limit=args.limit or 10,
                        date_from=args.date_from,
                        date_to=args.date_to
                    )
                elif method_name == 'get_team_info':
                    if not args.team1:
                        logger.error("❌ team1 is required for get_team_info")
                        continue
                    data = method(team_id=args.team1)
                elif method_name == 'search_teams':
                    if not args.team:
                        logger.error("❌ team name is required for search_teams")
                        continue
                    data = method(name=args.team)
                elif method_name == 'get_competition_matches':
                    if not args.competition_id:
                        logger.error("❌ competition_id is required for get_competition_matches")
                        continue
                    data = method(
                        competition_id=args.competition_id,
                        season=args.season,
                        matchday=args.matchday
                    )
                elif method_name == 'get_match_details':
                    if not args.match_id:
                        logger.error("❌ match_id is required for get_match_details")
                        continue
                    data = method(match_id=args.match_id)
                
                output_data["results"][method_name] = data
                logger.info(f"✅ Successfully fetched data from {method_name}")
                
            except Exception as e:
                logger.error(f"❌ Error in {method_name}: {str(e)}")
                output_data["results"][method_name] = {"error": str(e)}
        
        # Save to file
        output_file = output_dir / f"match_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"✅ Saved match history data to {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Error in match history fetch: {str(e)}", exc_info=True)

def get_cache_dir() -> Path:
    """Get the cache directory path."""
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def get_cached_params(endpoint: str) -> Optional[Dict]:
    """Get cached parameters for an endpoint."""
    cache_file = get_cache_dir() / f"{endpoint}_params.pkl"
    if not cache_file.exists():
        return None
        
    try:
        # Check if cache is expired (1 week)
        if datetime.now().timestamp() - cache_file.stat().st_mtime > 7 * 24 * 3600:
            return None
            
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error reading cache: {str(e)}")
        return None

def save_cached_params(endpoint: str, params: Dict) -> None:
    """Save parameters to cache."""
    cache_file = get_cache_dir() / f"{endpoint}_params.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(params, f)
    except Exception as e:
        logger.error(f"Error saving cache: {str(e)}")

def list_available_parameters(endpoint: str) -> None:
    """
    List available parameters for a specific endpoint.
    
    Args:
        endpoint (str): The endpoint to list parameters for
    """
    try:
        # Try to get cached parameters first
        cached_params = get_cached_params(endpoint)
        if cached_params:
            print(cached_params["output"])
            return
            
        output = []
        
        if endpoint == 'betting_odds':
            client = TheOddsApiClient(api_key=get_api_key('the-odds-api'))
            
            # Get available sports
            sports = client.get_sports()
            output.append("\nAvailable sports:")
            for sport in sports:
                output.append(f"- {sport['key']}: {sport['title']}")
            
            output.append("\nAvailable regions:")
            output.append("- eu (Europe)")
            output.append("- us (United States)")
            output.append("- uk (United Kingdom)")
            output.append("- au (Australia)")
            
            output.append("\nAvailable markets:")
            output.append("- h2h (Head to Head)")
            output.append("- spreads (Point Spreads)")
            output.append("- totals (Over/Under)")
            
            output.append("\nAvailable odds formats:")
            output.append("- decimal (e.g., 2.50)")
            output.append("- american (e.g., +150)")
            output.append("- hongkong (e.g., 1.50)")
            output.append("- malay (e.g., 0.50)")
            output.append("- indonesian (e.g., -2.00)")
            
            output.append("\nAvailable date formats:")
            output.append("- iso (ISO 8601 format)")
            output.append("- unix (Unix timestamp)")
            
        elif endpoint == 'team_ratings':
            client = ClubEloClient(api_key=get_api_key('clubelo'))
            
            output.append("\nTeam Ratings Client Methods:")
            output.append("\n1. get_team_elo(team_name=None, team_id=None, date=None):")
            
            # Get top teams to show available team names and IDs
            output.append("\n   Available teams (top 20):")
            top_teams = client.get_top_teams(limit=20)
            for team in top_teams:
                output.append(f"   - {team['team_name']} (ID: {team['team_id']})")
            
            output.append("\n   Parameters:")
            output.append("   - team_name: Name of the team (e.g., 'Barcelona')")
            output.append("   - team_id: ClubElo team ID")
            output.append("   - date: Date in YYYY-MM-DD format (default: latest available)")
            
            output.append("\n2. search_teams(query):")
            output.append("   - query: Search query (team name or part of it)")
            
            output.append("\n3. get_top_teams(limit=20, country=None, min_elo=None):")
            output.append("   - limit: Maximum number of teams to return")
            output.append("   - country: Filter by country code (e.g., 'ENG', 'ESP')")
            output.append("   - min_elo: Minimum Elo rating")
            
            output.append("\n4. get_historical_elos(team_id, start_date=None, end_date=None):")
            output.append("   - team_id: ClubElo team ID")
            output.append("   - start_date: Start date in YYYY-MM-DD format")
            output.append("   - end_date: End date in YYYY-MM-DD format")
            
            output.append("\n5. get_team_form(team_id, matches=5, competition=None):")
            output.append("   - team_id: ClubElo team ID")
            output.append("   - matches: Number of recent matches to include")
            output.append("   - competition: Filter by competition")
            
        elif endpoint == 'performance_stats':
            client = FootballDataPerformanceStatsClient(api_key=get_api_key('football-data'))
            
            output.append("\nPerformance Stats Client Methods:")
            output.append("\n1. get_team_statistics(team_id, season=None, competition_id=None):")
            
            # Get available competitions
            output.append("\n   Available competitions:")
            competitions = client.get_cached("/competitions")
            if isinstance(competitions, dict) and "competitions" in competitions:
                for comp in competitions.get("competitions", []):
                    if isinstance(comp, dict) and "name" in comp and "id" in comp:
                        output.append(f"   - {comp['name']} (ID: {comp['id']})")
            
            output.append("\n   Parameters:")
            output.append("   - team_id: ID of the team")
            output.append("   - season: Season year (e.g., 2023 for 2023/2024 season)")
            output.append("   - competition_id: ID of the competition")
            
            output.append("\n2. get_team_matches(team_id, season=None, competition_id=None, status='FINISHED'):")
            output.append("   - team_id: ID of the team")
            output.append("   - season: Season year")
            output.append("   - competition_id: ID of the competition")
            output.append("   - status: Match status (e.g., 'FINISHED', 'SCHEDULED')")
            
            output.append("\n3. get_team_standings(team_id, competition_id, season=None):")
            output.append("   - team_id: ID of the team")
            output.append("   - competition_id: ID of the competition")
            output.append("   - season: Season year")
            
        elif endpoint == 'match_history':
            client = FootballDataMatchHistoryClient(api_key=get_api_key('football-data'))
            
            output.append("\nMatch History Client Methods:")
            output.append("\n1. get_team_matches(team_id, limit=8, competitions=None, status='FINISHED'):")
            
            # Get available competitions
            output.append("\n   Available competitions:")
            competitions = client.get_cached("/competitions")
            if isinstance(competitions, dict) and "competitions" in competitions:
                for comp in competitions.get("competitions", []):
                    if isinstance(comp, dict) and "name" in comp and "id" in comp:
                        output.append(f"   - {comp['name']} (ID: {comp['id']})")
            
            output.append("\n   Parameters:")
            output.append("   - team_id: ID of the team")
            output.append("   - limit: Maximum number of matches to return")
            output.append("   - competitions: Filter by competition codes (e.g., ['PL', 'CL'])")
            output.append("   - status: Filter by match status (e.g., 'FINISHED', 'SCHEDULED')")
            
            output.append("\n2. get_head_to_head(team1_id, team2_id, limit=10, date_from=None, date_to=None):")
            output.append("   - team1_id: ID of the first team")
            output.append("   - team2_id: ID of the second team")
            output.append("   - limit: Maximum number of matches to return")
            output.append("   - date_from: Filter matches after this date (YYYY-MM-DD)")
            output.append("   - date_to: Filter matches before this date (YYYY-MM-DD)")
            
            output.append("\n3. get_team_info(team_id):")
            output.append("   - team_id: ID of the team")
            
            output.append("\n4. search_teams(name):")
            output.append("   - name: Team name to search for")
            
        else:
            output.append(f"❌ Unknown endpoint: {endpoint}")
            return
            
        # Join the output and print it
        output_str = "\n".join(output)
        print(output_str)
        
        # Cache the output
        save_cached_params(endpoint, {
            "output": output_str,
            "timestamp": datetime.now().timestamp()
        })
            
    except Exception as e:
        logger.error(f"❌ Error listing parameters: {str(e)}", exc_info=True)

async def main():
    """Main function to fetch data from specified endpoints"""
    parser = argparse.ArgumentParser(
        description='Football Data API Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available parameters for betting odds
  python client.py betting_odds --list-params

  # Fetch data from all endpoints
  python client.py all

  # Fetch only betting odds and team ratings
  python client.py betting_odds team_ratings

  # Get help for a specific endpoint
  python client.py betting_odds --info

  # Get betting odds with specific parameters
  python client.py betting_odds --sport soccer_epl --region eu --odds-format decimal

  # Get team ratings for a specific team
  python client.py team_ratings --team Barcelona --date 2024-01-01

  # Get performance stats for a team in a specific season
  python client.py performance_stats --team-id 81 --season 2023 --competition-id 2021

  # Get match history between two teams
  python client.py match_history --team1 81 --team2 65 --limit 10

  # Call multiple methods for betting odds
  python client.py betting_odds --betting-method get_sports get_odds get_scores --sport soccer_epl

  # Get team ratings with multiple methods
  python client.py team_ratings --ratings-method get_team_elo search_teams --team Barcelona

  # Get performance stats with multiple methods
  python client.py performance_stats --stats-method get_team_statistics get_team_matches --team-id 81

  # Get match history with multiple methods
  python client.py match_history --history-method get_team_matches get_head_to_head --team1 81 --team2 65
"""
    )
    
    # Common arguments
    parser.add_argument('endpoints', nargs='+', 
                      choices=['all', 'betting_odds', 'team_ratings', 'performance_stats', 'match_history'],
                      help='Endpoints to fetch data from (use "all" for all endpoints)')
    parser.add_argument('--list-params', action='store_true',
                      help='List available parameters for the specified endpoint')
    parser.add_argument('--info', action='store_true',
                      help='Show detailed information for the specified endpoint')
    
    # Betting odds specific arguments
    betting_group = parser.add_argument_group('Betting Odds Options')
    betting_group.add_argument('--betting-method', type=str, nargs='+',
                             choices=['get_odds', 'get_sports', 'get_scores', 'get_events', 'get_historical_odds', 'get_historical_odds_archive'],
                             default=['get_odds'],
                             help='Method(s) to call (default: get_odds). Can specify multiple methods.')
    betting_group.add_argument('--sport', type=str,
                             help='Sport key (e.g., soccer_epl)')
    betting_group.add_argument('--region', type=str, choices=['eu', 'us', 'uk', 'au'],
                             help='Region for odds (eu, us, uk, au)')
    betting_group.add_argument('--markets', type=str,
                             help='Markets to include (comma-separated, e.g., h2h,spreads,totals)')
    betting_group.add_argument('--odds-format', type=str,
                             choices=['decimal', 'american', 'hongkong', 'malay', 'indonesian'],
                             help='Format for odds')
    betting_group.add_argument('--bookmakers', type=str,
                             help='Comma-separated list of bookmakers')
    betting_group.add_argument('--date-from', type=str,
                             help='Filter events starting after this date (YYYY-MM-DD)')
    betting_group.add_argument('--date-to', type=str,
                             help='Filter events starting before this date (YYYY-MM-DD)')
    betting_group.add_argument('--all-available', action='store_true',
                             help='Include all available sports, not just active ones')
    betting_group.add_argument('--days-from', type=int,
                             help='Number of days to look back for completed events (max 3)')
    betting_group.add_argument('--date-format', type=str, choices=['iso', 'unix'],
                             help='Format for dates (iso or unix)')
    betting_group.add_argument('--event-id', type=str,
                             help='Event ID for historical odds')
    betting_group.add_argument('--commence-time', type=str,
                             help='Commence time for historical odds archive (YYYY-MM-DD)')
    
    # Team ratings specific arguments
    ratings_group = parser.add_argument_group('Team Ratings Options')
    ratings_group.add_argument('--ratings-method', type=str, nargs='+',
                             choices=['get_team_elo', 'search_teams', 'get_top_teams', 'get_historical_elos', 'get_team_form'],
                             default=['get_team_elo'],
                             help='Method(s) to call (default: get_team_elo). Can specify multiple methods.')
    ratings_group.add_argument('--team', type=str,
                             help='Team name (e.g., Barcelona)')
    ratings_group.add_argument('--team-id', type=int,
                             help='ClubElo team ID')
    ratings_group.add_argument('--date', type=str,
                             help='Date for ratings (YYYY-MM-DD)')
    ratings_group.add_argument('--country', type=str,
                             help='Country code (e.g., ENG, ESP)')
    ratings_group.add_argument('--min-elo', type=int,
                             help='Minimum Elo rating')
    ratings_group.add_argument('--matches', type=int,
                             help='Number of recent matches for form')
    ratings_group.add_argument('--competition', type=str,
                             help='Competition name for form')
    
    # Performance stats specific arguments
    stats_group = parser.add_argument_group('Performance Stats Options')
    stats_group.add_argument('--stats-method', type=str, nargs='+',
                           choices=['get_team_statistics', 'get_team_matches', 'get_team_standings', 'get_player_statistics'],
                           default=['get_team_statistics'],
                           help='Method(s) to call (default: get_team_statistics). Can specify multiple methods.')
    stats_group.add_argument('--season', type=int,
                           help='Season year (e.g., 2023 for 2023/2024)')
    stats_group.add_argument('--competition-id', type=int,
                           help='Competition ID')
    stats_group.add_argument('--status', type=str, choices=['FINISHED', 'SCHEDULED'],
                           help='Match status')
    
    # Match history specific arguments
    history_group = parser.add_argument_group('Match History Options')
    history_group.add_argument('--history-method', type=str, nargs='+',
                             choices=['get_team_matches', 'get_head_to_head', 'get_team_info', 'search_teams', 'get_competition_matches', 'get_match_details'],
                             default=['get_team_matches'],
                             help='Method(s) to call (default: get_team_matches). Can specify multiple methods.')
    history_group.add_argument('--team1', type=int,
                             help='ID of the first team')
    history_group.add_argument('--team2', type=int,
                             help='ID of the second team')
    history_group.add_argument('--limit', type=int,
                             help='Maximum number of matches to return')
    history_group.add_argument('--competitions', type=str,
                             help='Comma-separated list of competition codes')
    
    args = parser.parse_args()
    
    if args.info:
        if len(args.endpoints) != 1:
            print("❌ Please specify exactly one endpoint when using --info")
            return
            
        endpoint = args.endpoints[0]
        if endpoint == 'betting_odds':
            print("""
Betting Odds API Help
====================
This endpoint provides access to betting odds data from various bookmakers.

Key Features:
- Get odds for upcoming events
- Access historical odds data
- View odds in different formats (decimal, american, etc.)
- Filter by sport, region, and bookmaker
- Call multiple methods in a single run

Available Methods:
- get_odds: Get odds for upcoming events (default)
- get_sports: Get list of available sports
- get_scores: Get scores for recently completed events
- get_events: Get list of upcoming events
- get_historical_odds: Get historical odds for a specific event
- get_historical_odds_archive: Get historical odds for events at a specific time

Example Usage:
1. Get odds for Premier League matches:
   python client.py betting_odds --sport soccer_epl

2. List available sports:
   python client.py betting_odds --method get_sports

3. Get scores for recent matches:
   python client.py betting_odds --method get_scores --sport soccer_epl --days-from 3

4. Get historical odds for an event:
   python client.py betting_odds --method get_historical_odds --sport soccer_epl --event-id 12345

5. Call multiple methods in one run:
   python client.py betting_odds --method get_sports get_odds get_scores --sport soccer_epl

6. List available parameters:
   python client.py betting_odds --list-params
""")
        elif endpoint == 'team_ratings':
            print("""
Team Ratings API Help
====================
This endpoint provides access to team strength ratings and Elo scores.

Key Features:
- Get current Elo ratings for teams
- View historical Elo changes
- Access team form and performance metrics
- Search for teams by name

Example Usage:
1. Get top teams by Elo rating:
   python client.py team_ratings

2. List available parameters:
   python client.py team_ratings --list-params

3. Get Elo rating for a specific team:
   python client.py team_ratings --team Barcelona --date 2024-01-01
""")
        elif endpoint == 'performance_stats':
            print("""
Performance Stats API Help
=========================
This endpoint provides access to detailed team and player statistics.

Key Features:
- Get team performance metrics
- View match statistics
- Access league standings
- Track historical performance

Example Usage:
1. Get team statistics:
   python client.py performance_stats --team-id 81

2. List available parameters:
   python client.py performance_stats --list-params

3. Get stats for a specific team and season:
   python client.py performance_stats --team-id 81 --season 2023 --competition-id 2021
""")
        elif endpoint == 'match_history':
            print("""
Match History API Help
=====================
This endpoint provides access to match results and head-to-head data.

Key Features:
- Get recent match results
- View head-to-head statistics
- Access team match history
- Search for specific matches

Example Usage:
1. Get recent matches:
   python client.py match_history --team1 81

2. List available parameters:
   python client.py match_history --list-params

3. Get head-to-head stats:
   python client.py match_history --team1 81 --team2 65 --limit 10
""")
        return
    
    if args.list_params:
        if len(args.endpoints) != 1:
            print("❌ Please specify exactly one endpoint when using --list-params")
            return
        list_available_parameters(args.endpoints[0])
        return
    
    # Create output directory
    output_dir = Path('data') / datetime.now().strftime('%Y%m%d')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which endpoints to fetch
    endpoints = ['betting_odds', 'team_ratings', 'performance_stats', 'match_history'] if 'all' in args.endpoints else args.endpoints
    
    # Create tasks for each endpoint
    tasks = []
    if 'betting_odds' in endpoints:
        tasks.append(fetch_betting_odds(output_dir, args))
    if 'team_ratings' in endpoints:
        tasks.append(fetch_team_ratings(output_dir, args))
    if 'performance_stats' in endpoints:
        tasks.append(fetch_performance_stats(output_dir, args))
    if 'match_history' in endpoints:
        tasks.append(fetch_match_history(output_dir, args))
    
    # Run all tasks concurrently
    await asyncio.gather(*tasks)
    logger.info("All data fetching completed!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        raise
