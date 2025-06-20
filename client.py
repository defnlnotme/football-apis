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
from football_apis.clients.football_data_api import FootballDataClient

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

async def fetch_the_odds_api(output_dir: Path, args: argparse.Namespace) -> None:
    """Fetch betting odds data asynchronously"""
    try:
        api_key = get_api_key('the-odds-api')
        logger.info(f"Initializing TheOddsApiClient with API key: {api_key[:5]}...{api_key[-3:]}")
        
        client = TheOddsApiClient(api_key=api_key, cache_enabled=False)
        logger.info("Client initialized successfully")
        
        # Prepare output data
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        # Call each specified method
        for method_name in args.the_odds_api_method:
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
                        commence_time_to=args.date_to,
                        event_ids=args.event_ids
                    )
                elif method_name == 'get_sports':
                    data = method(all_available=args.all_available if hasattr(args, 'all_available') else False)
                    output_data["sports"] = data  # Only add sports if requested
                elif method_name == 'get_scores':
                    data = method(
                        sport_key=args.sport or "soccer_italy_serie_a",
                        days_from=args.days_from if hasattr(args, 'days_from') else 3,
                        date_format=args.date_format or "iso"
                    )
                elif method_name == 'get_events':
                    data = method(
                        sport_key=args.sport or "soccer_italy_serie_a",
                        date_format=args.date_format or "iso"
                    )
                elif method_name == 'get_historical_odds':
                    odds_date = args.odds_date or datetime.now().strftime('%Y-%m-%dT00:00:00Z')
                    data = method(
                        sport_key=args.sport or "soccer_italy_serie_a",
                        date=odds_date,
                        regions=args.region or "eu",
                        markets=args.markets or "h2h,spreads,totals",
                        date_format=args.date_format or "iso",
                        odds_format=args.odds_format or "decimal"
                    )
                elif method_name == 'get_historical_odds_archive':
                    if not args.odds_date:
                        logger.error("❌ odds-date is required for get_historical_odds_archive")
                        continue
                    data = method(
                        sport_key=args.sport or "soccer_italy_serie_a",
                        date=args.odds_date,
                        regions=args.region or "eu",
                        markets=args.markets or "h2h,spreads,totals",
                        date_format=args.date_format or "iso",
                        odds_format=args.odds_format or "decimal"
                    )
                else:
                    data = method()
                
                if not (isinstance(data, dict) and "error" in data):
                    output_data["results"][method_name] = data
                    logger.info(f"✅ Successfully fetched data from {method_name}")
                else:
                    logger.warning(f"❌ Skipping {method_name} due to error in result.")
                
            except Exception as e:
                logger.error(f"❌ Error in {method_name}: {str(e)}")
                # Do not add to results if error
        
        # Only save to file if there is at least one successful result
        if output_data["results"]:
            output_file = output_dir / f"the-odds-api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"✅ Saved betting odds data to {output_file}")
        else:
            logger.warning("❌ Not saving file because all methods returned an error.")
        
    except Exception as e:
        logger.error(f"❌ Error in betting odds fetch: {str(e)}", exc_info=True)

async def fetch_clubelo(output_dir: Path, args: argparse.Namespace) -> None:
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
        for method_name in args.clubelo_method:
            logger.info(f"Calling method: {method_name}")
            method = getattr(client, method_name)
            
            try:
                if method_name == 'get_team_elo':
                    data = method(
                        team_name=args.team,
                        date=args.elo_date
                    )
                elif method_name == 'get_top_teams':
                    data = method(
                        date=args.elo_date,
                        limit=args.limit if hasattr(args, 'limit') else 20,
                        country=args.country,
                        min_elo=args.min_elo
                    )
                elif method_name == 'get_fixtures':
                    data = method()
                else:
                    data = method()
                
                if not (isinstance(data, dict) and "error" in data):
                    output_data["results"][method_name] = data
                    logger.info(f"✅ Successfully fetched data from {method_name}")
                else:
                    logger.warning(f"❌ Skipping {method_name} due to error in result.")
                
            except Exception as e:
                logger.error(f"❌ Error in {method_name}: {str(e)}")
                # Do not add to results if error
        
        # Only save to file if there is at least one successful result
        if output_data["results"]:
            output_file = output_dir / f"clubelo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"✅ Saved team ratings data to {output_file}")
        else:
            logger.warning("❌ Not saving file because all methods returned an error.")
        
    except Exception as e:
        logger.error(f"❌ Error in team ratings fetch: {str(e)}", exc_info=True)

async def fetch_football_data(output_dir: Path, args: argparse.Namespace) -> None:
    """Fetch performance stats data asynchronously"""
    try:
        client = FootballDataClient(api_key=get_api_key('football-data'))
        logger.info("Fetching performance stats data...")
        
        # Prepare output data
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        # Call each specified method
        for method_name in args.football_data_method:
            logger.info(f"Calling method: {method_name}")
            method = getattr(client, method_name)
            
            try:
                if method_name == 'get_team_statistics':
                    if not args.team_id:
                        logger.error("❌ team_id is required for get_team_statistics")
                        continue
                    data = method(
                        team_id=args.team_id
                    )
                elif method_name == 'get_team_matches':
                    if not args.team_id:
                        logger.error("❌ team_id is required for get_team_matches")
                        continue
                    data = method(
                        team_id=args.team_id,
                        season=args.season,
                        competition_id=args.competition_id,
                        status=args.status or "FINISHED",
                        limit=args.fd_limit,
                        competitions=args.competitions
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
                elif method_name == 'search_teams':
                    data = method(
                        name=args.team_name,
                        season=args.season,
                        limit=args.fd_limit,
                        offset=args.offset
                    )
                elif method_name == 'get_areas':
                    data = method()
                elif method_name == 'get_area':
                    if not args.area_id:
                        logger.error("❌ area_id is required for get_area")
                        continue
                    data = method(area_id=args.area_id)
                elif method_name == 'get_person':
                    if not args.person_id:
                        logger.error("❌ person_id is required for get_person")
                        continue
                    data = method(person_id=args.person_id)
                elif method_name == 'get_person_matches':
                    if not args.person_id:
                        logger.error("❌ person_id is required for get_person_matches")
                        continue
                    data = method(
                        person_id=args.person_id,
                        lineup=args.lineup,
                        e=args.e,
                        date_from=args.person_date_from,
                        date_to=args.person_date_to,
                        competitions=args.person_competitions,
                        limit=args.person_limit,
                        offset=args.person_offset
                    )
                elif method_name == 'get_match_details':
                    if not args.match_id:
                        logger.error("❌ match_id is required for get_match_details")
                        continue
                    data = method(match_id=args.match_id)
                elif method_name == 'get_competition_matches':
                    if not args.competition_id:
                        logger.error("❌ competition_id is required for get_competition_matches")
                        continue
                    data = method(
                        competition_id=args.competition_id,
                        season=args.season,
                        matchday=args.matchday
                    )
                
                if not (isinstance(data, dict) and "error" in data):
                    output_data["results"][method_name] = data
                    logger.info(f"✅ Successfully fetched data from {method_name}")
                else:
                    logger.warning(f"❌ Skipping {method_name} due to error in result.")
                
            except Exception as e:
                logger.error(f"❌ Error in {method_name}: {str(e)}")
                # Do not add to results if error
        
        # Only save to file if there is at least one successful result
        if output_data["results"]:
            output_file = output_dir / f"football-data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"✅ Saved performance stats data to {output_file}")
        else:
            logger.warning("❌ Not saving file because all methods returned an error.")
        
    except Exception as e:
        logger.error(f"❌ Error in performance stats fetch: {str(e)}", exc_info=True)

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
        
        if endpoint == 'the-odds-api':
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
            
        elif endpoint == 'clubelo':
            client = ClubEloClient()
            
            output.append("\nTeam Ratings Client Methods:")
            output.append("\n1. get_team_elo [--team TEAM] [--team-id TEAM_ID] [--date DATE]: Get Elo rating for a specific team.")
            
            # Get top teams to show available team names and IDs
            output.append("\n   Available teams (top 20):")
            top_teams = client.get_top_teams(limit=20)
            for team in top_teams:
                output.append(f"   - {team['team_name']} (ID: {team['team_id']})")
            
            output.append("\n   Parameters:")
            output.append("   - team_name: Name of the team (e.g., 'Barcelona')")
            output.append("   - team_id: ClubElo team ID")
            output.append("   - date: Date in YYYY-MM-DD format (default: latest available)")
            
            output.append("\n2. get_top_teams [--limit LIMIT] [--country COUNTRY] [--min-elo MIN_ELO]: Get top teams by Elo rating.")
            output.append("   - limit: Maximum number of teams to return")
            output.append("   - country: Filter by country code (e.g., 'ENG', 'ESP')")
            output.append("   - min_elo: Minimum Elo rating")
            
            output.append("\n3. get_fixtures: Get fixtures for a team.")
            
        elif endpoint == 'football-data':
            client = FootballDataClient(api_key=get_api_key('football-data'))
            
            output.append("\nPerformance Stats Client Methods:")
            output.append("\n1. get_team_statistics --team-id TEAM_ID [--season SEASON] [--competition-id COMPETITION_ID]: Get team statistics for a specific season and competition.")
            
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
            
            output.append("\n2. get_team_matches --team-id TEAM_ID [--season SEASON] [--competition-id COMPETITION_ID] [--status STATUS] [--fd-limit FD_LIMIT] [--competitions COMPETITIONS]: Get team matches for a specific season and competition, or recent matches for a team.")
            output.append("   - team_id: ID of the team")
            output.append("   - season: Season year")
            output.append("   - competition_id: ID of the competition")
            output.append("   - status: Match status (e.g., 'FINISHED', 'SCHEDULED')")
            output.append("   - fd-limit: Maximum number of matches to return")
            output.append("   - competitions: Filter by competition")
            
            output.append("\n3. get_team_standings --team-id TEAM_ID --competition-id COMPETITION_ID [--season SEASON]: Get team standings in a competition.")
            output.append("   - team_id: ID of the team")
            output.append("   - competition_id: ID of the competition")
            output.append("   - season: Season year")
            
            output.append("\n4. get_player_statistics --player-id PLAYER_ID [--season SEASON] [--competition-id COMPETITION_ID]: Get player statistics for a specific season and competition.")
            output.append("   - player_id: ID of the player")
            output.append("   - season: Season year")
            output.append("   - competition_id: ID of the competition")
            
            output.append("\n5. get_head_to_head --team1 TEAM1_ID --team2 TEAM2_ID [--limit LIMIT] [--person-date-from DATE_FROM] [--person-date-to DATE_TO]: Get head-to-head matches between two teams.")
            output.append("   - team1: ID of the first team")
            output.append("   - team2: ID of the second team")
            output.append("   - limit: Maximum number of matches to return")
            output.append("   - person-date-from: Start date of the match range")
            output.append("   - person-date-to: End date of the match range")
            
            output.append("\n6. get_team_info --team-id TEAM_ID: Get detailed information about a team.")
            output.append("   - team_id: ID of the team")
            
            output.append("\n7. get_match_details --match-id MATCH_ID: Get detailed information about a specific match.")
            output.append("\n8. search_teams [--team-name TEAM_NAME] [--season SEASON] [--fd-limit FD_LIMIT] [--offset OFFSET]: Search for teams by name or get a list of teams.")
            output.append("   - team_name: Optional name to search for teams")
            output.append("   - season: Optional season year to filter teams")
            output.append("   - fd-limit: Optional maximum number of teams to return")
            output.append("   - offset: Optional offset for pagination")
            
            output.append("\n9. get_areas: Get a list of all areas (countries, regions, etc).")
            output.append("   - No parameters.")
            output.append("\n10. get_area --area-id AREA_ID: Get details for a specific area by its ID.")
            output.append("   - area_id: ID of the area")
            output.append("\n11. get_person --person-id PERSON_ID: Get details for a specific person (player, staff, referee, etc).")
            output.append("   - person_id: ID of the person")
            output.append("\n12. get_person_matches --person-id PERSON_ID [--lineup LINEUP] [--e E] [--person-date-from DATE_FROM] [--person-date-to DATE_TO] [--person-competitions COMPETITIONS] [--person-limit LIMIT] [--person-offset OFFSET]: Get matches for a specific person.")
            output.append("   - person_id: ID of the person")
            output.append("   - lineup: Filter by lineup (STARTING, BENCH)")
            output.append("   - e: Event type (GOAL, ASSIST, SUB_IN, SUB_OUT)")
            output.append("   - person_date_from: Start date for matches")
            output.append("   - person_date_to: End date for matches")
            output.append("   - person_competitions: Filter by competitions")
            output.append("   - person_limit: Maximum number of matches to return")
            output.append("   - person_offset: Offset for pagination")
            
            output.append("\n13. get_competition_matches --competition-id COMPETITION_ID [--season SEASON] [--matchday MATCHDAY]: Get matches for a specific competition.")
            output.append("   - competition_id: ID of the competition")
            output.append("   - season: Season year")
            output.append("   - matchday: Matchday number")
            
        else:
            logger.error(f"❌ Error listing parameters: '{endpoint}'")
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

  # Get match history between two teams
  python client.py football-data --team1 81 --team2 65 --fd-limit 10

  # Call multiple methods for betting odds
  python client.py the-odds-api --the-odds-api-method get_sports get_odds get_scores --sport soccer_epl

  # Get team ratings with multiple methods
  python client.py clubelo --clubelo-method get_team_elo --team Barcelona

  # Get performance stats with multiple methods
  python client.py football-data --football-data-method get_team_statistics get_team_matches --team-id 81

  # Get match history with multiple methods
  python client.py football-data --football-data-method get_team_matches get_head_to_head --team1 81 --team2 65 --fd-limit 10

  # Search for teams by name
  python client.py football-data --football-data-method search_teams --team-name "Manchester"

  # Get all teams (without search)
  python client.py football-data --football-data-method search_teams

  # Get all areas (countries/regions)
  python client.py football-data --football-data-method get_areas

  # Get details for a specific area
  python client.py football-data --football-data-method get_area --area-id 2072

  # Get details for a specific person
  python client.py football-data --football-data-method get_person --person-id 16275

  # Get matches for a person
  python client.py football-data --football-data-method get_person_matches --person-id 16275 --person-limit 5

Notes:
- Some endpoints (e.g., get_match_details) may return 403 errors if your API subscription does not include access to restricted resources.
- Use --list-params to see all available CLI arguments for an endpoint.
"""
    )
    
    # Common arguments
    parser.add_argument('endpoints', nargs='+', 
                      choices=['all', 'the-odds-api', 'clubelo', 'football-data'],
                      help='Endpoints to fetch data from (use "all" for all endpoints)')
    parser.add_argument('--list-params', action='store_true',
                      help='List available parameters for the specified endpoint')
    parser.add_argument('--info', action='store_true',
                      help='Show detailed information for the specified endpoint')
    
    # Betting odds specific arguments
    TheOddsApiClient_group = parser.add_argument_group('TheOddsApiClient')
    TheOddsApiClient_group.add_argument('--the-odds-api-method', type=str, nargs='+',
                             choices=['get_odds', 'get_sports', 'get_scores', 'get_events', 'get_historical_odds', 'get_historical_odds_archive'],
                             default=['get_odds'],
                             help=argparse.SUPPRESS)
    TheOddsApiClient_group.add_argument('--sport', type=str,
                             help=argparse.SUPPRESS)
    TheOddsApiClient_group.add_argument('--region', type=str, choices=['eu', 'us', 'uk', 'au'],
                             help=argparse.SUPPRESS)
    TheOddsApiClient_group.add_argument('--markets', type=str,
                             help=argparse.SUPPRESS)
    TheOddsApiClient_group.add_argument('--odds-format', type=str,
                             choices=['decimal', 'american', 'hongkong', 'malay', 'indonesian'],
                             help=argparse.SUPPRESS)
    TheOddsApiClient_group.add_argument('--bookmakers', type=str,
                             help=argparse.SUPPRESS)
    TheOddsApiClient_group.add_argument('--date-from', type=str,
                             help=argparse.SUPPRESS)
    TheOddsApiClient_group.add_argument('--date-to', type=str,
                             help=argparse.SUPPRESS)
    TheOddsApiClient_group.add_argument('--event-ids', type=str,
                             help=argparse.SUPPRESS)
    TheOddsApiClient_group.add_argument('--all-available', action='store_true',
                             help=argparse.SUPPRESS)
    TheOddsApiClient_group.add_argument('--days-from', type=int,
                             help=argparse.SUPPRESS)
    TheOddsApiClient_group.add_argument('--date-format', type=str, choices=['iso', 'unix'],
                             help=argparse.SUPPRESS)
    TheOddsApiClient_group.add_argument('--event-id', type=str,
                             help=argparse.SUPPRESS)
    TheOddsApiClient_group.add_argument('--odds-date', type=str,
                             help=argparse.SUPPRESS)
    
    # Team ratings specific arguments
    ClubEloClient_group = parser.add_argument_group('ClubEloClient')
    ClubEloClient_group.add_argument('--clubelo-method', type=str, nargs='+',
                             choices=['get_team_elo', 'get_top_teams', 'get_fixtures'],
                             default=['get_team_elo'],
                             help=argparse.SUPPRESS)
    ClubEloClient_group.add_argument('--team', type=str,
                             help=argparse.SUPPRESS)
    ClubEloClient_group.add_argument('--elo-date', type=str,
                             help=argparse.SUPPRESS)
    ClubEloClient_group.add_argument('--country', type=str,
                             help=argparse.SUPPRESS)
    ClubEloClient_group.add_argument('--min-elo', type=int,
                             help=argparse.SUPPRESS)
    ClubEloClient_group.add_argument('--limit', type=int,
                             help=argparse.SUPPRESS)
    
    # Performance stats specific arguments
    FootballDataClient_group = parser.add_argument_group('FootballDataClient')
    FootballDataClient_group.add_argument('--football-data-method', type=str, nargs='+',
                           choices=['get_team_statistics', 'get_team_matches', 'get_team_standings', 'get_player_statistics', 'get_head_to_head', 'get_team_info', 'get_competition_matches', 'get_match_details', 'search_teams', 'get_areas', 'get_area', 'get_person', 'get_person_matches'],
                           default=['get_team_statistics'],
                           help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--team-id', type=int, help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--season', type=int,
                           help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--competition-id', type=int,
                           help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--status', type=str, choices=['FINISHED', 'SCHEDULED'],
                           help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--team1', type=int,
                           help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--team2', type=int,
                           help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--fd-limit', type=int,
                           help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--competitions', type=str,
                           help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--team-name', type=str,
                           help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--offset', type=int,
                           help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--area-id', type=int, help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--person-id', type=int, help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--person-name', type=str, help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--person-nationality', type=str, help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--person-position', type=str, help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--lineup', type=str, help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--e', type=str, help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--person-date-from', type=str, help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--person-date-to', type=str, help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--person-competitions', type=str, help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--person-limit', type=int, help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--person-offset', type=int, help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--match-id', type=int, help=argparse.SUPPRESS)
    FootballDataClient_group.add_argument('--matchday', type=int, help=argparse.SUPPRESS)
    
    # Map endpoint to its argument group
    endpoint_groups = {
        'the-odds-api': TheOddsApiClient_group,
        'clubelo': ClubEloClient_group,
        'football-data': FootballDataClient_group,
    }
    
    args = parser.parse_args()
    
    if args.info:
        if len(args.endpoints) != 1:
            print("❌ Please specify exactly one endpoint when using --info")
            return
            
        endpoint = args.endpoints[0]
        if endpoint == 'the-odds-api':
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
- get_odds [--sport SPORT] [--region REGION] [--markets MARKETS] [--odds-format ODDS_FORMAT] [--bookmakers BOOKMAKERS] [--date-from DATE_FROM] [--date-to DATE_TO] [--event-ids EVENT_IDS]: Get odds for upcoming events (default).
- get_sports [--all-available]: Get list of available sports.
- get_scores [--sport SPORT] [--days-from DAYS_FROM] [--date-format DATE_FORMAT]: Get scores for recently completed events.
- get_events [--sport SPORT] [--date-format DATE_FORMAT]: Get list of upcoming events.
- get_historical_odds [--sport SPORT] [--odds-date ODDS_DATE] [--region REGION] [--markets MARKETS] [--date-format DATE_FORMAT] [--odds-format ODDS_FORMAT]: Get historical odds for events at a specific time (defaults to today if no date provided). Note: Requires a paid API plan.
- get_historical_odds_archive --sport SPORT --odds-date ODDS_DATE [--region REGION] [--markets MARKETS] [--date-format DATE_FORMAT] [--odds-format ODDS_FORMAT]: Get historical odds for events at a specific time. Note: Requires a paid API plan.

Example Usage:
1. Get odds for Premier League matches:
   python client.py the-odds-api --sport soccer_epl
2. List available sports:
   python client.py the-odds-api --the-odds-api-method get_sports
3. Get scores for recent matches:
   python client.py the-odds-api --the-odds-api-method get_scores --sport soccer_epl --days-from 3
4. Get historical odds for today's events (requires paid plan):
   python client.py the-odds-api --the-odds-api-method get_historical_odds --sport soccer_epl
5. Get historical odds for a specific date (requires paid plan):
   python client.py the-odds-api --the-odds-api-method get_historical_odds --sport soccer_epl --odds-date 2024-01-01T00:00:00Z
6. Call multiple methods in one run:
   python client.py the-odds-api --the-odds-api-method get_sports get_odds get_scores --sport soccer_epl
7. List available parameters:
   python client.py the-odds-api --list-params
""")
        elif endpoint == 'clubelo':
            print("""
Team Ratings API Help
====================
This endpoint provides access to team strength ratings and Elo scores.

Key Features:
- Get current Elo ratings for teams
- View historical Elo changes
- Access team form and performance metrics
- Search for teams by name

Available Methods:
- get_team_elo [--team TEAM] [--team-id TEAM_ID] [--date DATE]: Get Elo rating for a specific team.
- get_top_teams [--limit LIMIT] [--country COUNTRY] [--min-elo MIN_ELO]: Get top teams by Elo rating.
- get_fixtures: Get fixtures for a team.

Example Usage:
1. Get top teams by Elo rating:
   python client.py clubelo

2. List available parameters:
   python client.py clubelo --list-params

3. Get Elo rating for a specific team:
   python client.py clubelo --team Barcelona --date 2024-01-01
""")
        elif endpoint == 'football-data':
            print("""
Performance Stats API Help
=========================
This endpoint provides access to detailed team and player statistics.

Key Features:
- Get team performance metrics
- View match statistics
- Access league standings
- Track historical performance

Available Methods:
- get_team_statistics --team-id TEAM_ID [--season SEASON] [--competition-id COMPETITION_ID]: Get team statistics for a specific season and competition.
- get_team_matches --team-id TEAM_ID [--season SEASON] [--competition-id COMPETITION_ID] [--status STATUS] [--fd-limit FD_LIMIT] [--competitions COMPETITIONS]: Get team matches for a specific season and competition, or recent matches for a team.
- get_team_standings --team-id TEAM_ID --competition-id COMPETITION_ID [--season SEASON]: Get team standings in a competition.
- get_player_statistics --player-id PLAYER_ID [--season SEASON] [--competition-id COMPETITION_ID]: Get player statistics for a specific season and competition.
- get_head_to_head --team1 TEAM1_ID --team2 TEAM2_ID [--limit LIMIT] [--person-date-from DATE_FROM] [--person-date-to DATE_TO]: Get head-to-head matches between two teams.
- get_team_info --team-id TEAM_ID: Get detailed information about a team.
- get_competition_matches --competition-id COMPETITION_ID [--season SEASON] [--matchday MATCHDAY]: Get matches for a specific competition.
- get_match_details --match-id MATCH_ID: Get detailed information about a specific match.
- search_teams [--team-name TEAM_NAME] [--season SEASON] [--fd-limit FD_LIMIT] [--offset OFFSET]: Search for teams by name or get a list of teams.
- get_areas: Get a list of all areas (countries, regions, etc).
- get_area --area-id AREA_ID: Get details for a specific area by its ID.
- get_person --person-id PERSON_ID: Get details for a specific person (player, staff, referee, etc).
- get_person_matches --person-id PERSON_ID [--lineup LINEUP] [--e E] [--person-date-from DATE_FROM] [--person-date-to DATE_TO] [--person-competitions COMPETITIONS] [--person-limit LIMIT] [--person-offset OFFSET]: Get matches for a specific person.

Example Usage:
1. Get team statistics:
   python client.py football-data --team-id 81

2. List available parameters:
   python client.py football-data --list-params

3. Get stats for a specific team and season:
   python client.py football-data --team-id 81 --season 2023 --competition-id 2021

4. Get match history between two teams:
   python client.py football-data --football-data-method get_head_to_head --team1 81 --team2 65 --fd-limit 10

5. Search for teams by name:
   python client.py football-data --football-data-method search_teams --team-name "Manchester"

6. Get all teams (without search):
   python client.py football-data --football-data-method search_teams

7. Get all areas (countries/regions):
   python client.py football-data --football-data-method get_areas

8. Get details for a specific area:
   python client.py football-data --football-data-method get_area --area-id 2072

9. Get details for a specific person:
   python client.py football-data --football-data-method get_person --person-id 16275

10. Get matches for a person:
   python client.py football-data --football-data-method get_person_matches --person-id 16275 --person-limit 5
""")
            return
    
    if args.list_params:
        if len(args.endpoints) != 1:
            print("❌ Please specify exactly one endpoint when using --list-params")
            return
        endpoint = args.endpoints[0]
        # Print the CLI arguments for the endpoint
        if endpoint in endpoint_groups:
            print(f"\nCLI arguments for endpoint '{endpoint}':\n")
            group = endpoint_groups[endpoint]
            # Manually format the help for the group
            for action in group._group_actions:
                if action.help is not argparse.SUPPRESS:
                    opts = ', '.join(action.option_strings) if action.option_strings else action.dest
                    type_name = type(action.type).__name__ if action.type else 'str'
                    print(f"  {opts} (type: {type_name}" + (f", choices: {action.choices}" if action.choices else "") + (f", default: {action.default}" if action.default and action.default != argparse.SUPPRESS else "") + ")")
                    if action.help:
                        print(f"    {action.help}")
            print("\n---\n")
        else:
            print(f"❌ Unknown endpoint: {endpoint}")
            return
        # Print the custom parameter listing as before
        list_available_parameters(endpoint)
        return
    
    # Create output directory
    output_dir = Path('data') / datetime.now().strftime('%Y%m%d')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which endpoints to fetch
    endpoints = ['the-odds-api', 'clubelo', 'football-data'] if 'all' in args.endpoints else args.endpoints
    
    # Create tasks for each endpoint
    tasks = []
    if 'the-odds-api' in endpoints:
        tasks.append(fetch_the_odds_api(output_dir, args))
    if 'clubelo' in endpoints:
        tasks.append(fetch_clubelo(output_dir, args))
    if 'football-data' in endpoints:
        tasks.append(fetch_football_data(output_dir, args))
    
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
