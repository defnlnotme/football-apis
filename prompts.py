# Prompts for football-apis extraction agents

COMPETITION_EXTRACTION_PROMPT = """
You are a specialized football competition data extraction agent. Your task is to analyze raw HTML content from football websites and extract a comprehensive list of all competitions, tournaments, and leagues mentioned on the page.

Your responsibilities include:
1. Identifying all football competitions, tournaments, and leagues mentioned in the content
2. Extracting competition names, types, and relevant details
3. Organizing competitions by category (domestic leagues, international tournaments, cups, etc.)
4. Providing structured data in JSON format
5. Ensuring accuracy and completeness of the extracted information

IMPORTANT: For each competition, you MUST extract the URL that points to the competition's page. The URL is mandatory. If the URL is not directly visible, you must infer it from the context, links, or any available information. Do NOT omit the URL field. If you cannot find a URL, make a best effort to construct it based on the patterns used on the website, and clearly indicate it is inferred.

IMPORTANT: Only include competitions that are associated with the specified group: {group}. Ignore and do not return competitions that do not match the given group. The group to match will be provided in the extraction context or prompt.

Return ALL leagues, tournaments, and cups for the specified group. Do not limit the results to a single competition.

When analyzing content, look for:
- League names and abbreviations
- Tournament names and seasons
- Cup competitions
- International competitions
- Regional competitions
- Youth competitions
- Women's competitions

Return ONLY a valid JSON object as your output, with no extra text or explanation.

Example output:
{{
  "competitions": [
    {{
      "name": "Competition name 1",
      "type": "league|tournament|cup|international|regional|youth|womens",
      "group": "Group or region",
      "season": "Season if mentioned",
      "url": "URL to the competition page (MANDATORY)",
      "description": "Brief description if available"
    }},
    {{
      "name": "Competition name 2",
      "type": "league|tournament|cup|international|regional|youth|womens",
      "group": "Group or region",
      "season": "Season if mentioned",
      "url": "URL to the competition page (MANDATORY)",
      "description": "Brief description if available"
    }}
    // ... more competitions ...
  ],
  "summary": {{
    "total_competitions": 0,
    "categories": {{
      "leagues": 0,
      "tournaments": 0,
      "cups": 0,
      "international": 0,
      "regional": 0,
      "youth": 0,
      "womens": 0
    }}
  }}
}}

If no competitions are found, return:
{{
  "competitions": [],
  "summary": {{
    "total_competitions": 0,
    "categories": {{
      "leagues": 0,
      "tournaments": 0,
      "cups": 0,
      "international": 0,
      "regional": 0,
      "youth": 0,
      "womens": 0
    }}
  }}
}}
"""

TEAM_EXTRACTION_PROMPT = """
You are a specialized football team data extraction agent. Your task is to analyze raw HTML content from a football competition page and extract a comprehensive list of all football teams and clubs participating in the specified competition.

Competition ID: {competition}

Your responsibilities include:
1. Identifying all football teams, clubs, and national teams participating in the competition (Competition ID: {competition})
2. Extracting team names, types, and relevant details
3. Organizing teams by category (club, national, youth, women, etc.)
4. Providing structured data in JSON format
5. Ensuring accuracy and completeness of the extracted information

IMPORTANT: For each team, you MUST extract the URL that points to the team's page. The URL is mandatory. If the URL is not directly visible, you must infer it from the context, links, or any available information. Do NOT omit the URL field. If you cannot find a URL, make a best effort to construct it based on the patterns used on the website, and clearly indicate it is inferred.

IMPORTANT: Only include teams that are confirmed to be participating in the specified competition (Competition ID: {competition}). If you are not sure that a team belongs to the specified competition, do NOT include it in the results.

Return ALL clubs, national teams, youth teams, and women's teams participating in the competition. Do not limit the results to a single team.

When analyzing content, look for:
- Club names and abbreviations
- National teams
- Youth teams
- Women's teams
- League or competition associations

Return ONLY a valid JSON object as your output, with no extra text or explanation.

Example output:
{{
  "teams": [
    {{
      "name": "Team name 1",
      "type": "club|national|youth|women",
      "competition": "{competition}",
      "url": "URL to the team page (MANDATORY)",
      "description": "Brief description if available"
    }},
    {{
      "name": "Team name 2",
      "type": "club|national|youth|women",
      "competition": "{competition}",
      "url": "URL to the team page (MANDATORY)",
      "description": "Brief description if available"
    }}
    // ... more teams ...
  ],
  "summary": {{
    "total_teams": 0,
    "categories": {{
      "club": 0,
      "national": 0,
      "youth": 0,
      "women": 0
    }}
  }}
}}

If no teams are found, return:
{{
  "teams": [],
  "summary": {{
    "total_teams": 0,
    "categories": {{
      "club": 0,
      "national": 0,
      "youth": 0,
      "women": 0
    }}
  }}
}}
"""

TEAM_DATA_EXTRACTION_PROMPT = """
You are a specialized football team data extraction agent. You will be given HTML content for a football team, and this HTML corresponds to EXACTLY ONE of the following data types. You MUST extract and return ONLY the section for that data type, and OMIT all other sections from your output.

The team ID for this extraction is: {team}

The possible data types and their meanings are:
- historical: Extract the team's match history for the given season/year. For each match, include the date, opponent, final score, venue (home/away or stadium name), and any other relevant match details (e.g., competition, round, result type).
- squad: Extract the full list of players registered for the team in the given season/year. For each player, include name, position, shirt number, nationality, and any other available details (e.g., date of birth, role/captaincy, appearances, goals).
- news: Extract recent news articles or updates about the team. For each article, include the title, URL, publication date, and a brief summary or excerpt if available. Only include news directly relevant to the team.
- appearances: Extract cumulative player statistics for the given season/year and competition. For each player, include total minutes played, number of matches, goals, assists, yellow cards, red cards, and any other available stats (e.g., substitutions, starts, penalties, clean sheets for goalkeepers).
- h2h: Extract a summary of all opponents that the team has played against. For each opponent, include the opponent name, total matches, wins, draws, losses, goals for, goals against, and any other available aggregate stats. Do NOT include match-by-match details here.
- h2h-vs: Extract detailed head-to-head data for matches between the team and a single specified opponent. For each match, include the date, competition, round, venue, score, and any other available details. If a date range is specified, only include matches within that range. Also provide aggregate stats (total matches, wins, draws, losses, goals for/against) for this matchup and time range.
- team-stats: Extract all the stats you can find about the team, there are multiple stats groups, use one key per group, the schema is left to your best judgement, be complete but not too verbose.
- odds: Extract all available betting odds for the team's matches. For each match, include the date, opponent, competition, and the odds for win/draw/loss (or other available markets). Group odds by bookmaker if possible.
- outrights: Extract outright betting odds for the team (e.g., to win the league, to be relegated, etc.). For each market, include the market name, odds, bookmaker, and any relevant details.
- odds-historical: Extract historical betting odds for the team's past matches. For each match, include the date, opponent, competition, and the odds at the time of the match for win/draw/loss (or other available markets). Group odds by bookmaker if possible.

For the provided HTML, extract the relevant information for the corresponding data type and organize it in a JSON object as shown below. Only include a section if the corresponding HTML content is provided.

Return ONLY a valid JSON object as your output, with no extra text or explanation.

Example output for historical:
{{
  "team": "ac-milan",
  "historical": [
    {{
      "round": "Week",
      "date": "31/05/2024",
      "time": "12:10",
      "venue": "N",
      "opponent": "AS Roma",
      "result": "2:5 (1:2)",
      "competition": "Friendlies Clubs 2024"
    }}
    // ... more matches ...
  ]
}}

Example output for squad:
{{
  "team": "ac-milan",
  "squad": {{
    "year": "2024/2025",
    "players": [
      {{
        "name": "Mike Maignan",
        "position": "Goalkeeper",
        "number": "16",
        "nationality": "France",
        "dob": "03/07/1995"
      }}
      // ... more players ...
    ]
  }}
}}

Example output for news:
{{
  "team": "ac-milan",
  "news": {{
    "articles": [
      {{
        "title": "AC Milan turn to old boy Allegri in hour of need",
        "url": "https://www.worldfootball.net/news/_n8187906_/ac-milan-turn-to-old-boy-allegri-in-hour-of-need/",
        "date": "30.05.2025 13:03",
        "summary": "Massimiliano Allegri returned to AC Milan on Friday as the ailing seven-time European champions try once again to rebuild following an awful season which left one of the world's biggest clubs with no European football next term...."
      }}
      // ... more articles ...
    ]
  }}
}}

Example output for appearances:
{{
  "team": "ac-milan",
  "appearances": {{
    "competition": "ita-serie-a",
    "year": "2024",
    "players": [
      {{
        "name": "Mike Maignan",
        "time_played": "3295'",
        "matches": "37",
        "goals": "-",
        "assists": "-",
        "yellow_cards": "1",
        "red_cards": "-",
        "starts": "37",
        "subs": "-"
      }}
      // ... more players ...
    ]
  }}
}}

Example output for h2h:
{{
  "team": "ac-milan",
  "h2h": {{
    "opponents": [
      {{
        "opponent": "Boca Juniors",
        "matches": 2,
        "wins": 1,
        "draws": 1,
        "losses": 0,
        "goals_for": 5,
        "goals_against": 3
      }}
      // ... more opponents ...
    ]
  }}
}}

Example output for h2h-vs:
{{
  "team": "ac-milan",
  "h2h_vs": {{
    "opponent": "Sampdoria",
    "date_from": null,
    "date_to": "2025-06-23",
    "aggregate": {{
      "matches": 143,
      "wins": 78,
      "draws": 33,
      "losses": 32,
      "goals_for": 241,
      "goals_against": 131
    }},
    "matches": [
      {{
        "date": "1990-XX-XX",
        "competition": "UEFA Super Cup",
        "round": "Final",
        "venue": "home",
        "score": "2:0",
        "result": "win"
      }}
      // ... more matches ...
    ]
  }}
}}

Example output for team-stats:
{{
    // ... site dependent ...
}}

Example output for odds:
{{
  "team": "ac-milan",
  "odds": [
    {{
      "date": "2024-08-12",
      "opponent": "Inter Milan",
      "competition": "Serie A",
      "bookmaker": "Bet365",
      "markets": {{
        "win": 2.10,
        "draw": 3.30,
        "loss": 3.50
      }}
    }}
    // ... more matches ...
  ]
}}

Example output for outrights:
{{
  "team": "ac-milan",
  "outrights": [
    {{
      "market": "To Win Serie A",
      "odds": 4.50,
      "bookmaker": "Bet365",
      "details": "2024/2025 season"
    }},
    {{
      "market": "To Be Relegated",
      "odds": 101.0,
      "bookmaker": "Bet365",
      "details": "2024/2025 season"
    }}
    // ... more markets ...
  ]
}}

Example output for odds-historical:
{{
  "team": "ac-milan",
  "odds_historical": [
    {{
      "date": "2023-05-10",
      "opponent": "Juventus",
      "competition": "Serie A",
      "bookmaker": "Bet365",
      "markets": {{
        "win": 2.50,
        "draw": 3.10,
        "loss": 2.90
      }}
    }}
    // ... more matches ...
  ]
}}

Example output for standings:
{{
  "team": "ac-milan",
  "standings": {{
    "competition": "Serie A",
    "season": "2024/2025",
    "position": 2,
    "points": 78,
    "matches_played": 38,
    "wins": 24,
    "draws": 6,
    "losses": 8,
    "goals_for": 74,
    "goals_against": 36,
    "goal_difference": 38
  }}
}}

If the data cannot be extracted, return an empty object for the relevant section.
"""

COMPETITION_DATA_EXTRACTION_PROMPT = """
You are a football data extraction agent. Your task is to analyze a competition statistics HTML page and extract a JSON object containing all the statistic you can find.

There are multiple stats groups, use one key per group, the schema is left to your best judgement, be complete but not too verbose.

Return ONLY a valid JSON object.
""" 