SITE_URLS = {
    "worldfootball": {
        "url": "https://www.worldfootball.net/",
        "description": "World Football - Football statistics and information",
        "cache_days": {"default": 1, "competition": 1},
        "competition": [
            "/continents/fifa",
            "/continents/uefa",
            "/continents/conmebol",
            "/continents/caf",
            "/continents/caf",
            "/continents/concacaf",
            "/continents/afc",
            "/continents/ofc",
        ],
        "teams": {
            "competition": "/competition/{competition}",
            "historical": "/teams/{team}/{year}/3",
            "news": "/news/{team}/1", 
            "appearances": "/team_performance/{team}/{competition}-{year_prev}-{year}",
            "squad": "/teams/{team}/{year}/2",
            "h2h": "/teams/{team}/11/",
            "h2h-vs": "/teams/{team}/{vs_team}/11/"
        }
    },
    "footystats": {
        "url": "https://footystats.org",
        "description": "FootyStats - Football statistics, analytics, and data",
        "cache_days": {"default": 1, "competition": 1},
        "competition": [
            "/{group}",
        ],
        "teams": {
            "competition": "/{group}/{competition}",
            "stats": "/{group}/{competition}",
            "team-stats": "/clubs/{team}"
        }
    },
    "oddsportal": {
        "url": "https://centroquote.it",
        "description": "CentroQuote - Comparatore quote scommesse sportive",
        "cache_days": {"default": 1, "competition": 1},
        "competition": [
            "/calcio/{competition}",
            "/{sport}/{competition}"
        ],
        "teams": {
            "competition": "/calcio/{competition}",
            "team-stats": "/squadra/{team}"
        }
    }
} 