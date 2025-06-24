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
            "competition": "/competition/{competition_id}",
            "historical": "/teams/{team_id}/{year}/3",
            "news": "/news/{team_id}/1", 
            "appearances": "/team_performance/{team_id}/{competition_id}-{year_prev}-{year}",
            "squad": "/teams/{team_id}/{year}/2",
            "h2h": "/teams/{team_id}/11/",
            "h2h-vs": "/teams/{team_id}/{vs_team}/11/"
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
            "competition": "/{group}/{competition_id}",
            "stats": "/{group}/{competition_id}",
            "team-stats": "/clubs/{team_id}"
        }
    }
} 