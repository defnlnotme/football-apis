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
    "transfermarkt": {
        "url": "https://www.transfermarkt.com/",
        "description": "Transfermarkt - Football transfer market and player statistics",
        "cache_days": {"default": 1, "competition": 1},
        "competition": [
            "/"
        ]
    },
    "espn": {
        "url": "https://www.espn.com/soccer/",
        "description": "ESPN Soccer - Latest football news and scores",
        "cache_days": {"default": 1, "competition": 1},
        "competition": [
            "/"
        ]
    },
    "bbc_sport": {
        "url": "https://www.bbc.com/sport/football",
        "description": "BBC Sport Football - UK football news and live scores",
        "cache_days": {"default": 1, "competition": 1},
        "competition": [
            "/"
        ]
    },
    "goal": {
        "url": "https://www.goal.com/",
        "description": "Goal.com - International football news and live scores",
        "cache_days": {"default": 1, "competition": 1},
        "competition": [
            "/"
        ]
    },
    "skysports": {
        "url": "https://www.skysports.com/football",
        "description": "Sky Sports Football - Premier League and football coverage",
        "cache_days": {"default": 1, "competition": 1},
        "competition": [
            "/"
        ]
    },
    "fifa": {
        "url": "https://www.fifa.com/",
        "description": "FIFA - Official website of international football",
        "cache_days": {"default": 1, "competition": 1},
        "competition": [
            "/"
        ]
    },
    "uefa": {
        "url": "https://www.uefa.com/",
        "description": "UEFA - European football governing body",
        "cache_days": {"default": 1, "competition": 1},
        "competition": [
            "/"
        ]
    }
} 