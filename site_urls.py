SITE_URLS = {
    "worldfootball": {
        "url": "https://www.worldfootball.net/",
        "description": "World Football - Football statistics and information",
        "cache_days": {"default": 1, "competition": 1},
        "paths": {
            "competitions": [
                "/continents/fifa",
                "/continents/uefa",
                "/continents/conmebol",
                "/continents/caf",
                "/continents/caf",
                "/continents/concacaf",
                "/continents/afc",
                "/continents/ofc",
            ],
            "competition-stats": "/competition/{competition}",
            "historical": "/teams/{team}/{year}/3",
            "news": "/news/{team}/1",
            "appearances": "/team_performance/{team}/{competition}-{year_prev}-{year}",
            "squad": "/teams/{team}/{year}/2",
            "h2h": "/teams/{team}/11/",
            "h2h-vs": "/teams/{team}/{vs_team}/11/",
        },
    },
    "footystats": {
        "url": "https://footystats.org",
        "description": "FootyStats - Football statistics, analytics, and data",
        "cache_days": {"default": 1, "competition": 1},
        "paths": {
            "competitions": "/{group}",
            "competition-teams": "/{group}/{competition}",
            "competition-stats": "/{group}/{competition}",
            "team-stats": "/clubs/{team}",
        },
    },
    "oddsportal": {
        "url": "https://centroquote.it",
        "description": "CentroQuote - Comparatore quote scommesse sportive",
        "cache_days": {"default": 1, "competition": 1},
        "paths": {
            "competitions": "/football",
            "odds": "/football/{group}/{competition}",
            "outrights": "/football/{group}/{competition}/outrights/",
            "odds-historical": "/football/{group}/{competition}-{year_prev}-{year}/results/#/page/{page}",
            "competition-stats": "/football/{group}/{competition}-{year_prev}-{year}/standings/",
        },
    },
    "oddschecker": {
        "url": "https://www.oddschecker.com",
        "description": "Oddschecker Italia - Comparatore quote calcio e scommesse sportive",
        "cache_days": {"default": 1, "competition": 1},
        "paths": {
            "competitions": "/it/calcio",
            "odds": "/it/calcio/{group}/{competition}",
            "odds-match": "/it/calcio/{group}/{competition}/{team}-{vs_team}"
        }
    }
}
