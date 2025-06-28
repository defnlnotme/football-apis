#!/usr/bin/env python3
"""
Simple test for market type extraction logic.
"""

def identify_market_type_from_text(text: str) -> str:
    """Identify market type from text content (simplified version)."""
    # Simplified approach - let the LLM handle market type identification
    # Just extract a meaningful name from the text
    text_lower = text.lower()
    
    # Basic fallback for common patterns, but don't restrict to these
    if any(word in text_lower for word in ['vincente', 'winner', 'win']):
        return 'match_winner'
    elif any(word in text_lower for word in ['totale gol', 'total goals', 'over under']):
        return 'total_goals'
    elif any(word in text_lower for word in ['handicap', 'asiatico']):
        return 'asian_handicap'
    elif any(word in text_lower for word in ['risultato esatto', 'exact score']):
        return 'exact_score'
    elif any(word in text_lower for word in ['primo tempo', 'first half']):
        return 'first_half'
    elif any(word in text_lower for word in ['secondo tempo', 'second half']):
        return 'second_half'
    elif any(word in text_lower for word in ['margine vittoria', 'victory margin']):
        return 'victory_margin'
    elif any(word in text_lower for word in ['both teams', 'entrambe le squadre']):
        return 'both_teams_score'
    else:
        # Extract a meaningful name from the text instead of just "unknown"
        words = text.split()
        if len(words) >= 2:
            # Take first two meaningful words
            meaningful_words = [w for w in words[:3] if len(w) > 2 and w.lower() not in ['the', 'and', 'for', 'with', 'bet', 'odds']]
            if meaningful_words:
                return '_'.join(meaningful_words[:2]).lower()
        return 'market'

def test_market_type_extraction():
    """Test the market type extraction logic with sample text."""
    
    test_cases = [
        "Match Winner - Inter Milan vs Juventus",
        "Total Goals Over/Under 2.5",
        "Asian Handicap -1.5",
        "Exact Score 2-1",
        "First Half Winner",
        "Both Teams to Score",
        "Corner Kicks Over 9.5",
        "Yellow Cards Over 4.5",
        "Player to Score First",
        "Clean Sheet - Yes",
        "Some Random Market We Don't Know About",
        "Double Chance Home or Draw",
        "Half Time/Full Time Result",
        "Corner Kicks First Half",
        "Red Card in Match",
        "Penalty Awarded",
        "Own Goal Scored"
    ]
    
    print("=== MARKET TYPE EXTRACTION TEST ===")
    
    for test_text in test_cases:
        market_type = identify_market_type_from_text(test_text)
        print(f"Text: '{test_text}'")
        print(f"  Extracted Type: {market_type}")
        print("---")

if __name__ == "__main__":
    test_market_type_extraction() 