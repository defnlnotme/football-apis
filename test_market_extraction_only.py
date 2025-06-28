#!/usr/bin/env python3
"""
Simple test for market type extraction logic only.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import create_scraping_agent

def test_market_type_extraction():
    """Test the market type extraction logic with sample text."""
    
    # Create a test instance to test the methods
    agent = create_scraping_agent(headless=True)
    
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
        market_type = agent._identify_market_type_from_text(test_text)
        print(f"Text: '{test_text}'")
        print(f"  Extracted Type: {market_type}")
        print("---")

if __name__ == "__main__":
    test_market_type_extraction() 