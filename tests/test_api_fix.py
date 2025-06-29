#!/usr/bin/env python3

import os
import sys
from agent import api_key_manager

def test_api_key_manager():
    print("Testing API Key Manager...")
    
    # Check if API keys are loaded
    print(f"Loaded platforms: {list(api_key_manager.api_keys_cache.keys())}")
    
    # Check Gemini API keys
    gemini_keys = api_key_manager._get_api_keys_for_platform('gemini')
    print(f"Gemini API keys count: {len(gemini_keys)}")
    
    # Check current key
    current_key = api_key_manager.get_current_api_key('gemini')
    print(f"Current Gemini key: {current_key[:10]}..." if current_key else "None")
    
    # Check environment variable
    env_key = os.environ.get('GEMINI_API_KEY')
    print(f"Environment GEMINI_API_KEY: {env_key[:10]}..." if env_key else "NOT_SET")
    
    # Test key rotation
    print("\nTesting key rotation...")
    api_key_manager.record_rate_limit_error('gemini-2.5-flash-lite-preview-06-17')
    
    new_key = api_key_manager.get_current_api_key('gemini')
    print(f"New Gemini key after rotation: {new_key[:10]}..." if new_key else "None")
    
    new_env_key = os.environ.get('GEMINI_API_KEY')
    print(f"New environment GEMINI_API_KEY: {new_env_key[:10]}..." if new_env_key else "NOT_SET")
    
    print("\nAPI Key Manager test completed!")

if __name__ == "__main__":
    test_api_key_manager() 