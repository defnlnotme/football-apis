"""
Caching mixin for API clients.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CachingMixin:
    """Mixin class to add caching functionality to API clients."""
    
    def __init__(self, cache_enabled: bool = True, cache_dir: str = 'cache'):
        """
        Initialize the caching mixin.
        
        Args:
            cache_enabled (bool): Whether caching is enabled
            cache_dir (str): Directory to store cache files
        """
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)
        if cache_enabled:
            self.cache_dir.mkdir(exist_ok=True)
    
    def get_cached(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Get data from cache if available and not expired.
        
        Args:
            endpoint (str): API endpoint
            params (Dict, optional): Query parameters
            
        Returns:
            Optional[Dict]: Cached data if available and not expired, None otherwise
        """
        if not self.cache_enabled:
            return None
            
        cache_file = self._get_cache_file(endpoint, params)
        if not cache_file.exists():
            return None
            
        try:
            # Check if cache is expired (1 hour)
            if datetime.now().timestamp() - cache_file.stat().st_mtime > 3600:
                return None
                
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")
            return None
    
    def save_to_cache(self, endpoint: str, data: Dict, params: Optional[Dict] = None) -> None:
        """
        Save data to cache.
        
        Args:
            endpoint (str): API endpoint
            data (Dict): Data to cache
            params (Dict, optional): Query parameters
        """
        if not self.cache_enabled:
            return
            
        cache_file = self._get_cache_file(endpoint, params)
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def _get_cache_file(self, endpoint: str, params: Optional[Dict] = None) -> Path:
        """
        Get the cache file path for an endpoint and parameters.
        
        Args:
            endpoint (str): API endpoint
            params (Dict, optional): Query parameters
            
        Returns:
            Path: Path to the cache file
        """
        # Create a unique filename based on endpoint and params
        filename = endpoint.replace('/', '_')
        if params:
            # Sort params to ensure consistent filenames
            param_str = '_'.join(f"{k}={v}" for k, v in sorted(params.items()))
            filename = f"{filename}_{param_str}"
        return self.cache_dir / f"{filename}.json" 