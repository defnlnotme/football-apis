import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import requests
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseAPIClient(ABC):
    """Base class for all football API clients."""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize the base API client.
        
        Args:
            api_key: API key for authentication (if required)
            base_url: Base URL for the API
        """
        load_dotenv()  # Load environment variables from .env file
        
        self.api_key = api_key or os.getenv(f"{self.__class__.__name__.upper()}_API_KEY")
        self.base_url = base_url.rstrip('/') if base_url else ""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FootballAPIClient/1.0',
            'Accept': 'application/json',
        })
        
        if self.api_key:
            self._add_auth()
    
    def _add_auth(self):
        """Add authentication to the request. Override in child classes if needed."""
        if self.api_key:
            self.session.headers.update({
                'X-Auth-Token': self.api_key,
                'Authorization': f'Bearer {self.api_key}'
            })
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (relative to base_url)
            params: Query parameters
            data: Request body data
            headers: Additional headers
            
        Returns:
            Parsed JSON response
            
        Raises:
            HTTPError: If the request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = headers or {}
        
        logger.info(f"Making {method} request to {url}")
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers={**self.session.headers, **headers}
            )
            response.raise_for_status()
            
            # Handle empty responses
            if not response.text.strip():
                return {}
                
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs):
        """Make a GET request."""
        return self._make_request('GET', endpoint, params=params, **kwargs)
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, **kwargs):
        """Make a POST request."""
        return self._make_request('POST', endpoint, data=data, **kwargs)
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test the API connection. Must be implemented by subclasses."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close the session."""
        self.session.close()


class CachingMixin:
    """Mixin to add caching functionality to API clients."""
    
    def __init__(self, *args, **kwargs):
        self._cache = {}
        self.cache_enabled = kwargs.pop('cache_enabled', True)
        self.cache_ttl = kwargs.pop('cache_ttl', 300)  # 5 minutes default
        super().__init__(*args, **kwargs)
    
    def _get_cache_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Generate a cache key from the endpoint and parameters."""
        params_str = json.dumps(params or {}, sort_keys=True)
        return f"{endpoint}:{params_str}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if a cache entry is still valid."""
        if not cache_entry:
            return False
            
        timestamp = cache_entry.get('timestamp', 0)
        return (datetime.now().timestamp() - timestamp) < self.cache_ttl
    
    def get_cached(self, endpoint: str, params: Optional[Dict] = None, **kwargs):
        """Get a cached response if available and valid."""
        if not self.cache_enabled:
            return self.get(endpoint, params, **kwargs)
            
        cache_key = self._get_cache_key(endpoint, params)
        cached = self._cache.get(cache_key)
        
        if cached and self._is_cache_valid(cached):
            logger.debug(f"Cache hit for {cache_key}")
            return cached['data']
            
        logger.debug(f"Cache miss for {cache_key}")
        data = self.get(endpoint, params, **kwargs)
        
        if data is not None:
            self._cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now().timestamp()
            }
            
        return data
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache = {}


class RateLimitMixin:
    """Mixin to handle rate limiting."""
    
    def __init__(self, *args, **kwargs):
        self.rate_limit_remaining = float('inf')
        self.rate_limit_reset = 0
        super().__init__(*args, **kwargs)
    
    def _update_rate_limits(self, response):
        """Update rate limit information from response headers."""
        headers = response.headers
        
        if 'X-RateLimit-Remaining' in headers:
            self.rate_limit_remaining = int(headers['X-RateLimit-Remaining'])
        if 'X-RateLimit-Reset' in headers:
            self.rate_limit_reset = int(headers['X-RateLimit-Reset'])
    
    def _check_rate_limit(self):
        """Check if we've hit the rate limit."""
        if self.rate_limit_remaining <= 0:
            reset_time = datetime.fromtimestamp(self.rate_limit_reset)
            wait_seconds = (reset_time - datetime.now()).total_seconds()
            if wait_seconds > 0:
                import time
                logger.warning(f"Rate limit reached. Waiting {wait_seconds:.1f} seconds...")
                time.sleep(wait_seconds + 1)  # Add a small buffer
                
    def _make_request(self, *args, **kwargs):
        """Override _make_request to handle rate limiting."""
        self._check_rate_limit()
        response = super()._make_request(*args, **kwargs)
        self._update_rate_limits(response)
        return response
