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
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the base API client.
        
        Args:
            api_key: API key for authentication (if required)
            base_url: Base URL for the API
        """
        load_dotenv()  # Load environment variables from .env file
        
        self.base_url = base_url.rstrip('/') if base_url else ""
        self.api_key = api_key # Assign api_key directly

        # If api_key is still None after direct assignment, try loading from environment
        if self.api_key is None:
            env_api_key = os.getenv(f"{self.__class__.__name__.upper()}_API_KEY")
            if env_api_key:
                self.api_key = env_api_key

        logger.info(f"[DEBUG] BaseAPIClient initialized with base_url: {self.base_url}")
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
    
    def parse_response(self, response: requests.Response) -> Union[Dict[str, Any], List[Dict[str, Any]], str]:
        """
        Parses the response from the API.
        By default, tries to parse as JSON. Subclasses can override for different formats.
        """
        if not response.text.strip():
            return {}
        try:
            return response.json()
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON response for {response.url}. Returning raw text.")
            return response.text
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], str]:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (relative to base_url)
            params: Query parameters
            data: Request body data
            headers: Additional headers
            
        Returns:
            Parsed JSON response or raw text if JSON decoding fails
            
        Raises:
            HTTPError: If the request fails
        """
        if not self.base_url:
            raise ValueError("Base URL is required for making API requests")
            
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
            
            # Handle rate limiting before raising any errors
            if response.status_code == 429:
                self._handle_response(response)
                # Retry the request
                return self._make_request(method, endpoint, params, data, headers)
            
            # For non-rate-limit responses, handle normally
            self._handle_response(response)
            response.raise_for_status()
            
            return self.parse_response(response)
            
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                # Handle rate limit error
                self._handle_response(e.response)
                # Retry the request
                return self._make_request(method, endpoint, params, data, headers)
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise
    
    def _handle_response(self, response):
        """Hook for subclasses to handle the raw response (e.g., for rate limits)."""
        pass
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs):
        """Make a GET request."""
        logger.info(f"[DEBUG] BaseAPIClient.get called with endpoint: {endpoint}")
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


class CachedAPIClient(BaseAPIClient):
    """API client with caching functionality."""
    
    def __init__(self, *args, cache_enabled: bool = True, cache_ttl: int = 300, **kwargs):
        """
        Initialize the cached API client.
        
        Args:
            cache_enabled: Whether to enable response caching
            cache_ttl: Time-to-live for cache entries in seconds (default: 5 minutes)
        """
        super().__init__(*args, **kwargs)
        self._cache = {}
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
    
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


class RateLimitedAPIClient(CachedAPIClient):
    """API client with rate limiting functionality."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limit_remaining = float('inf')
        self.rate_limit_reset = 0
    
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
    
    def _handle_response(self, response):
        """Handle the response, including rate limiting."""
        self._update_rate_limits(response)
        
        # Handle rate limit response
        if response.status_code == 429:
            reset_time = int(response.headers.get('x-ratelimit-requests-reset', 0))
            if reset_time > 0:
                wait_seconds = max(0, reset_time - datetime.now().timestamp())
                if wait_seconds > 0:
                    import time
                    logger.warning(f"Rate limit reached. Waiting {wait_seconds:.1f} seconds...")
                    time.sleep(wait_seconds + 1)  # Add a small buffer
