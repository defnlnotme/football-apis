"""
Rate limiting mixin for API clients.
"""
import time
import threading

class RateLimitMixin:
    """Mixin class to add basic rate limiting to API clients."""
    
    _lock = threading.Lock()
    _last_call = 0.0
    _min_interval = 1.0  # seconds between calls (default: 1 per second)

    def __init__(self, min_interval: float = 1.0, *args, **kwargs):
        self._min_interval = min_interval
        super().__init__(*args, **kwargs)

    def rate_limited(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            wait = self._min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time() 