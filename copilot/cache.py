"""
Model metadata cache for Copilot Gateway.

Thread-safe storage for available model information
with TTL and lazy loading support.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from copilot.config import MODEL_CACHE_TTL, DEFAULT_MAX_INPUT_TOKENS


class ModelInfoCache:
    """
    Thread-safe cache for storing model metadata.

    Uses Lazy Loading for population - data is loaded
    only on first access or when cache is stale.

    Attributes:
        cache_ttl: Cache time-to-live in seconds

    Example:
        >>> cache = ModelInfoCache()
        >>> await cache.update([{"modelId": "gpt-4o", "tokenLimits": {...}}])
        >>> info = cache.get("gpt-4o")
        >>> max_tokens = cache.get_max_input_tokens("gpt-4o")
    """

    def __init__(self, cache_ttl: int = MODEL_CACHE_TTL):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._last_update: Optional[float] = None
        self._cache_ttl = cache_ttl

    async def update(self, models_data: List[Dict[str, Any]]) -> None:
        """
        Updates the model cache.

        Args:
            models_data: List of dicts with model information.
                         Each dict must contain the "modelId" key.
        """
        async with self._lock:
            logger.info(f"Updating model cache. Found {len(models_data)} models.")
            self._cache = {model["modelId"]: model for model in models_data}
            self._last_update = time.time()

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Returns model information or None if not found."""
        return self._cache.get(model_id)

    def is_valid_model(self, model_id: str) -> bool:
        """Check if model exists in dynamic cache."""
        return model_id in self._cache

    def get_max_input_tokens(self, model_id: str) -> int:
        """Returns maxInputTokens for the model."""
        model = self._cache.get(model_id)
        if model and model.get("tokenLimits"):
            return model["tokenLimits"].get("maxInputTokens") or DEFAULT_MAX_INPUT_TOKENS
        return DEFAULT_MAX_INPUT_TOKENS

    def is_empty(self) -> bool:
        """Checks if the cache is empty."""
        return not self._cache

    def is_stale(self) -> bool:
        """Checks if the cache is stale."""
        if not self._last_update:
            return True
        return time.time() - self._last_update > self._cache_ttl

    def get_all_model_ids(self) -> List[str]:
        """Returns a list of all model IDs in the cache."""
        return list(self._cache.keys())

    @property
    def size(self) -> int:
        """Number of models in the cache."""
        return len(self._cache)

    @property
    def last_update_time(self) -> Optional[float]:
        """Last update time (timestamp) or None."""
        return self._last_update
