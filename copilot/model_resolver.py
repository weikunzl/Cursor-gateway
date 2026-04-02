"""
Model name resolution for Copilot Gateway.

Handles model name normalization and alias resolution.
Model names from GitHub Copilot API are passed through mostly as-is.
"""

import re
from typing import Dict, List, Optional

from loguru import logger

from copilot.cache import ModelInfoCache
from copilot.config import MODEL_ALIASES, HIDDEN_FROM_LIST


class ModelResolver:
    """
    Resolves model names for Copilot API.

    Resolution pipeline:
    1. Check aliases
    2. Basic normalization
    3. Cache lookup
    4. Pass-through (let Copilot decide)
    """

    def __init__(
        self,
        cache: ModelInfoCache,
        aliases: Optional[Dict[str, str]] = None,
        hidden_from_list: Optional[List[str]] = None,
    ):
        self._cache = cache
        self._aliases = aliases or MODEL_ALIASES
        self._hidden_from_list = set(hidden_from_list or HIDDEN_FROM_LIST)

    def resolve(self, model_name: str) -> str:
        """
        Resolve a model name to the ID to send to Copilot API.

        Args:
            model_name: Model name from client request

        Returns:
            Resolved model ID for Copilot API
        """
        original = model_name

        # Step 1: Check aliases
        if model_name in self._aliases:
            model_name = self._aliases[model_name]
            logger.debug(f"Model alias resolved: {original} -> {model_name}")

        # Step 2: Basic normalization
        normalized = self._normalize(model_name)
        if normalized != model_name:
            logger.debug(f"Model normalized: {model_name} -> {normalized}")
            model_name = normalized

        # Step 3: Cache lookup (verify it exists)
        if self._cache.is_valid_model(model_name):
            return model_name

        # Step 4: Pass-through - let Copilot decide
        logger.debug(f"Model not in cache, passing through: {model_name}")
        return model_name

    def _normalize(self, name: str) -> str:
        """Basic model name normalization."""
        # Strip date suffixes (e.g., gpt-4o-2024-05-13 -> gpt-4o)
        name = re.sub(r'-\d{8}$', '', name)
        # Strip YYYY-MM-DD suffixes
        name = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', name)
        return name

    def get_available_models(self) -> List[str]:
        """
        Returns list of model IDs available for the /v1/models endpoint.

        Excludes models in hidden_from_list and adds aliases.
        """
        models = []

        for model_id in self._cache.get_all_model_ids():
            if model_id not in self._hidden_from_list:
                models.append(model_id)

        for alias_name in self._aliases:
            if alias_name not in models:
                models.append(alias_name)

        return sorted(models)
