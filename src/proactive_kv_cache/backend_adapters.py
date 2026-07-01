"""
EXPERIMENTAL backend adapter system for ShadowKV++.

This module defines a clean separation between backend-specific logic
and the core caching engine, with adapters for HuggingFace, FakeBackend,
and a generic fallback. It is NOT currently wired into engines.py — the
engines call backend methods directly.

TODO: If adopted, wire into ShadowKVPlusEngine via the adapter layer
instead of direct backend calls, to enable new backends (e.g., SGLang)
without modifying engine core logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Tuple

from .models import Backend, PrefillResult


class BackendAdapter(ABC):
    """Abstract base class for backend adapters."""

    def __init__(self, backend: Backend):
        self.backend = backend

    @abstractmethod
    def preprocess_tokens(self, tokens: Sequence[int]) -> Tuple[int, ...]:
        """Preprocess tokens before caching or reuse."""
        pass

    @abstractmethod
    def postprocess_kv_cache(self, past_key_values: Any) -> Any:
        """Postprocess KV cache after retrieval from cache."""
        pass

    @abstractmethod
    def estimate_benefit(self, tokens: Sequence[int], past_key_values: Any) -> float:
        """Estimate the benefit of reusing a cached prefix."""
        pass

    @abstractmethod
    def can_reuse_cache(self, past_key_values: Any) -> bool:
        """Check if a cached KV cache can be reused."""
        pass


class HuggingFaceAdapter(BackendAdapter):
    """Adapter for Hugging Face backends."""

    def preprocess_tokens(self, tokens: Sequence[int]) -> Tuple[int, ...]:
        """Hugging Face typically doesn't need special token preprocessing."""
        return tuple(tokens)

    def postprocess_kv_cache(self, past_key_values: Any) -> Any:
        """Ensure KV cache is in the correct format for Hugging Face."""
        if isinstance(past_key_values, dict):
            # Hugging Face expects past_key_values as a dictionary
            return past_key_values
        # For other formats, try to convert
        return {'past_key_values': past_key_values}

    def estimate_benefit(self, tokens: Sequence[int], past_key_values: Any) -> float:
        """Estimate benefit based on token count and cache size."""
        if isinstance(past_key_values, dict):
            cached_length = past_key_values.get('prefix_len', 0)
            return float(cached_length) * 0.8  # Simple heuristic
        return float(len(tokens)) * 0.5

    def can_reuse_cache(self, past_key_values: Any) -> bool:
        """Check if cache can be reused with Hugging Face."""
        if past_key_values is None:
            return False
        if isinstance(past_key_values, dict):
            return 'prefix_len' in past_key_values or 'past_key_values' in past_key_values
        return True


class FakeBackendAdapter(BackendAdapter):
    """Adapter for the FakeBackend used in testing."""

    def preprocess_tokens(self, tokens: Sequence[int]) -> Tuple[int, ...]:
        """Fake backend doesn't need special preprocessing."""
        return tuple(tokens)

    def postprocess_kv_cache(self, past_key_values: Any) -> Any:
        """Ensure KV cache is in FakeBackend format."""
        if isinstance(past_key_values, dict):
            # FakeBackend expects tokens and device
            past_key_values.setdefault('device', self.backend.device)
            past_key_values.setdefault('tokens', ())
        return past_key_values

    def estimate_benefit(self, tokens: Sequence[int], past_key_values: Any) -> float:
        """Simple benefit estimation for testing."""
        if isinstance(past_key_values, dict):
            return float(past_key_values.get('prefix_len', 0))
        return float(len(tokens))

    def can_reuse_cache(self, past_key_values: Any) -> bool:
        """Fake backend can reuse any non-None cache."""
        return past_key_values is not None


class BackendAdapterFactory:
    """Factory for creating backend adapters."""

    @staticmethod
    def create_adapter(backend: Backend) -> BackendAdapter:
        """Create an appropriate adapter for the given backend."""
        backend_name = getattr(backend, 'backend_name', 'unknown')

        if backend_name == 'fake':
            return FakeBackendAdapter(backend)
        elif backend_name == 'huggingface':
            return HuggingFaceAdapter(backend)
        elif 'hugging' in backend_name.lower():
            return HuggingFaceAdapter(backend)
        else:
            # Default adapter for unknown backends
            return GenericBackendAdapter(backend)


class GenericBackendAdapter(BackendAdapter):
    """Generic adapter that works with most backends."""

    def preprocess_tokens(self, tokens: Sequence[int]) -> Tuple[int, ...]:
        return tuple(tokens)

    def postprocess_kv_cache(self, past_key_values: Any) -> Any:
        return past_key_values

    def estimate_benefit(self, tokens: Sequence[int], past_key_values: Any) -> float:
        if hasattr(past_key_values, '__len__'):
            try:
                return float(len(past_key_values))
            except TypeError:
                pass
        return float(len(tokens)) * 0.7

    def can_reuse_cache(self, past_key_values: Any) -> bool:
        return past_key_values is not None