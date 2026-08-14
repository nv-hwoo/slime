"""Backend-neutral view of a rollout engine.

``slime/ray/`` drives rollout engines only through the names declared here, so
selecting a backend is the only thing that pulls an inference engine into the
process. Nothing in this module may import one.

This is a minimum, not a lowest common denominator: an engine is free to expose
far more than :class:`RolloutEngine`, and backend-specific code reaches those
extras directly rather than through this seam.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable


class RolloutBackend(str, Enum):
    """Inference engine serving the rollout."""

    SGLANG = "sglang"

    def __str__(self) -> str:
        return self.value


class MemoryTag(str, Enum):
    """A separately resumable region of an engine's GPU memory.

    slime names these regions itself so that ``--offload-rollout`` can stage a
    resume ("weights first, update, then KV cache") without knowing what any
    engine calls them. Engines translate at their own edge.
    """

    WEIGHTS = "weights"
    KV_CACHE = "kv_cache"
    CUDA_GRAPH = "cuda_graph"

    def __str__(self) -> str:
        return self.value


@runtime_checkable
class RolloutEngine(Protocol):
    """What ``ServerGroup``, ``RolloutServer``, ``RolloutHealthMonitor``, and
    ``RolloutManager`` need from an engine actor.

    ``_get_current_node_ip_and_free_port`` is also required, but comes from
    :class:`slime.ray.ray_actor.RayActor` rather than from each backend.
    """

    def init(self, dist_init_addr: str, port: int, nccl_port: int, **kwargs) -> None: ...

    def get_url(self) -> str: ...

    def health_generate(self, timeout: float = ...) -> bool: ...

    def release_memory_occupation(self): ...

    def resume_memory_occupation(self, tags: list[MemoryTag] | None = ...): ...

    def check_weights(self, action: str): ...

    def simulate_crash(self) -> None: ...

    def shutdown(self) -> None: ...


def get_rollout_engine_cls(backend: RolloutBackend | str) -> type[RolloutEngine]:
    """Return the engine class for *backend*, importing it on demand."""
    backend = RolloutBackend(backend)
    if backend is RolloutBackend.SGLANG:
        from slime.backends.sglang_utils.sglang_engine import SGLangEngine

        return SGLangEngine
    raise ValueError(f"No rollout engine registered for backend {backend}")
