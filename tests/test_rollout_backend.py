import ast
from pathlib import Path

import pytest

from slime.backends.rollout_backend import (
    MemoryTag,
    RolloutBackend,
    RolloutEngine,
    get_rollout_engine_cls,
)

pytestmark = pytest.mark.unit

ENGINE_PACKAGES = ("sglang", "sglang_router", "vllm")
RAY_DIR = Path(__file__).resolve().parents[1] / "slime" / "ray"


@pytest.fixture
def sglang_engine_module():
    pytest.importorskip("sglang")
    from slime.backends.sglang_utils import sglang_engine

    return sglang_engine


@pytest.mark.parametrize("member", [*RolloutBackend, *MemoryTag], ids=str)
def test_enum_members_render_as_their_value(member):
    """Callers pass these straight into HTTP payloads and log lines."""
    assert str(member) == member.value
    assert f"{member}" == member.value


@pytest.mark.parametrize("backend", [RolloutBackend.SGLANG, "sglang"], ids=str)
def test_get_rollout_engine_cls_returns_the_sglang_engine(sglang_engine_module, backend):
    assert get_rollout_engine_cls(backend) is sglang_engine_module.SGLangEngine


def test_sglang_engine_satisfies_the_protocol(sglang_engine_module):
    assert issubclass(sglang_engine_module.SGLangEngine, RolloutEngine)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        get_rollout_engine_cls("tensorrt")


@pytest.mark.parametrize(
    "tags",
    [None, [MemoryTag.WEIGHTS], [MemoryTag.KV_CACHE, MemoryTag.CUDA_GRAPH]],
    ids=["all", "weights", "kv_and_graph"],
)
def test_resume_memory_occupation_translates_tags(monkeypatch, sglang_engine_module, tags):
    """slime speaks MemoryTag; the engine is what knows SGLang's own tag names."""
    engine = sglang_engine_module.SGLangEngine.__new__(sglang_engine_module.SGLangEngine)
    calls = []
    monkeypatch.setattr(engine, "_make_request", lambda endpoint, payload=None: calls.append((endpoint, payload)))

    engine.resume_memory_occupation(tags)

    expected = None if tags is None else [sglang_engine_module._SGLANG_MEMORY_TAGS[tag] for tag in tags]
    assert calls == [("resume_memory_occupation", {"tags": expected})]


def _module_level_imports(path: Path) -> set[str]:
    names = set()
    for node in ast.parse(path.read_text(), filename=str(path)).body:
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("path", sorted(RAY_DIR.rglob("*.py")), ids=lambda path: path.name)
def test_slime_ray_does_not_import_an_inference_engine_at_module_level(path):
    """Selecting a backend must be the only thing that loads an engine, so
    ``slime/ray/`` reaches engines through ``get_rollout_engine_cls`` instead."""
    offenders = sorted(name for name in _module_level_imports(path) if name.split(".")[0] in ENGINE_PACKAGES)
    assert not offenders, f"{path.name} imports {offenders} at module level"
