from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_modelexpress_proxy_sends_only_exact_target(monkeypatch):
    pytest.importorskip("sglang")
    pytest.importorskip("sglang_router")
    from slime.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    calls = []
    monkeypatch.setattr(
        engine,
        "_make_request",
        lambda endpoint, payload=None: calls.append((endpoint, payload)),
    )

    engine.prepare_weights_from_modelexpress("7")
    engine.update_weights_from_modelexpress("7")

    assert calls == [
        ("prepare_weights_from_modelexpress", {"target_version": "7"}),
        ("update_weights_from_modelexpress", {"target_version": "7"}),
    ]


def test_modelexpress_private_startup_config_reaches_sglang(monkeypatch):
    pytest.importorskip("sglang")
    pytest.importorskip("sglang_router")
    from slime.backends.sglang_utils import sglang_engine

    args = SimpleNamespace(
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        colocate=False,
        debug_rollout_only=False,
        fp16=False,
        hf_checkpoint="/models/model",
        modelexpress_catalog_endpoint="dns:///catalog:50051",
        modelexpress_initial_version="0",
        modelexpress_model_id="policy",
        modelexpress_preparation_cache_dir="/models/mx-cache",
        modelexpress_ready_timeout_seconds=321.0,
        modelexpress_s3_bucket="weights",
        modelexpress_s3_endpoint=None,
        modelexpress_s3_prefix="run/policy",
        num_gpus_per_node=8,
        offload_rollout=False,
        rollout_num_gpus_per_engine=1,
        seed=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        sglang_pp_size=1,
        update_weight_backend="modelexpress",
        use_rollout_routing_replay=False,
    )
    monkeypatch.setattr(sglang_engine, "_to_local_gpu_id", lambda value: value)

    server_args, _ = sglang_engine._compute_server_args(
        args,
        rank=0,
        dist_init_addr="127.0.0.1:1234",
        nccl_port=1235,
        host="127.0.0.1",
        port=30000,
    )

    assert server_args["modelexpress_model_id"] == "policy"
    assert server_args["modelexpress_catalog_endpoint"] == "dns:///catalog:50051"
    assert "modelexpress_delta_s3_bucket" not in server_args
    assert "modelexpress_delta_s3_prefix" not in server_args
    assert server_args["modelexpress_initial_version"] == "0"
    assert server_args["modelexpress_ready_timeout_seconds"] == 321.0
    assert server_args["modelexpress_preparation_cache_dir"] == "/models/mx-cache"
