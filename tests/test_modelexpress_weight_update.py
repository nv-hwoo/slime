import sys
import threading
import types
from argparse import Namespace

import pytest
import torch

from slime.backends.megatron_utils.update_weight import update_weight_from_modelexpress as mx_module
from slime.backends.megatron_utils.update_weight import update_weight_from_distributed as distributed_module
from slime.backends.megatron_utils.update_weight.update_weight_from_modelexpress import UpdateWeightFromModelExpress

pytestmark = pytest.mark.unit


class RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeVersion:
    def __init__(self, version_id, state="READY", model_name="policy"):
        self.version_id = version_id
        self.state = state
        self.model_name = model_name
        self.ref = types.SimpleNamespace(version_id=version_id)


class FakeControlClient:
    def __init__(self):
        self.versions = {"base-uid": FakeVersion("base-uid")}
        self.creates = []
        self.deletes = []

    def get_weight_version(self, version_id):
        return self.versions[version_id]

    def create_weight_version(self, **kwargs):
        self.creates.append(kwargs)
        version = FakeVersion(f"target-{kwargs['version_number']}", state="STAGING")
        self.versions[version.version_id] = version
        return version

    def delete_weight_version(self, version_id):
        self.deletes.append(version_id)


class FakeStagedShard:
    def __init__(self, control, version_id):
        self.control = control
        self.version_id = version_id
        self.publish_count = 0

    def publish(self):
        self.publish_count += 1
        self.control.versions[self.version_id] = FakeVersion(self.version_id)


class FakeTrainerClient:
    def __init__(self, control):
        self.control = control
        self.stages = []
        self.releases = []
        self.base_preparations = []
        self.metrics = {
            "changed_bytes": 25,
            "total_bytes": 100,
            "wire_bytes": 123,
            "stage_delta_time": 7.0,
            "publish_s3_time": 8.0,
            "publish_server_time": 9.0,
        }

    def stage_shard(self, *, version, hf_tensor_iter):
        staged = FakeStagedShard(self.control, version.version_id)
        self.stages.append((version, hf_tensor_iter, threading.current_thread().name, staged))
        return staged

    def prepare_delta_base(self, *, hf_tensor_iter):
        self.base_preparations.append(list(hf_tensor_iter))

    def pop_metrics(self):
        return dict(self.metrics)

    def release_version(self, *, version):
        self.releases.append(version)


class FakeEngine:
    def __init__(self, events):
        self.version = "base-uid"
        self.events = events
        self.prepare_weights_from_modelexpress = RemoteMethod(self._prepare)
        self.pause_generation = RemoteMethod(self._pause)
        self.flush_cache = RemoteMethod(self._flush)
        self.update_weights_from_modelexpress = RemoteMethod(self._install)
        self.get_modelexpress_status = RemoteMethod(self._status)
        self.continue_generation = RemoteMethod(self._continue)

    def _event(self, name):
        self.events.append((name, threading.current_thread().name))

    def _prepare(self, target):
        self._event(f"prepare:{target}")
        return {
            "success": True,
            "metrics": {"perf/mx_receive_prepare_time": 2.0},
        }

    def _pause(self):
        self._event("pause")

    def _flush(self):
        self._event("flush")

    def _install(self, target):
        self._event(f"install:{target}")
        self.version = target
        return {
            "success": True,
            "installed_version": self.version,
            "detail": "",
            "metrics": {"perf/mx_receive_install_time": 3.0},
        }

    def _status(self):
        return {
            "success": True,
            "installed_version": self.version,
            "state": "VERIFIED",
        }

    def _continue(self):
        self._event("continue")


def args():
    return Namespace(
        hf_checkpoint="/models/model",
        update_weight_buffer_size=512 * 1024**2,
        modelexpress_initial_version="0",
        modelexpress_model_id="policy",
        modelexpress_base_version_id="base-uid",
    )


@pytest.fixture(autouse=True)
def patch_runtime(monkeypatch):
    class WeightVersionRef:
        def __init__(self, version_id):
            self.version_id = version_id

    modelexpress_rl = types.ModuleType("modelexpress_rl")
    modelexpress_rl.WeightPayloadFormat = types.SimpleNamespace(XOR_DELTA="XOR_DELTA")
    modelexpress_rl.WeightVersionRef = WeightVersionRef
    monkeypatch.setitem(sys.modules, "modelexpress_rl", modelexpress_rl)
    monkeypatch.setattr(mx_module.ray, "get", lambda refs: refs)
    monkeypatch.setattr(mx_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(mx_module.dist, "barrier", lambda group=None: None)
    monkeypatch.setattr(
        mx_module.dist,
        "all_reduce",
        lambda value, op=None, group=None: None,
    )
    monkeypatch.setattr(
        mx_module.dist,
        "broadcast_object_list",
        lambda values, src, group=None: None,
    )
    monkeypatch.setattr(mx_module.mpu, "get_data_parallel_rank", lambda **kwargs: 0)
    monkeypatch.setattr(mx_module.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(mx_module, "get_gloo_group", lambda: object())


def updater(control=None, trainer=None, stub_gather=True):
    control = control or FakeControlClient()
    trainer = trainer or FakeTrainerClient(control)
    instance = UpdateWeightFromModelExpress(
        args(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3",
        quantization_config=None,
        control_client=control,
        trainer_client=trainer,
    )
    if stub_gather:
        instance._iter_hf_buckets = lambda: iter(())
    return instance, control, trainer


@pytest.mark.parametrize(
    ("override", "expected"),
    [(None, 123), ("1024", 1024)],
)
def test_model_express_bucket_size_prefers_explicit_env(
    monkeypatch, override, expected
):
    config = args()
    config.update_weight_buffer_size = 123
    if override is None:
        monkeypatch.delenv("MX_REFIT_DELTA_BUCKET_BYTES", raising=False)
    else:
        monkeypatch.setenv("MX_REFIT_DELTA_BUCKET_BYTES", override)
    control = FakeControlClient()
    instance = UpdateWeightFromModelExpress(
        config,
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3",
        quantization_config=None,
        control_client=control,
        trainer_client=FakeTrainerClient(control),
    )
    instance._activation_executor.shutdown()

    assert instance.args.update_weight_buffer_size == expected


def test_model_express_bucket_size_rejects_nonpositive_env(monkeypatch):
    monkeypatch.setenv("MX_REFIT_DELTA_BUCKET_BYTES", "0")

    with pytest.raises(ValueError, match="MX_REFIT_DELTA_BUCKET_BYTES must be positive"):
        updater()


def test_receive_metrics_merge_by_max_latency():
    assert mx_module._receiver_metrics(
        [
            {"metrics": {"perf/mx_receive_prepare_time": 2.0}},
            {"metrics": {"perf/mx_receive_prepare_time": 3.0}},
        ]
    ) == {"perf/mx_receive_prepare_time": 3.0}


def test_trainer_config_owns_the_process_group(monkeypatch):
    control = FakeControlClient()
    trainer = FakeTrainerClient(control)
    captured = {}
    gloo_group = object()

    class Config(types.SimpleNamespace):
        pass

    class ControlClient:
        @staticmethod
        def connect(*, server_url):
            return control

    class TrainerClient:
        @staticmethod
        def initialize(config):
            captured["config"] = config
            return trainer

    modelexpress_rl = types.ModuleType("modelexpress_rl")
    modelexpress_rl.ModelExpressControlClient = ControlClient
    modelexpress_rl.ModelExpressTrainerClient = TrainerClient
    modelexpress_rl.ModelExpressTrainerConfig = Config
    modelexpress_rl.S3Config = Config
    modelexpress_rl.TrainerStagingMode = types.SimpleNamespace(WRITE_TO_STORAGE="WRITE_TO_STORAGE")
    modelexpress_rl.WeightPayloadFormat = types.SimpleNamespace(XOR_DELTA="XOR_DELTA")
    monkeypatch.setitem(sys.modules, "modelexpress_rl", modelexpress_rl)
    monkeypatch.setattr(mx_module, "get_gloo_group", lambda: gloo_group)
    config_args = args()
    config_args.modelexpress_server_url = "dns:///modelexpress:8001"
    config_args.modelexpress_s3_bucket = "weights"
    config_args.modelexpress_s3_prefix = ""
    config_args.modelexpress_s3_endpoint = None
    config_args.update_weight_buffer_size = 1024

    instance = UpdateWeightFromModelExpress(
        config_args,
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3",
        quantization_config=None,
    )
    instance._activation_executor.shutdown()

    assert captured["config"].process_group is gloo_group
    assert not hasattr(captured["config"].s3, "process_group")


def test_slime_publishes_on_main_thread_and_activates_on_control_thread(monkeypatch):
    instance, control, trainer = updater()
    events = []
    instance.connect_rollout_engines([FakeEngine(events)], object())

    instance.update_weights()
    assert trainer.base_preparations == [[]]
    assert instance._delta_base_prepared is True
    assert control.creates == []
    assert trainer.stages == []
    assert events == []
    assert instance.weight_version == 0

    clock = iter(
        [0.0, 1.0, 10.0, 12.0, 20.0, 23.0, 30.0, 34.0, 40.0, 45.0]
    )
    reductions = []
    monkeypatch.setattr(mx_module, "perf_counter", lambda: next(clock))

    def reduce_phases(value, op=None, group=None):
        reductions.append((value.tolist(), op))
        if value.dtype == torch.int64:
            value.copy_(torch.tensor([50, 200, 246], dtype=value.dtype))
        else:
            value.copy_(
                torch.tensor(
                    [17.0, 18.0, 19.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                    dtype=value.dtype,
                )
            )

    monkeypatch.setattr(
        mx_module.dist,
        "all_reduce",
        reduce_phases,
    )
    instance.update_weights()
    instance._activation_executor.shutdown()

    main = threading.current_thread().name
    assert control.creates == [
        {
            "model_name": "policy",
            "idempotency_key": "slime:policy:base-uid:1",
            "version_number": 1,
            "payload_format": "XOR_DELTA",
            "base_version_id": "base-uid",
            "expected_source_slots": ["canonical.delta.root"],
        }
    ]
    assert trainer.stages[0][0].version_id == "target-1"
    assert iter(trainer.stages[0][1]) is trainer.stages[0][1]
    assert trainer.stages[0][2] == main
    assert trainer.stages[0][3].publish_count == 1
    assert control.deletes == []
    assert trainer.releases == []
    assert all(thread.startswith("modelexpress-control") for _event, thread in events)
    assert instance.pop_metrics() == {
        "perf/update_weights_density": 0.25,
        "perf/update_weights_wire_bytes": 246,
        "perf/mx_stage_delta_time": 17.0,
        "perf/mx_publish_s3_time": 18.0,
        "perf/mx_publish_server": 19.0,
        "perf/mx_receive_prepare_time": 2.0,
        "perf/mx_receive_install_time": 3.0,
        "perf/mx_control_create_weight_version": 11.0,
        "perf/mx_stage_shard": 12.0,
        "perf/mx_publish_shard": 13.0,
        "perf/mx_control_get_weight_version_ready": 14.0,
        "perf/mx_update_activate_time": 15.0,
    }
    assert reductions == [
        ([25, 100, 123], torch.distributed.ReduceOp.SUM),
        ([7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0], torch.distributed.ReduceOp.MAX),
    ]
    assert instance.weight_version == 1


def test_reconnect_validates_current_ready_base():
    instance, _control, _trainer = updater()
    engine = FakeEngine([])

    instance.connect_rollout_engines([engine], object())
    instance.connect_rollout_engines([engine], object())
    instance._activation_executor.shutdown()


def test_next_target_keeps_previous_version():
    instance, control, trainer = updater()
    events = []
    engine = FakeEngine(events)
    instance.connect_rollout_engines([engine], object())
    instance.update_weights()
    instance.update_weights()
    instance.update_weights()
    instance._activation_executor.shutdown()

    assert len(control.creates) == 2
    assert len(trainer.stages) == 2
    assert [event for event, _thread in events].count("install:target-1") == 1
    assert [event for event, _thread in events].count("install:target-2") == 1
    assert control.deletes == []
    assert trainer.releases == []
    assert instance._base_version_id == "target-2"
    assert instance.weight_version == 2


def test_receive_metrics_are_broadcast_to_the_logging_rank(monkeypatch):
    instance, _control, _trainer = updater()
    instance.connect_rollout_engines([FakeEngine([])], object())
    instance.update_weights()
    monkeypatch.setattr(mx_module.dist, "get_rank", lambda: 1)
    broadcasts = iter(
        [
            FakeVersion("target-1", state="STAGING"),
            {"perf/mx_receive_prepare_time": 7.0},
        ]
    )

    def broadcast(values, src, group=None):
        values[0] = next(broadcasts)

    monkeypatch.setattr(mx_module.dist, "broadcast_object_list", broadcast)

    instance.update_weights()
    assert instance._activation_executor is not None
    instance._activation_executor.shutdown()

    assert instance.pop_metrics()["perf/mx_receive_prepare_time"] == 7.0


def test_slime_hf_bucket_stream_preserves_framework_buckets(monkeypatch):
    instance, _control, _trainer = updater(stub_gather=False)
    buckets = [[("a", object())], [("b", object()), ("c", object())]]
    events = []

    def non_experts():
        yield buckets[0]
        events.append("nonexperts-finished")

    def experts():
        events.append("experts-started")
        yield buckets[1]
        events.append("experts-finished")

    monkeypatch.setattr(instance, "_iter_non_expert_chunks", non_experts)
    monkeypatch.setattr(instance, "_iter_expert_chunks", experts)
    monkeypatch.setattr(mx_module.dist, "barrier", lambda group=None: events.append("barrier"))

    assert list(instance._iter_hf_buckets()) == buckets
    assert events == [
        "nonexperts-finished",
        "barrier",
        "experts-started",
        "experts-finished",
        "barrier",
    ]


def test_model_express_retains_expert_ep_gather_batching(monkeypatch):
    instance, _control, _trainer = updater(stub_gather=False)
    instance.args.update_weight_buffer_size = 1024
    experts = [
        ("layer.experts.a", torch.tensor([1.0])),
        ("layer.experts.b", torch.tensor([2.0])),
    ]
    gathered = []
    monkeypatch.setattr(
        distributed_module,
        "named_params_and_buffers",
        lambda *_args: iter(experts),
    )
    monkeypatch.setattr(
        distributed_module,
        "all_gather_param",
        lambda _name, param: param,
    )
    monkeypatch.setattr(
        distributed_module.mpu,
        "get_expert_model_parallel_world_size",
        lambda: 1,
    )

    def gather_and_convert(batch):
        gathered.append(tuple(name for name, _param in batch))
        return [(f"hf.{name}", param) for name, param in batch]

    monkeypatch.setattr(instance, "_ep_gather_and_convert", gather_and_convert)

    assert list(instance._iter_expert_chunks()) == [
        [(f"hf.{name}", param) for name, param in experts]
    ]
    assert gathered == [("layer.experts.a", "layer.experts.b")]
