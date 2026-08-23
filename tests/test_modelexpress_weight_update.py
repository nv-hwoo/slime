import sys
import threading
import types
from argparse import Namespace

import pytest
import torch

from slime.backends.megatron_utils.update_weight import update_weight_from_modelexpress as mx_module
from slime.backends.megatron_utils.update_weight.update_weight_from_modelexpress import (
    ModelExpressUpdateError,
    UpdateWeightFromModelExpress,
)

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
        self.fail_next_delete = False

    def get_weight_version(self, version_id):
        return self.versions[version_id]

    def create_weight_version(self, **kwargs):
        self.creates.append(kwargs)
        version = FakeVersion(f"target-{kwargs['version_number']}", state="STAGING")
        self.versions[version.version_id] = version
        return version

    def delete_weight_version(self, version_id):
        if self.fail_next_delete:
            self.fail_next_delete = False
            raise RuntimeError("retirement failed")
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
        self.metric_collections = 0
        self.metrics = {
            "perf/update_weights_density": 0.25,
            "perf/update_weights_wire_bytes": 123.0,
        }

    def stage_shard(self, *, version, hf_tensor_iter):
        staged = FakeStagedShard(self.control, version.version_id)
        self.stages.append((version, hf_tensor_iter, threading.current_thread().name, staged))
        return staged

    def pop_metrics(self):
        metrics, self.metrics = self.metrics, {}
        return metrics

    def collect_metrics(self):
        self.metric_collections += 1

    def release_version(self, *, version):
        self.releases.append(version)


class FakeEngine:
    def __init__(self, events, install_success=True):
        self.version = "base-uid"
        self.install_success = install_success
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
        if self.install_success:
            self.version = target
        return {
            "success": self.install_success,
            "installed_version": self.version,
            "detail": "install failed" if not self.install_success else "",
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
        modelexpress_initial_version="0",
        modelexpress_model_id="policy",
        modelexpress_base_version_id="base-uid",
    )


@pytest.fixture(autouse=True)
def patch_runtime(monkeypatch):
    modelexpress_rl = types.ModuleType("modelexpress_rl")
    modelexpress_rl.WeightPayloadFormat = types.SimpleNamespace(XOR_DELTA="XOR_DELTA")
    monkeypatch.setitem(sys.modules, "modelexpress_rl", modelexpress_rl)
    monkeypatch.setattr(mx_module.ray, "get", lambda refs: refs)
    monkeypatch.setattr(mx_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(mx_module.dist, "barrier", lambda group=None: None)
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
        instance._iter_hf_tensors = lambda: iter(())
    return instance, control, trainer


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


def test_slime_publishes_on_main_thread_and_activates_on_control_thread():
    instance, control, trainer = updater()
    events = []
    instance.connect_rollout_engines([FakeEngine(events)], object())

    instance.update_weights()
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
        "perf/update_weights_wire_bytes": 123.0,
        "perf/mx_receive_prepare_time": 2.0,
        "perf/mx_receive_install_time": 3.0,
    }
    assert trainer.metric_collections == 1
    assert instance.weight_version == 1


def test_reconnect_validates_current_ready_base():
    instance, _control, _trainer = updater()
    engine = FakeEngine([])

    instance.connect_rollout_engines([engine], object())
    instance.connect_rollout_engines([engine], object())
    instance._activation_executor.shutdown()


def test_connect_broadcasts_rank_zero_ray_errors(monkeypatch):
    instance, _control, _trainer = updater()
    transmitted = {}
    ray_get_calls = []

    def fail_ray_get(refs):
        ray_get_calls.append(refs)
        raise RuntimeError("status transport failed")

    def capture(values, src, group=None):
        transmitted["value"] = values[0]

    monkeypatch.setattr(mx_module.ray, "get", fail_ray_get)
    monkeypatch.setattr(mx_module.dist, "broadcast_object_list", capture)
    with pytest.raises(ModelExpressUpdateError) as rank_zero_error:
        instance.connect_rollout_engines([FakeEngine([])], object())

    monkeypatch.setattr(mx_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(
        mx_module.dist,
        "broadcast_object_list",
        lambda values, src, group=None: values.__setitem__(0, transmitted["value"]),
    )
    with pytest.raises(ModelExpressUpdateError) as non_root_error:
        instance.connect_rollout_engines([FakeEngine([])], object())
    instance._activation_executor.shutdown()

    assert str(non_root_error.value) == str(rank_zero_error.value)
    assert "status transport failed" in str(non_root_error.value)
    assert len(ray_get_calls) == 1


def test_ready_lookup_is_rank_zero_only_and_broadcasts_error(monkeypatch):
    instance, control, trainer = updater()
    events = []
    instance.connect_rollout_engines([FakeEngine(events)], object())
    instance.update_weights()
    transmitted = {}
    ready_lookups = []
    original_get = control.get_weight_version

    def fail_target_lookup(version_id):
        if version_id == "target-1":
            ready_lookups.append(version_id)
            raise RuntimeError("ready lookup failed")
        return original_get(version_id)

    def capture_error(values, src, group=None):
        if values[0] is not None and "error" in values[0]:
            transmitted["value"] = values[0]

    control.get_weight_version = fail_target_lookup
    monkeypatch.setattr(mx_module.dist, "broadcast_object_list", capture_error)
    with pytest.raises(ModelExpressUpdateError) as rank_zero_error:
        instance.update_weights()

    monkeypatch.setattr(mx_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(
        mx_module.dist,
        "broadcast_object_list",
        lambda values, src, group=None: values.__setitem__(0, transmitted["value"]),
    )
    with pytest.raises(ModelExpressUpdateError) as non_root_error:
        instance.update_weights()
    instance._activation_executor.shutdown()

    assert str(non_root_error.value) == str(rank_zero_error.value)
    assert "ready lookup failed" in str(non_root_error.value)
    assert ready_lookups == ["target-1"]
    assert len(control.creates) == 1
    assert len(trainer.stages) == 1
    assert events == []


def test_failed_install_retries_the_same_ready_target():
    instance, control, trainer = updater()
    events = []
    engine = FakeEngine(events, install_success=False)
    instance.connect_rollout_engines([engine], object())
    instance.update_weights()

    with pytest.raises(ModelExpressUpdateError, match="install failed"):
        instance.update_weights()
    assert instance._pending_version.version_id == "target-1"
    assert instance.weight_version == 0

    engine.install_success = True
    instance.update_weights()
    instance._activation_executor.shutdown()

    assert len(control.creates) == 1
    assert len(trainer.stages) == 1
    assert [event for event, _thread in events].count("install:target-1") == 2
    assert instance.weight_version == 1


def test_next_target_retries_retirement_before_advancing():
    instance, control, trainer = updater()
    events = []
    engine = FakeEngine(events)
    instance.connect_rollout_engines([engine], object())
    instance.update_weights()
    instance.update_weights()

    control.fail_next_delete = True
    with pytest.raises(ModelExpressUpdateError, match="retirement failed"):
        instance.update_weights()

    assert instance._pending_version.version_id == "target-2"
    assert instance._pending_applied is True
    assert instance.weight_version == 1
    assert len(control.creates) == 2
    assert len(trainer.stages) == 2
    assert [event for event, _thread in events].count("install:target-2") == 1

    instance.update_weights()
    instance._activation_executor.shutdown()

    assert len(control.creates) == 2
    assert len(trainer.stages) == 2
    assert [event for event, _thread in events].count("install:target-2") == 1
    assert control.deletes == ["target-1"]
    assert [version.version_id for version in trainer.releases] == ["target-1"]
    assert "base-uid" not in control.deletes
    assert instance._installed_version.version_id == "target-2"
    assert instance._pending_version is None
    assert instance.weight_version == 2


def test_receive_metrics_are_broadcast_to_the_logging_rank(monkeypatch):
    instance, _control, _trainer = updater()
    instance.connect_rollout_engines([FakeEngine([])], object())
    instance.update_weights()
    instance._pending_version = FakeVersion("target-1")
    monkeypatch.setattr(mx_module.dist, "get_rank", lambda: 1)

    def broadcast(values, src, group=None):
        values[0] = {"metrics": {"perf/mx_receive_prepare_time": 7.0}}

    monkeypatch.setattr(mx_module.dist, "broadcast_object_list", broadcast)

    instance.update_weights()
    assert instance._activation_executor is not None
    instance._activation_executor.shutdown()

    assert instance.pop_metrics()["perf/mx_receive_prepare_time"] == 7.0


def test_slime_hf_tensor_stream_preserves_native_order(monkeypatch):
    instance, _control, _trainer = updater(stub_gather=False)
    tensors = [("a", object()), ("b", object())]
    monkeypatch.setattr(instance, "_iter_non_expert_tensors", lambda: iter(tensors[:1]))
    monkeypatch.setattr(instance, "_iter_expert_tensors", lambda: iter(tensors[1:]))

    assert list(instance._iter_hf_tensors()) == tensors


def test_non_expert_hf_tensors_are_streamed_after_conversion(monkeypatch):
    instance, _control, _trainer = updater(stub_gather=False)
    instance._is_pp_src_rank = True
    params = [
        ("layer.a", torch.tensor([1.0])),
        ("layer.b", torch.tensor([2.0])),
    ]
    converted = []
    monkeypatch.setattr(mx_module, "named_params_and_buffers", lambda *_args: iter(params))
    monkeypatch.setattr(mx_module, "all_gather_param", lambda _name, param: param)

    def convert(_args, _model_name, name, param, _quantization_config):
        converted.append(name)
        return [(f"hf.{name}", param)]

    monkeypatch.setattr(mx_module, "convert_to_hf", convert)
    tensors = instance._iter_non_expert_tensors()

    assert next(tensors) == ("hf.layer.a", params[0][1])
    assert converted == ["layer.a"]
    assert list(tensors) == [("hf.layer.b", params[1][1])]


def test_expert_hf_tensors_are_streamed_after_conversion(monkeypatch):
    instance, _control, _trainer = updater(stub_gather=False)
    instance._is_pp_src_rank = True
    batch = [
        ("layer.experts.a", torch.tensor([1.0])),
        ("layer.experts.b", torch.tensor([2.0])),
    ]
    gathered = []
    monkeypatch.setattr(mx_module, "named_params_and_buffers", lambda *_args: iter(batch))
    monkeypatch.setattr(mx_module, "all_gather_param", lambda _name, param: param)

    def gather_and_convert(named_tensors):
        name, param = named_tensors[0]
        gathered.append(name)
        return [(f"hf.{name}", param)]

    monkeypatch.setattr(instance, "_ep_gather_and_convert", gather_and_convert)
    tensors = instance._iter_expert_tensors()

    assert next(tensors) == ("hf.layer.experts.a", batch[0][1])
    assert gathered == ["layer.experts.a"]
    assert list(tensors) == [("hf.layer.experts.b", batch[1][1])]
