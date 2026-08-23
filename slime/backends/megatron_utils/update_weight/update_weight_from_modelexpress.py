from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param, named_params_and_buffers
from .update_weight_from_distributed import UpdateWeightFromDistributed


class ModelExpressUpdateError(RuntimeError):
    pass


def _require_success(result: Mapping[str, Any], operation: str, target_version: str) -> None:
    if result.get("success") is not True:
        raise ModelExpressUpdateError(
            f"ModelExpress {operation} failed for {target_version}: {result.get('detail', '')}"
        )


def _non_null(results: Sequence[Any]) -> list[Mapping[str, Any]]:
    return [result for result in results if result is not None]


def _receiver_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for result in results:
        for key, value in result.get("metrics", {}).items():
            metrics[key] = max(metrics.get(key, 0.0), float(value))
    return metrics


class UpdateWeightFromModelExpress(UpdateWeightFromDistributed):
    """Slime gather/control integration for ModelExpress S3/XOR refit."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        control_client=None,
        trainer_client=None,
    ) -> None:
        del weights_getter
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = int(args.modelexpress_initial_version)
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._connection_stale = False
        self._skip_startup_update = True
        self._base_version_id = args.modelexpress_base_version_id
        self._pending_version = None
        self._pending_applied = False
        self._pending_receiver_metrics = {}
        self._installed_version = None

        if control_client is None or trainer_client is None:
            from modelexpress_rl import (
                ModelExpressControlClient,
                ModelExpressTrainerClient,
                ModelExpressTrainerConfig,
                S3Config,
                TrainerStagingMode,
                WeightPayloadFormat,
            )

            if control_client is None:
                control_client = ModelExpressControlClient.connect(server_url=args.modelexpress_server_url)

        self._control_client = control_client
        base = self._control_client.get_weight_version(self._base_version_id)
        if base.model_name != args.modelexpress_model_id:
            raise ModelExpressUpdateError("ModelExpress base version belongs to a different model")
        if getattr(base.state, "value", base.state) != "READY":
            raise ModelExpressUpdateError("ModelExpress base version is not READY")
        if trainer_client is None:
            trainer_client = ModelExpressTrainerClient.initialize(
                ModelExpressTrainerConfig(
                    model_name=args.modelexpress_model_id,
                    server_url=args.modelexpress_server_url,
                    staging_mode=TrainerStagingMode.WRITE_TO_STORAGE,
                    payload_format=WeightPayloadFormat.XOR_DELTA,
                    process_group=get_gloo_group(),
                    s3=S3Config(
                        bucket=args.modelexpress_s3_bucket,
                        prefix=args.modelexpress_s3_prefix,
                        endpoint_url=args.modelexpress_s3_endpoint,
                        initial_base_version_id=self._base_version_id,
                        launch_checkpoint=args.hf_checkpoint,
                    ),
                )
            )
        self._trainer_client = trainer_client

        self._activation_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="modelexpress-control")
            if dist.get_rank() == 0
            else None
        )

    def is_rollout_engines_fresh(self) -> bool:
        return self.rollout_engines is not None and not self._connection_stale

    def mark_engine_connection_stale(self) -> None:
        self._connection_stale = True

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        engine_parallel_configs: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        del rollout_engine_lock, engine_gpu_counts, engine_gpu_offsets, engine_parallel_configs
        self.rollout_engines = tuple(rollout_engines)
        self._connection_stale = False
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        expected_version = self._base_version_id
        validation = [None]
        if dist.get_rank() == 0:
            try:
                statuses = _non_null(
                    ray.get([engine.get_modelexpress_status.remote() for engine in self.rollout_engines])
                )
                if len(statuses) != len(self.rollout_engines):
                    raise ModelExpressUpdateError("rollout launch cohort is incomplete")
                for status in statuses:
                    if status.get("installed_version") != expected_version or status.get("state") != "VERIFIED":
                        raise ModelExpressUpdateError("rollout cohort does not match the current revision")
                validation[0] = {"success": True}
            except Exception as error:
                validation[0] = {"error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(validation, src=0, group=get_gloo_group())
        if "error" in validation[0]:
            raise ModelExpressUpdateError(validation[0]["error"])

    def disconnect_rollout_engines(self) -> None:
        self.rollout_engines = None
        self._connection_stale = True

    def pop_metrics(self) -> dict[str, float]:
        metrics = getattr(self, "_metrics", {})
        self._metrics = {}
        return metrics

    @torch.no_grad()
    def update_weights(self) -> None:
        # Slime invokes the updater once at startup. The explicit READY base
        # already represents those launch weights, so there is nothing to publish.
        if self._skip_startup_update:
            self._skip_startup_update = False
            return
        if self.rollout_engines is None:
            raise ModelExpressUpdateError("rollout engines are not connected")

        if self._pending_version is None:
            from modelexpress_rl import WeightPayloadFormat

            payload = [None]
            if dist.get_rank() == 0:
                try:
                    payload[0] = {
                        "version": self._control_client.create_weight_version(
                            model_name=self.args.modelexpress_model_id,
                            idempotency_key=(
                                f"slime:{self.args.modelexpress_model_id}:"
                                f"{self._base_version_id}:{self.weight_version + 1}"
                            ),
                            version_number=self.weight_version + 1,
                            payload_format=WeightPayloadFormat.XOR_DELTA,
                            base_version_id=self._base_version_id,
                            expected_source_slots=["canonical.delta.root"],
                        )
                    }
                except Exception as error:
                    payload[0] = {"error": f"{type(error).__name__}: {error}"}
            dist.broadcast_object_list(payload, src=0, group=get_gloo_group())
            if "error" in payload[0]:
                raise ModelExpressUpdateError(f"ModelExpress create failed: {payload[0]['error']}")
            target = payload[0]["version"]
            staged = self._trainer_client.stage_shard(
                version=target.ref,
                hf_tensor_iter=self._iter_hf_tensors(),
            )
            staged.publish()
            self._pending_version = target

        if getattr(self._pending_version.state, "value", self._pending_version.state) != "READY":
            ready_payload = [None]
            if dist.get_rank() == 0:
                try:
                    ready = self._control_client.get_weight_version(self._pending_version.version_id)
                    if getattr(ready.state, "value", ready.state) != "READY":
                        raise ModelExpressUpdateError("published ModelExpress target is not READY")
                    ready_payload[0] = {"version": ready}
                except Exception as error:
                    ready_payload[0] = {"error": f"{type(error).__name__}: {error}"}
            dist.broadcast_object_list(ready_payload, src=0, group=get_gloo_group())
            if "error" in ready_payload[0]:
                raise ModelExpressUpdateError(ready_payload[0]["error"])
            self._pending_version = ready_payload[0]["version"]

        gloo_group = get_gloo_group()
        target_version = self._pending_version.version_id
        if not self._pending_applied:
            future = None
            if dist.get_rank() == 0:
                future = self._activation_executor.submit(self._activate_on_rank_zero, target_version)
            activation = [None]
            if future is not None:
                try:
                    activation[0] = {"metrics": future.result()}
                except Exception as error:
                    activation[0] = {"error": f"{type(error).__name__}: {error}"}
            dist.broadcast_object_list(activation, src=0, group=gloo_group)
            if "error" in activation[0]:
                raise ModelExpressUpdateError(activation[0]["error"])
            self._pending_receiver_metrics = activation[0]["metrics"]
            self._pending_applied = True

        previous = self._installed_version
        if previous is not None:
            retirement = [None]
            if dist.get_rank() == 0:
                try:
                    self._control_client.delete_weight_version(previous.version_id)
                    retirement[0] = {"success": True}
                except Exception as error:
                    retirement[0] = {"error": f"{type(error).__name__}: {error}"}
            dist.broadcast_object_list(retirement, src=0, group=gloo_group)
            if "error" in retirement[0]:
                raise ModelExpressUpdateError(f"ModelExpress retirement failed: {retirement[0]['error']}")
            self._trainer_client.release_version(version=previous.ref)

        dist.barrier(group=gloo_group)
        self._base_version_id = target_version
        self._installed_version = self._pending_version
        self._pending_version = None
        self._pending_applied = False
        self.weight_version += 1
        self._trainer_client.collect_metrics()
        self._metrics = {
            **self._trainer_client.pop_metrics(),
            **self._pending_receiver_metrics,
        }
        self._pending_receiver_metrics = {}

    def _iter_hf_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        for tensor_iter in (
            self._iter_non_expert_tensors(),
            self._iter_expert_tensors(),
        ):
            yield from tensor_iter
            dist.barrier(group=get_gloo_group())

    def _iter_non_expert_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            param = all_gather_param(name, param)
            if self._is_pp_src_rank:
                yield from convert_to_hf(
                    self.args,
                    self.model_name,
                    name,
                    param,
                    self.quantization_config,
                )

    def _iter_expert_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." not in name:
                continue
            param = all_gather_param(name, param)
            yield from self._ep_gather_and_convert([(name, param)])

    def _activate_on_rank_zero(self, target_version: str) -> dict[str, float]:
        engines = tuple(self.rollout_engines or ())
        if not engines:
            raise ModelExpressUpdateError("ModelExpress requires rollout engines")
        prepared = _non_null(
            ray.get([engine.prepare_weights_from_modelexpress.remote(target_version) for engine in engines])
        )
        if len(prepared) != len(engines):
            raise ModelExpressUpdateError("rollout prepare cohort is incomplete")
        for result in prepared:
            _require_success(result, operation="prepare", target_version=target_version)
        ray.get([engine.pause_generation.remote() for engine in engines])
        ray.get([engine.flush_cache.remote() for engine in engines])
        installed = _non_null(
            ray.get([engine.update_weights_from_modelexpress.remote(target_version) for engine in engines])
        )
        if len(installed) != len(engines):
            raise ModelExpressUpdateError("rollout install cohort is incomplete")
        for result in installed:
            _require_success(result, operation="install", target_version=target_version)
            if result.get("installed_version") != target_version:
                raise ModelExpressUpdateError("receiver installed the wrong version")
        statuses = _non_null(ray.get([engine.get_modelexpress_status.remote() for engine in engines]))
        if len(statuses) != len(engines):
            raise ModelExpressUpdateError("rollout verification cohort is incomplete")
        for status in statuses:
            if status.get("installed_version") != target_version or status.get("state") != "VERIFIED":
                raise ModelExpressUpdateError("receiver status is not VERIFIED")
        ray.get([engine.continue_generation.remote() for engine in engines])
        return _receiver_metrics([*prepared, *installed])
