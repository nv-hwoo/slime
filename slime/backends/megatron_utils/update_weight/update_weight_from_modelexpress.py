from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

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


_UPDATE_PHASE_METRICS = (
    "perf/mx_control_create_weight_version",
    "perf/mx_stage_shard",
    "perf/mx_publish_shard",
    "perf/mx_control_get_weight_version_ready",
    "perf/mx_update_activate_time",
    "perf/mx_update_finalize_time",
)


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
        if "MX_REFIT_DELTA_BUCKET_BYTES" in os.environ:
            bucket_bytes = int(os.environ["MX_REFIT_DELTA_BUCKET_BYTES"])
            if bucket_bytes <= 0:
                raise ValueError("MX_REFIT_DELTA_BUCKET_BYTES must be positive")
            self.args.update_weight_buffer_size = bucket_bytes
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = int(args.modelexpress_initial_version)
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._connection_stale = False
        self._delta_base_prepared = False
        self._base_version_id = args.modelexpress_base_version_id

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
        if dist.get_rank() == 0:
            statuses = _non_null(
                ray.get([engine.get_modelexpress_status.remote() for engine in self.rollout_engines])
            )
            if len(statuses) != len(self.rollout_engines):
                raise ModelExpressUpdateError("rollout launch cohort is incomplete")
            for status in statuses:
                if status.get("installed_version") != expected_version or status.get("state") != "VERIFIED":
                    raise ModelExpressUpdateError("rollout cohort does not match the current revision")
        dist.barrier(group=get_gloo_group())

    def disconnect_rollout_engines(self) -> None:
        self.rollout_engines = None
        self._connection_stale = True

    def pop_metrics(self) -> dict[str, float]:
        metrics = getattr(self, "_metrics", {})
        self._metrics = {}
        return metrics

    @torch.no_grad()
    def update_weights(self) -> None:
        if not self._delta_base_prepared:
            self._trainer_client.prepare_delta_base(
                hf_tensor_iter=self._iter_hf_buckets(),
            )
            self._delta_base_prepared = True
            return

        if self.rollout_engines is None:
            raise ModelExpressUpdateError("rollout engines are not connected")

        phase_times = dict.fromkeys(_UPDATE_PHASE_METRICS, 0.0)

        from modelexpress_rl import WeightPayloadFormat, WeightVersionRef

        phase_started = perf_counter()
        payload = [None]
        if dist.get_rank() == 0:
            payload[0] = self._control_client.create_weight_version(
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
        dist.broadcast_object_list(payload, src=0, group=get_gloo_group())
        target = payload[0]
        phase_times["perf/mx_control_create_weight_version"] = perf_counter() - phase_started

        phase_started = perf_counter()
        staged = self._trainer_client.stage_shard(
            version=target.ref,
            hf_tensor_iter=self._iter_hf_buckets(),
        )
        phase_times["perf/mx_stage_shard"] = perf_counter() - phase_started

        phase_started = perf_counter()
        staged.publish()
        phase_times["perf/mx_publish_shard"] = perf_counter() - phase_started

        if dist.get_rank() == 0:
            phase_started = perf_counter()
            ready = self._control_client.get_weight_version(target.version_id)
            if getattr(ready.state, "value", ready.state) != "READY":
                raise ModelExpressUpdateError("published ModelExpress target is not READY")
            phase_times["perf/mx_control_get_weight_version_ready"] = perf_counter() - phase_started

        gloo_group = get_gloo_group()
        target_version = target.version_id
        activation_started = None
        future = None
        if dist.get_rank() == 0:
            activation_started = perf_counter()
            future = self._activation_executor.submit(self._activate_on_rank_zero, target_version)
        activation = [None]
        if future is not None:
            activation[0] = future.result()
        dist.broadcast_object_list(activation, src=0, group=gloo_group)
        receiver_metrics = activation[0]
        if activation_started is not None:
            phase_times["perf/mx_update_activate_time"] = perf_counter() - activation_started

        phase_started = None
        if self._base_version_id != self.args.modelexpress_base_version_id:
            if dist.get_rank() == 0:
                phase_started = perf_counter()
                self._control_client.delete_weight_version(self._base_version_id)
            dist.barrier(group=gloo_group)
            self._trainer_client.release_version(version=WeightVersionRef(self._base_version_id))
        if phase_started is not None:
            phase_times["perf/mx_update_finalize_time"] = perf_counter() - phase_started

        self._base_version_id = target_version
        self.weight_version += 1
        self._metrics = self._gather_metrics(
            phase_times=phase_times,
            receiver_metrics=receiver_metrics,
            group=gloo_group,
        )

    def _gather_metrics(
        self,
        *,
        phase_times: Mapping[str, float],
        receiver_metrics: Mapping[str, float],
        group: Any,
    ) -> dict[str, float]:
        local_metrics = self._trainer_client.pop_metrics()
        counts = torch.tensor(
            [
                local_metrics.get("changed_bytes", 0),
                local_metrics.get("total_bytes", 0),
                local_metrics.get("wire_bytes", 0),
            ],
            dtype=torch.int64,
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)

        timings = torch.tensor(
            [
                local_metrics.get("stage_delta_time", 0.0),
                local_metrics.get("publish_s3_time", 0.0),
                local_metrics.get("publish_server_time", 0.0),
                *(phase_times[name] for name in _UPDATE_PHASE_METRICS),
            ],
            dtype=torch.float64,
        )
        dist.all_reduce(timings, op=dist.ReduceOp.MAX, group=group)

        changed_bytes, total_bytes, wire_bytes = counts.tolist()
        stage_delta_time, publish_s3_time, publish_server_time, *phase_values = timings.tolist()
        return {
            "perf/update_weights_density": changed_bytes / max(total_bytes, 1),
            "perf/update_weights_wire_bytes": wire_bytes,
            "perf/mx_stage_delta_time": stage_delta_time,
            "perf/mx_publish_s3_time": publish_s3_time,
            "perf/mx_publish_server": publish_server_time,
            **receiver_metrics,
            **dict(zip(_UPDATE_PHASE_METRICS, phase_values, strict=True)),
        }

    def _iter_hf_buckets(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        for bucket_iter in (
            self._iter_non_expert_chunks(),
            self._iter_expert_chunks(),
        ):
            yield from bucket_iter
            dist.barrier(group=get_gloo_group())

    def _activate_on_rank_zero(self, target_version: str) -> dict[str, float]:
        engines = tuple(self.rollout_engines or ())
        prepared = ray.get([engine.prepare_weights_from_modelexpress.remote(target_version) for engine in engines])
        ray.get([engine.pause_generation.remote() for engine in engines])
        ray.get([engine.flush_cache.remote() for engine in engines])
        installed = ray.get([engine.update_weights_from_modelexpress.remote(target_version) for engine in engines])
        ray.get([engine.get_modelexpress_status.remote() for engine in engines])
        ray.get([engine.continue_generation.remote() for engine in engines])
        return _receiver_metrics([*prepared, *installed])
