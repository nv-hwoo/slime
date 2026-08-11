from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle

from slime.utils.disk_delta import make_tensor_reader
from slime.utils.distributed_utils import get_gloo_group

from .update_weight_from_disk_delta import UpdateWeightFromDiskDelta


class ModelExpressUpdateError(RuntimeError):
    pass


def _require_success(result: Mapping[str, Any], *, operation: str, target_version: str) -> None:
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


class UpdateWeightFromModelExpress(UpdateWeightFromDiskDelta):
    """Slime gather/control integration for the ModelExpress publisher."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        publisher=None,
    ) -> None:
        del weights_getter
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = int(args.modelexpress_initial_version)
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._connection_stale = False
        self._baseline_captured = False

        if publisher is None:
            from modelexpress.refit import Publisher, PublisherConfig, S3Config

            publisher = Publisher(
                launch_checkpoint=args.hf_checkpoint,
                bucket_bytes=args.update_weight_buffer_size,
                group=get_gloo_group(),
            )
            publisher.initialize(
                PublisherConfig(
                    model_id=args.modelexpress_model_id,
                    catalog_endpoint=args.modelexpress_catalog_endpoint,
                    s3=S3Config(
                        bucket=args.modelexpress_s3_bucket,
                        prefix=args.modelexpress_s3_prefix,
                        endpoint_url=args.modelexpress_s3_endpoint,
                    ),
                )
            )
        self._publisher = publisher
        self._catalog = publisher.catalog
        self._control = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="modelexpress-control")
            if dist.get_rank() == 0
            else None
        )
        self._publisher.publish_version("0")

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
        version = str(self.weight_version)
        launch_pending = self._publisher.pending_version == version
        if dist.get_rank() == 0:
            statuses = _non_null(ray.get([engine.get_modelexpress_status.remote() for engine in self.rollout_engines]))
            if len(statuses) != len(self.rollout_engines):
                raise ModelExpressUpdateError("rollout launch cohort is incomplete")
            for status in statuses:
                if (
                    status.get("installed_version") != version
                    or status.get("target_digest") != self._publisher.target_digest
                    or status.get("state") != "VERIFIED"
                ):
                    raise ModelExpressUpdateError("rollout cohort does not match the current revision")
            if launch_pending:
                self._catalog.commit_revision(self.args.modelexpress_model_id, version)
        if launch_pending:
            self._publisher.wait_for_commit(version)

    def disconnect_rollout_engines(self) -> None:
        self.rollout_engines = None
        self._connection_stale = True

    def pop_metrics(self) -> dict[str, float]:
        metrics = getattr(self, "_metrics", {})
        self._metrics = {}
        return metrics

    @torch.no_grad()
    def update_weights(self) -> None:
        if not self._baseline_captured:
            self._publisher.capture_baseline(
                self._for_each_hf_bucket,
                make_tensor_reader(self.args.hf_checkpoint),
            )
            self._baseline_captured = True
            return
        if self.rollout_engines is None:
            raise ModelExpressUpdateError("rollout engines are not connected")

        target_version = str(self.weight_version + 1)
        self._publisher.publish_version(
            target_version,
            base_version=str(self.weight_version),
            gather_hf_buckets=self._for_each_hf_bucket,
        )
        future = None
        if dist.get_rank() == 0:
            future = self._control.submit(self._activate_on_rank_zero, target_version)
        self._publisher.wait_for_commit(target_version, future)
        gloo_group = cast(dist.ProcessGroup, get_gloo_group())
        metrics_payload = [future.result() if future is not None else None]
        dist.broadcast_object_list(metrics_payload, src=0, group=gloo_group)
        receiver_metrics = metrics_payload[0] or {}
        dist.barrier(group=gloo_group)
        self.weight_version += 1
        self._metrics = {**self._publisher.pop_metrics(), **receiver_metrics}

    def _for_each_hf_bucket(self, consume) -> None:
        for chunk_iter in (
            self._iter_non_expert_chunks(),
            self._iter_expert_chunks(),
        ):
            for bucket in chunk_iter:
                consume(bucket)
            dist.barrier(group=get_gloo_group())

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
            if (
                status.get("installed_version") != target_version
                or status.get("target_digest") != self._publisher.pending_digest
                or status.get("state") != "VERIFIED"
            ):
                raise ModelExpressUpdateError("receiver status is not VERIFIED")
        self._catalog.commit_revision(self.args.modelexpress_model_id, target_version)
        ray.get([engine.continue_generation.remote() for engine in engines])
        return _receiver_metrics([*prepared, *installed])
