"""Profile a few real TaskNet+PointACT batches without saving or updating weights.

Accepts the same model/data arguments as scripts/train.py. Intended for a
separate GPU allocation; never attach it to an existing training process.
"""

import hashlib
import os
import time

import torch
from transformers import set_seed

import train as training
from pointact.model.vla_pointact.action_head_3d import polar_router
from pointact.model.ptv3.concerto.model_ca_action import SerializedAttentionWithAction
from pointact.train.script_utils import log_trainable_parameters, parse_training_args
from pointact.train.train_utils import (
    configure_processor,
    configure_vlm,
    smart_tokenizer_and_embedding_resize,
)
from train_registry import resolve_recipe


def mark_call(owner, method, name, timings):
    original = getattr(owner, method)

    def wrapped(*args, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        timings.setdefault(name, []).append(time.perf_counter() - started)
        return result

    setattr(owner, method, wrapped)


def mark_cpu_call(owner, method, name, timings):
    original = getattr(owner, method)

    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        result = original(*args, **kwargs)
        timings.setdefault(name, []).append(time.perf_counter() - started)
        return result

    setattr(owner, method, wrapped)


def enable_experimental_route_cache():
    """Within one Point object, reuse exact routes for repeated serialized groups."""
    original = polar_router.PolarTokenRouter.__call__
    counters = {"hits": 0, "misses": 0}

    def cached(self, point, order, group_lengths):
        cache = point.setdefault("_profile_route_cache", {})
        digest = hashlib.blake2b(digest_size=16)
        digest.update(order.detach().cpu().contiguous().numpy().tobytes())
        digest.update(group_lengths.detach().cpu().contiguous().numpy().tobytes())
        key = (
            self.level, self.mode, self.neighbor_radius, self.max_tokens,
            point.coord.data_ptr(), point.polar_features.data_ptr(),
            point.polar_K.data_ptr(), point.T_camera_from_model.data_ptr(),
            digest.digest(),
        )
        if key in cache:
            counters["hits"] += 1
            routes, stats = cache[key]
            point.setdefault("polar_route_stats", []).append(dict(stats))
            return routes
        counters["misses"] += 1
        routes = original(self, point, order, group_lengths)
        cache[key] = (routes, dict(point.polar_route_stats[-1]))
        return routes

    polar_router.PolarTokenRouter.__call__ = cached
    return counters


def main():
    steps = int(os.environ.get("POINTACT_PROFILE_STEPS", "3"))
    if not 1 <= steps <= 10:
        raise ValueError("POINTACT_PROFILE_STEPS must be between 1 and 10")
    args = parse_training_args(logger=training.logger)
    set_seed(args.seed)
    recipe = resolve_recipe(args.model_class)
    dtype = training._compute_dtype(args)
    model = training.build_model(recipe, args, dtype)
    training.maybe_load_ptv3_checkpoint(model, args)
    processor = training.load_processor(recipe, args)
    smart_tokenizer_and_embedding_resize(processor, model)
    configure_vlm(model.vlm_backbone, args, dtype, args.device)
    create_data_module = training._import_object(recipe.data_module_fn)
    data_module = create_data_module(processor=processor, args=args)
    configure_processor(processor, data_module["train_dataset"], args, logger=training.logger)
    model.config.use_cache = False
    log_trainable_parameters(model, training.logger)
    trainer_class = training._import_object(recipe.trainer_class)
    trainer = trainer_class(model=model, processing_class=processor, args=args, **data_module)
    loader = trainer.get_train_dataloader()
    model.to(args.device)
    model.train()

    timings = {}
    mark_call(model.vlm_backbone.model, "forward", "Qwen", timings)
    mark_call(model.polarapp_encoder, "forward_task_features", "TaskNetFeatures", timings)
    mark_call(model.polarapp_encoder, "build_pyramid", "TaskNetPyramid", timings)
    mark_call(model.polarapp_encoder, "decode_normals", "TaskNetNormals", timings)
    mark_call(model.ptv3_model, "forward", "PTv3Interaction", timings)
    mark_call(model, "_decode_polar_completion", "PolarCompletionTotal", timings)
    mark_call(model.polar_depth_self_supervision, "forward", "DepthDecoder", timings)
    route_cache = (
        enable_experimental_route_cache()
        if os.environ.get("POINTACT_PROFILE_ROUTE_CACHE", "0") == "1"
        else None
    )
    mark_cpu_call(polar_router.PolarTokenRouter, "__call__", "PolarRouterCPU", timings)
    mark_cpu_call(polar_router, "_balanced_spatial_sample", "BalancedSampleCPU", timings)
    mark_cpu_call(polar_router, "_farthest_2d", "Farthest2DCPU", timings)
    mark_cpu_call(SerializedAttentionWithAction, "_forward_polar", "PolarAttentionCPU", timings)

    data_iter = iter(loader)
    for step in range(steps):
        started = time.perf_counter()
        batch = next(data_iter)
        data_wait = time.perf_counter() - started
        started = time.perf_counter()
        batch = trainer._prepare_inputs(batch)
        torch.cuda.synchronize()
        prepare = time.perf_counter() - started
        started = time.perf_counter()
        with trainer.compute_loss_context_manager():
            loss = trainer.compute_loss(model, batch)
        torch.cuda.synchronize()
        forward = time.perf_counter() - started
        started = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward = time.perf_counter() - started
        model.zero_grad(set_to_none=True)
        print(
            f"step={step} data_wait_s={data_wait:.3f} prepare_s={prepare:.3f} "
            f"forward_s={forward:.3f} backward_s={backward:.3f} loss={loss.item():.5f}",
            flush=True,
        )
        for name, values in timings.items():
            if name.endswith("CPU"):
                print(
                    f"stage={name} step={step} calls={len(values)} "
                    f"cumulative_seconds={sum(values):.3f}",
                    flush=True,
                )
            elif len(values) > step:
                print(f"stage={name} step={step} seconds={values[-1]:.3f}", flush=True)
        route_stats = getattr(model.ptv3_model, "last_polar_route_stats", None)
        if route_stats:
            print(
                "polar_route_stats="
                + str({
                    "route_calls": len(route_stats),
                    **{
                        key: sum(int(item.get(key, 0)) for item in route_stats)
                        for key in ("groups", "valid_projections", "candidates", "selected")
                    },
                }),
                flush=True,
            )
        if route_cache is not None:
            print(f"route_cache={route_cache}", flush=True)


if __name__ == "__main__":
    main()
