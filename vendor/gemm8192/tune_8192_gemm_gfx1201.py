"""Shape-specific AITER tuning before the independent 8192 GEMM race.

Only explicit candidate configs are searched here. PyTorch baselines retain
their library dispatch and do not participate in AITER's configuration search.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import statistics
import time
import traceback
from typing import Any, Callable


SIZE = 8192


def triton_variants(precision: str) -> tuple[dict[str, int], ...]:
    """A bounded, large-square search; the installed default is added separately."""
    if precision in ("bf16", "fp16"):
        tiles = (
            (128, 128, 64, 8, 2), (128, 256, 64, 8, 2),
            (256, 128, 64, 8, 2), (128, 128, 128, 8, 2),
            (128, 256, 128, 8, 2), (256, 128, 128, 8, 2),
            (64, 128, 128, 4, 2), (128, 64, 128, 4, 2),
        )
    elif precision == "fp8":
        tiles = (
            (16, 64, 128, 4, 2), (32, 64, 128, 4, 2),
            (64, 64, 128, 4, 2), (64, 128, 128, 4, 2),
            (128, 64, 128, 4, 2), (128, 128, 128, 8, 2),
            (64, 128, 256, 8, 2), (128, 128, 256, 8, 2),
        )
    else:
        raise ValueError(f"unknown precision: {precision}")
    return tuple(
        {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
         "num_warps": warps, "num_stages": stages}
        for bm, bn, bk, warps, stages in tiles
    )


def gluon_variants() -> tuple[dict[str, Any], ...]:
    return tuple(
        {"variant": variant, "GROUP_SIZE_M": group}
        for variant in (
            "reuse_128x256", "reuse_256x128", "square_256x256",
            "aspect_256x128", "ordered_128x256",
        )
        # H3 NK evidence includes a G4 winner; G32 had no stable advantage.
        for group in (4, 8, 16)
    )


def expand_triton_configs(base: dict[str, Any], precision: str) -> list[dict[str, Any]]:
    candidates = []
    seen = set()
    variants = [({}, None)]
    variants.extend((delta, split_k) for delta in triton_variants(precision)
                    for split_k in (1, 2, 4))
    for delta, split_k in variants:
        config = copy.deepcopy(base)
        config.update(delta)
        if split_k is not None:
            config["NUM_KSPLIT"] = split_k
            config["SPLITK_BLOCK_SIZE"] = SIZE // split_k
        key = json.dumps(config, sort_keys=True)
        if key not in seen:
            candidates.append(config)
            seen.add(key)
    return candidates


def triton_configs(precision: str) -> list[dict[str, Any]]:
    if precision == "fp8":
        from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8 import _get_config
    else:
        from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import _get_config
    base, _ = _get_config(SIZE, SIZE, SIZE)
    return expand_triton_configs(base, precision)


def ck_kernel_ids() -> list[tuple[int, str]]:
    from csrc.ck_gemm_a8w8.gemm_a8w8_common import kernels_list
    return [(kernel_id, kernel.name) for kernel_id, kernel in sorted(kernels_list.items())]


def save_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def best_valid(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if row["status"] == "ok" and row.get("median_ms", 0) > 0]
    return min(valid, key=lambda row: row["median_ms"]) if valid else None


def tune_torch_baseline(
    torch: Any, precision: str, x: Any, w: Any, x_scale: Any | None,
    w_scale: Any | None, indices: Any, reference: Any,
    build_calls: Callable[..., dict[str, Any]],
    accuracy: Callable[..., dict[str, float]], args: Any, path: Path,
) -> dict[str, Any]:
    """Tune PyTorch's eligible GEMM dispatch, then disable further tuning."""
    name = {"bf16": "torch_mm", "fp16": "torch_mm_then_bf16",
            "fp8": "torch_scaled_mm"}[precision]
    state: dict[str, Any] = {"precision": precision, "candidate": name,
                             "status": "pending"}
    save_state(path, state)
    try:
        call = build_calls(torch, precision, x, w, x_scale, w_scale, {})[name]
        before = {tuple(map(str, item)) for item in torch.cuda.tunable.get_results()}
        started = time.perf_counter()
        torch.cuda.tunable.tuning_enable(True)
        try:
            output = call()
            torch.cuda.synchronize(x.device)
        finally:
            torch.cuda.tunable.tuning_enable(False)
        state["tune_wall_ms"] = (time.perf_counter() - started) * 1000
        metrics = accuracy(torch, output, indices, reference)
        state.update(metrics)
        state["max_abs_limit"] = max(1.0, 10 * args.max_nrmse * metrics["reference_rms"])
        if metrics["nrmse"] > args.max_nrmse or metrics["max_abs"] > state["max_abs_limit"]:
            state["status"] = "accuracy_failed"
        else:
            after = {tuple(map(str, item)) for item in torch.cuda.tunable.get_results()}
            state["new_tunable_results"] = sorted(after - before)
            state["status"] = "ok" if state["new_tunable_results"] else "library_dispatch"
    except Exception:
        state.update(status="failed", error=traceback.format_exc(limit=5))
    save_state(path, state)
    return state


def tune_lane(
    torch: Any,
    precision: str,
    x: Any,
    w: Any,
    x_scale: Any | None,
    w_scale: Any | None,
    indices: Any,
    reference: Any,
    accuracy: Callable[..., dict[str, float]],
    measure: Callable[..., float],
    args: Any,
    path: Path,
) -> dict[str, dict[str, Any]]:
    """Return only verified, measured configs; never silently use defaults."""
    state: dict[str, Any] = {"precision": precision, "phase": "tuning", "rows": [], "selected": {}}
    save_state(path, state)
    selected: dict[str, dict[str, Any]] = {}
    candidates = ["aiter_triton"]
    if precision == "bf16":
        candidates.append("aiter_gluon")
    if precision == "fp8":
        candidates.append("aiter_ck")
        if os.getenv("AITER_MODE") == "image":
            candidates.append("aiter_gfx1201_mm8")
    y = torch.empty((SIZE, SIZE), dtype=torch.bfloat16, device=x.device)

    for candidate in candidates:
        choices: list[tuple[str, dict[str, Any]]] = []
        try:
            if candidate == "aiter_triton":
                choices = [(f"config_{index:02d}", {"config": config})
                           for index, config in enumerate(triton_configs(precision))]
            elif candidate == "aiter_gluon":
                choices = [(config["variant"] + f"_g{config['GROUP_SIZE_M']}",
                            {"config": config}) for config in gluon_variants()]
            elif candidate == "aiter_gfx1201_mm8":
                # Public image API has fixed geometry; validate and time it as shipped.
                choices = [("image_default", {"implementation": "mm8_out", "tunable": False})]
            else:
                choices = [(f"kernel_{kernel_id}_split{split_k}",
                            {"kernel_id": kernel_id, "split_k": split_k,
                             "kernel_name": name})
                           for kernel_id, name in ck_kernel_ids()
                           for split_k in (0, 1)]
        except Exception:
            state["rows"].append({"candidate": candidate, "choice": "discovery",
                                  "status": "failed", "error": traceback.format_exc(limit=5)})
            save_state(path, state)
            continue

        candidate_rows = []
        for index, (label, spec) in enumerate(choices):
            if index % 10 == 0:
                print(f"tune {precision}/{candidate}: starting {index + 1}/{len(choices)} ({label}); first call may compile", flush=True)
            row: dict[str, Any] = {"candidate": candidate, "choice": label,
                                   "spec": spec, "status": "pending"}
            state["rows"].append(row)
            candidate_rows.append(row)
            save_state(path, state)
            try:
                if candidate == "aiter_triton" and precision != "fp8":
                    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
                    call = lambda: gemm_a16w16(x, w, dtype=torch.bfloat16, y=y,
                                               config=spec["config"], backend="triton")
                elif candidate == "aiter_triton":
                    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8
                    call = lambda: gemm_a8w8(x, w, x_scale, w_scale,
                                             dtype=torch.bfloat16, y=y,
                                             config=spec["config"], backend="triton")
                elif candidate == "aiter_gluon":
                    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
                    call = lambda: gemm_a16w16(x, w, dtype=torch.bfloat16, y=y,
                                               config=spec["config"], backend="gluon")
                elif candidate == "aiter_gfx1201_mm8":
                    from aiter.ops.gfx1201.fp8_gemm import mm8_out
                    input_scales, weight_scales = x_scale.view(-1), w_scale.view(-1)
                    call = lambda: mm8_out(x, input_scales, w, weight_scales, y)
                else:
                    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_tune
                    call = lambda: gemm_a8w8_tune(x, w, x_scale, w_scale, y,
                                                  spec["kernel_id"], spec["split_k"])
                started = time.perf_counter()
                output = call()
                torch.cuda.synchronize(x.device)
                row["first_call_wall_ms"] = (time.perf_counter() - started) * 1000
                metrics = accuracy(torch, output, indices, reference)
                row.update(metrics)
                row["max_abs_limit"] = max(1.0, 10 * args.max_nrmse * metrics["reference_rms"])
                if metrics["nrmse"] > args.max_nrmse or metrics["max_abs"] > row["max_abs_limit"]:
                    row["status"] = "accuracy_failed"
                else:
                    for _ in range(args.tune_warmup):
                        call()
                    torch.cuda.synchronize(x.device)
                    samples = [measure(torch, call, 1) for _ in range(args.tune_samples)]
                    row.update(status="ok", median_ms=statistics.median(samples),
                               samples_ms=samples)
            except Exception:
                row.update(status="failed", error=traceback.format_exc(limit=5))
            save_state(path, state)
            if index % 10 == 0 or index + 1 == len(choices):
                print(f"tune {precision}/{candidate}: {index + 1}/{len(choices)}", flush=True)
        winner = best_valid(candidate_rows)
        if winner is not None:
            selected[candidate] = winner["spec"]
            state["selected"][candidate] = {"choice": winner["choice"],
                                             "spec": winner["spec"],
                                             "tune_median_ms": winner["median_ms"]}
        save_state(path, state)
    state["phase"] = "complete"
    save_state(path, state)
    return selected
