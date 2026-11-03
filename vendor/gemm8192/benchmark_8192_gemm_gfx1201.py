#!/usr/bin/env python3
"""Customer 8192^3 GEMM race. --dry-run and pure helpers need no torch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any


SIZE = 8192
SHAPE = (SIZE, SIZE, SIZE)
FLOPS = 2 * SIZE**3
CANDIDATES = {
    "bf16": ("torch_mm", "aiter_triton", "aiter_gluon"),
    "fp16": ("torch_mm_then_bf16", "aiter_triton"),
    "fp8": ("torch_scaled_mm", "aiter_triton", "aiter_ck"),
}


def candidate_names(precision: str) -> tuple[str, ...]:
    names = CANDIDATES[precision]
    if precision == "fp8" and os.getenv("AITER_MODE") == "image":
        return (*names, "aiter_gfx1201_mm8")
    return names


def tflops(latency_ms: float) -> float:
    if latency_ms <= 0:
        raise ValueError("latency_ms must be positive")
    return FLOPS / (latency_ms / 1000.0) / 1e12


def memory_bytes(precision: str) -> int:
    element_bytes = 1 if precision == "fp8" else 2
    scales = 2 * SIZE * 4 if precision == "fp8" else 0
    return 2 * SIZE * SIZE * element_bytes + SIZE * SIZE * 2 + scales


def winners(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for precision in CANDIDATES:
        valid = [
            row for row in rows
            if row["precision"] == precision
            and row["status"] == "ok"
            and row.get("median_ms", 0) > 0
        ]
        if valid:
            result[precision] = min(valid, key=lambda row: row["median_ms"])
    return result


def git_commit(directory: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def verify_source_aiter(aiter: Any, out_dir: Path) -> dict[str, Any]:
    """Fail before tuning if the image wheel shadows the pinned source tree."""
    source_root = Path(os.environ["AITER_SOURCE_ROOT"]).resolve()
    expected = os.environ["AITER_SOURCE_REVISION"]
    # The launcher leaves its managed checkout detached at the pinned revision.
    # Reading HEAD avoids requiring the otherwise-offline image to contain Git.
    revision = (source_root / ".git" / "HEAD").read_text(encoding="ascii").strip()
    gemm_module = importlib.import_module("aiter.ops.triton.gemm.basic.gemm_a16w16")
    csrc_module = importlib.import_module("csrc.cpp_itfs.utils")
    ck_module = importlib.import_module("csrc.ck_gemm_a8w8.gemm_a8w8_common")
    paths = {"aiter": Path(aiter.__file__).resolve(),
             "gluon_wrapper": Path(gemm_module.__file__).resolve(),
             "csrc": Path(csrc_module.__file__).resolve(),
             "ck_catalog": Path(ck_module.__file__).resolve()}
    supported = tuple(gemm_module._GLUON_SUPPORTED_ARCHS)
    report = {"expected_commit": expected, "actual_commit": revision,
              "source_root": str(source_root),
              "paths": {name: str(path) for name, path in paths.items()},
              "gluon_supported_archs": supported}
    (out_dir / "aiter-source-preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if revision != expected:
        raise RuntimeError(f"AITER source revision mismatch: {revision} != {expected}")
    if any(not path.is_relative_to(source_root) for path in paths.values()):
        raise RuntimeError(f"AITER/image Python source mixing: {report['paths']}")
    if "gfx1201" not in supported:
        raise RuntimeError(f"source AITER Gluon still rejects gfx1201: {supported}")
    return report


def verify_image_aiter(aiter: Any, out_dir: Path) -> dict[str, Any]:
    """Record the installed wheel and reject accidental host-source imports."""
    report: dict[str, Any] = {"mode": "image", "status": "pending", "sources": {}}
    try:
        distribution = importlib.metadata.distribution("amd-aiter")
        package_root = Path(distribution.locate_file("aiter")).resolve()
        csrc_root = Path(distribution.locate_file("aiter_meta/csrc")).resolve()
        expected_version = os.getenv("EXPERIMENT_AITER_VERSION")
        report.update(version=distribution.version,
                      image_version=expected_version,
                      image_revision=os.getenv("EXPERIMENT_AITER_REVISION"),
                      package_root=str(package_root), csrc_root=str(csrc_root))
        if expected_version and expected_version != "<no value>" and distribution.version != expected_version:
            raise RuntimeError(f"image AITER version mismatch: {distribution.version} != {expected_version}")
        modules = {
            "aiter": aiter,
            **{name: importlib.import_module(name) for name in (
                "aiter.ops.triton.gemm.basic.gemm_a16w16",
                "aiter.ops.triton.gemm.basic.gemm_a16w16_gfx1201",
                "aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16",
                "aiter.ops.triton.gemm.basic.gemm_a8w8",
                "aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8",
                "aiter.ops.gemm_op_a8w8",
                "aiter.ops.gfx1201.fp8_gemm",
                "csrc.cpp_itfs.utils",
                "csrc.ck_gemm_a8w8.gemm_a8w8_common",
            )},
        }
        for name, module in modules.items():
            path = Path(module.__file__).resolve()
            allowed_root = csrc_root if name.startswith("csrc.") else package_root
            report["sources"][name] = {"path": str(path)}
            if not path.is_relative_to(allowed_root):
                raise RuntimeError(f"image AITER/source mixing: {name} loaded from {path}")
            report["sources"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        supported = tuple(modules["aiter.ops.triton.gemm.basic.gemm_a16w16"]._GLUON_SUPPORTED_ARCHS)
        report["gluon_supported_archs"] = supported
        if "gfx1201" not in supported:
            raise RuntimeError(f"image BF16 Gluon rejects gfx1201: {supported}")
        if not callable(modules["aiter.ops.gfx1201.fp8_gemm"].mm8_out):
            raise RuntimeError("image lacks callable gfx1201 FP8 mm8_out")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (out_dir / "aiter-image-preflight.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def write_results(out_dir: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    payload = {"metadata": metadata, "results": rows,
               "winners": {key: row["candidate"] for key, row in winners(rows).items()}}
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fields = ["precision", "candidate", "status", "first_call_wall_ms",
              "median_ms", "min_ms", "max_ms", "tflops", "nrmse", "max_abs",
              "max_rel", "max_abs_limit", "error"]
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 8192 GEMM 算子比赛", "", "单卡、本地完整 8192³；GPU event 时间；输入已量化/就绪，未计量化与预打包。", "",
             "| 输入 | 候选 | 状态 | 中位数 ms | TFLOP/s | NRMSE |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        fmt = lambda key: "—" if row.get(key) is None else f"{row[key]:.4f}"
        lines.append(f"| {row['precision']} | {row['candidate']} | {row['status']} | {fmt('median_ms')} | {fmt('tflops')} | {fmt('nrmse')} |")
    lines += ["", "各精度胜者：" + ("；".join(f"{p}: {r['candidate']}" for p, r in winners(rows).items()) or "无通过正确性门槛的候选")]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fill_matrix(torch: Any, tensor: Any, seed: int) -> None:
    torch.manual_seed(seed)
    for start in range(0, SIZE, 128):
        stop = min(start + 128, SIZE)
        tensor[start:stop].uniform_(-1.0, 1.0)


def quantize_rows(torch: Any, target: Any, scales: Any, seed: int) -> None:
    torch.manual_seed(seed)
    for start in range(0, SIZE, 128):
        stop = min(start + 128, SIZE)
        source = torch.empty((stop - start, SIZE), device=target.device,
                             dtype=torch.float32).uniform_(-1.0, 1.0)
        scale = (source.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-8)
        target[start:stop].copy_((source / scale).clamp(-448, 448).to(target.dtype))
        scales[start:stop].copy_(scale)


def sampled_reference(torch: Any, precision: str, x: Any, w: Any,
                      x_scale: Any | None, w_scale: Any | None) -> tuple[Any, Any]:
    indices = torch.linspace(0, SIZE - 1, 16, device=x.device).round().long()
    if precision == "fp8":
        left = x.index_select(0, indices).float() * x_scale.index_select(0, indices)
        right = w.float() * w_scale.T
    else:
        left = x.index_select(0, indices).float()
        right = w.float()
    return indices, (left @ right.T).to(torch.bfloat16)


def accuracy(torch: Any, output: Any, indices: Any, reference: Any) -> dict[str, float]:
    if tuple(output.shape) != (SIZE, SIZE) or output.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 [{SIZE},{SIZE}], got {output.shape}, {output.dtype}")
    actual = output.index_select(0, indices).float()
    expected = reference.float()
    if not bool(torch.isfinite(actual).all()):
        raise ValueError("non-finite output")
    difference = (actual - expected).abs()
    nrmse = float(torch.sqrt((difference**2).mean()) /
                   torch.sqrt((expected**2).mean()).clamp_min(1e-6))
    maximum = float(difference.max())
    relative = float((difference / expected.abs().clamp_min(1.0)).max())
    return {"nrmse": nrmse, "max_abs": maximum, "max_rel": relative,
            "reference_rms": float(torch.sqrt((expected**2).mean()))}


def measure(torch: Any, call: Any, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def build_calls(torch: Any, precision: str, x: Any, w: Any,
                x_scale: Any | None, w_scale: Any | None,
                selection: dict[str, Any] | None = None) -> dict[str, Any]:
    calls = {}
    if precision in ("bf16", "fp16"):
        def aiter_half(backend: str, y: Any, config: dict[str, Any] | None) -> Any:
            from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
            return gemm_a16w16(x, w, dtype=torch.bfloat16, y=y,
                               config=config, backend=backend)

        y = torch.empty((SIZE, SIZE), dtype=torch.bfloat16, device=x.device)
        if precision == "bf16":
            calls["torch_mm"] = lambda: torch.mm(x, w.T, out=y)
            if selection is None or "aiter_gluon" in selection:
                gluon_config = selection["aiter_gluon"]["config"] if selection is not None else None
                calls["aiter_gluon"] = lambda: aiter_half("gluon", y, gluon_config)
        else:
            temp = torch.empty((SIZE, SIZE), dtype=torch.float16, device=x.device)

            def torch_fp16_then_bf16() -> Any:
                torch.mm(x, w.T, out=temp)
                y.copy_(temp)
                return y

            calls["torch_mm_then_bf16"] = torch_fp16_then_bf16
        if selection is None or "aiter_triton" in selection:
            triton_config = selection["aiter_triton"]["config"] if selection is not None else None
            calls["aiter_triton"] = lambda: aiter_half("triton", y, triton_config)
    else:
        y = torch.empty((SIZE, SIZE), dtype=torch.bfloat16, device=x.device)
        calls["torch_scaled_mm"] = lambda: torch._scaled_mm(
            x, w.T, scale_a=x_scale, scale_b=w_scale,
            out_dtype=torch.bfloat16, use_fast_accum=False,
        )
        if selection is None or "aiter_triton" in selection:
            triton_config = selection["aiter_triton"]["config"] if selection is not None else None

            def aiter_fp8_triton() -> Any:
                from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8
                return gemm_a8w8(x, w, x_scale, w_scale, dtype=torch.bfloat16,
                                 y=y, config=triton_config, backend="triton")

            calls["aiter_triton"] = aiter_fp8_triton
        if selection is None or "aiter_ck" in selection:
            def aiter_fp8_ck() -> Any:
                if selection is None:
                    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_CK
                    return gemm_a8w8_CK(x, w, x_scale, w_scale, dtype=torch.bfloat16)
                from aiter.ops.gemm_op_a8w8 import gemm_a8w8_tune
                spec = selection["aiter_ck"]
                return gemm_a8w8_tune(x, w, x_scale, w_scale, y,
                                      spec["kernel_id"], spec["split_k"])

            calls["aiter_ck"] = aiter_fp8_ck
        if "aiter_gfx1201_mm8" in candidate_names(precision) and (
            selection is None or "aiter_gfx1201_mm8" in selection
        ):
            # Shape-only views; quantization and scale creation remain outside timing.
            input_scales, weight_scales = x_scale.view(-1), w_scale.view(-1)

            def image_fp8_mm8() -> Any:
                from aiter.ops.gfx1201.fp8_gemm import mm8_out
                return mm8_out(x, input_scales, w, weight_scales, y)

            calls["aiter_gfx1201_mm8"] = image_fp8_mm8
    return calls


def prepare_lane(torch: Any, precision: str, gpu: int) -> tuple[Any, ...]:
    device = torch.device("cuda", gpu)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp8": torch.float8_e4m3fn}[precision]
    x = torch.empty((SIZE, SIZE), dtype=dtype, device=device)
    w = torch.empty_like(x)
    x_scale = w_scale = None
    if precision == "fp8":
        x_scale = torch.empty((SIZE, 1), dtype=torch.float32, device=device)
        w_scale_rows = torch.empty((SIZE, 1), dtype=torch.float32, device=device)
        quantize_rows(torch, x, x_scale, 8192)
        quantize_rows(torch, w, w_scale_rows, 9600)
        w_scale = w_scale_rows.T.contiguous()
    else:
        fill_matrix(torch, x, 8192)
        fill_matrix(torch, w, 9600)
    indices, reference = sampled_reference(torch, precision, x, w, x_scale, w_scale)
    return x, w, x_scale, w_scale, indices, reference


def run_lane(torch: Any, precision: str, args: Any, out_dir: Path,
             metadata: dict[str, Any], rows: list[dict[str, Any]],
             selection: dict[str, Any] | None,
             torch_tuning: dict[str, Any] | None = None) -> None:
    device = torch.device("cuda", args.gpu)
    x, w, x_scale, w_scale, indices, reference = prepare_lane(torch, precision, args.gpu)
    try:
        calls = build_calls(torch, precision, x, w, x_scale, w_scale, selection)
    except Exception as exc:
        for candidate in candidate_names(precision):
            rows.append({"precision": precision, "candidate": candidate,
                         "status": "import_error", "error": repr(exc)})
        write_results(out_dir, metadata, rows)
        return
    valid = {}
    for candidate in candidate_names(precision):
        if torch_tuning is not None and candidate == torch_tuning.get("candidate") and \
                torch_tuning.get("status") not in ("ok", "library_dispatch"):
            rows.append({"precision": precision, "candidate": candidate,
                         "status": "tune_failed", "error": torch_tuning.get("error", torch_tuning["status"])})
            write_results(out_dir, metadata, rows)
            continue
        if candidate not in calls:
            rows.append({"precision": precision, "candidate": candidate,
                         "status": "tune_failed",
                         "error": "No verified configuration; see tuning_<precision>.json"})
            write_results(out_dir, metadata, rows)
            continue
        row = {"precision": precision, "candidate": candidate,
               "status": "pending"}
        if selection is not None and candidate in selection:
            row["tuning_spec"] = selection[candidate]
        rows.append(row)
        try:
            started = time.perf_counter()
            output = calls[candidate]()
            torch.cuda.synchronize(device)
            row["first_call_wall_ms"] = (time.perf_counter() - started) * 1000
            metrics = accuracy(torch, output, indices, reference)
            row.update(metrics)
            row["max_abs_limit"] = max(1.0, 10 * args.max_nrmse * metrics["reference_rms"])
            if metrics["nrmse"] > args.max_nrmse or metrics["max_abs"] > row["max_abs_limit"]:
                row["status"] = "accuracy_failed"
                row["error"] = (f"NRMSE {metrics['nrmse']:.6f} > {args.max_nrmse} "
                                f"or max_abs {metrics['max_abs']:.6f} > {row['max_abs_limit']:.6f}")
            else:
                row["status"] = "correct"
                valid[candidate] = calls[candidate]
        except Exception:
            row["status"] = "failed"
            row["error"] = traceback.format_exc(limit=5)
        write_results(out_dir, metadata, rows)
    for candidate, call in list(valid.items()):
        try:
            for _ in range(args.warmup):
                call()
            torch.cuda.synchronize(device)
        except Exception:
            row = next(r for r in rows if r["precision"] == precision and r["candidate"] == candidate)
            row.update(status="warmup_failed", error=traceback.format_exc(limit=5))
            del valid[candidate]
            write_results(out_dir, metadata, rows)
    timings = {name: [] for name in valid}
    for sample in range(args.samples):
        order = list(valid) if sample % 2 == 0 else list(reversed(valid))
        for candidate in order:
            try:
                timings[candidate].append(measure(torch, valid[candidate], args.repeats))
            except Exception:
                row = next(r for r in rows if r["precision"] == precision and r["candidate"] == candidate)
                row.update(status="timing_failed", error=traceback.format_exc(limit=5))
                del valid[candidate]
                del timings[candidate]
                write_results(out_dir, metadata, rows)
    for row in rows:
        candidate = row["candidate"]
        if row["precision"] == precision and candidate in timings:
            values = timings[candidate]
            row.update(status="ok", median_ms=statistics.median(values),
                       min_ms=min(values), max_ms=max(values),
                       tflops=tflops(statistics.median(values)), samples_ms=values)
    write_results(out_dir, metadata, rows)
    print(f"{precision}: " + ", ".join(
        f"{r['candidate']}={r.get('median_ms', math.nan):.3f} ms [{r['status']}]"
        for r in rows if r["precision"] == precision), flush=True)
    del x, w, x_scale, w_scale, indices, reference, calls, valid
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-nrmse", type=float, default=0.02)
    parser.add_argument("--tune-warmup", type=int, default=1)
    parser.add_argument("--tune-samples", type=int, default=5)
    parser.add_argument("--torch-tune-duration-ms", type=int, default=100)
    parser.add_argument("--torch-tune-iterations", type=int, default=100)
    parser.add_argument("--skip-tune", action="store_true",
                        help="Diagnostic only: run installed defaults, not an official race")
    args = parser.parse_args()
    if min(args.warmup, args.samples, args.repeats) < 0 or not args.samples or not args.repeats:
        parser.error("warmup >= 0 and samples/repeats > 0 are required")
    if args.max_nrmse <= 0 or args.gpu < 0:
        parser.error("max-nrmse must be positive and gpu must be nonnegative")
    if args.tune_warmup < 0 or args.tune_samples <= 0:
        parser.error("tune-warmup >= 0 and tune-samples > 0 are required")
    if args.torch_tune_duration_ms <= 0 or args.torch_tune_iterations <= 0:
        parser.error("torch tuning duration and iterations must be positive")
    contract = {"shape": SHAPE, "flops": FLOPS, "formula": "Y = X @ W.T",
                "output_dtype": "BF16", "accumulation_requested": "FP32",
                "fp8_format": "E4M3FN", "fp8_scale": "FP32 per-row for X and W",
                "scope": "single GPU, full local shape; prequantized kernel-only",
                "memory_bytes": {p: memory_bytes(p) for p in CANDIDATES},
                "candidates": {p: candidate_names(p) for p in CANDIDATES}}
    if args.dry_run:
        print(json.dumps(contract, indent=2, ensure_ascii=False))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required for a GPU run")
    import torch
    if not torch.cuda.is_available() or torch.version.hip is None:
        parser.error("AMD ROCm GPU is required (this host is not a benchmark target)")
    torch.cuda.set_device(args.gpu)
    props = torch.cuda.get_device_properties(args.gpu)
    arch = getattr(props, "gcnArchName", "")
    if not arch.startswith("gfx1201"):
        parser.error(f"expected gfx1201 GPU, got {props.name} {arch}")
    if args.output_dir.exists():
        if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
            parser.error("output directory must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import aiter
    import triton
    image_preflight = None
    source_preflight = None
    if os.getenv("AITER_MODE") == "image":
        image_preflight = verify_image_aiter(aiter, args.output_dir)
    elif os.getenv("AITER_SOURCE_ROOT"):
        source_preflight = verify_source_aiter(aiter, args.output_dir)
    metadata = {"contract": contract, "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "host": platform.node(), "gpu": props.name, "arch": arch,
                "torch": torch.__version__, "hip": torch.version.hip,
                "triton": triton.__version__, "aiter_path": str(Path(aiter.__file__).resolve()),
                "aiter_commit": source_preflight["actual_commit"] if source_preflight else git_commit(Path(aiter.__file__).resolve().parent),
                "aiter_source_preflight": source_preflight,
                "aiter_image_preflight": image_preflight,
                "aiter_mode": os.getenv("AITER_MODE", "source" if source_preflight else "installed"),
                "tune_commit": git_commit(Path(__file__).resolve().parent),
                "image": os.environ.get("EXPERIMENT_IMAGE"),
                "image_id": os.environ.get("EXPERIMENT_IMAGE_ID"),
                "auto_tune": not args.skip_tune,
                "aiter_jit_dir": os.environ.get("AITER_JIT_DIR"),
                "aiter_root_dir": os.environ.get("AITER_ROOT_DIR"),
                "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
                "parameters": vars(args) | {"output_dir": str(args.output_dir)}}
    rows: list[dict[str, Any]] = []
    write_results(args.output_dir, metadata, rows)
    selections: dict[str, dict[str, Any] | None] = {p: None for p in CANDIDATES}
    torch_tuning: dict[str, dict[str, Any]] = {}
    if not args.skip_tune:
        from tune_8192_gemm_gfx1201 import tune_lane, tune_torch_baseline

        print("Phase 1/2: tune all eligible candidates across BF16, FP16 and FP8", flush=True)
        tuning_search_complete = True
        torch_tunable_error = None
        try:
            tunable = torch.cuda.tunable
            tunable.set_filename(str(args.output_dir / "pytorch_tunableop.csv"),
                                 insert_device_ordinal=False)
            tunable.set_max_tuning_duration(args.torch_tune_duration_ms)
            tunable.set_max_tuning_iterations(args.torch_tune_iterations)
            tunable.enable(True)
            tunable.tuning_enable(False)
            if not tunable.is_enabled():
                raise RuntimeError("PyTorch TunableOp did not enable")
        except Exception:
            torch_tunable_error = traceback.format_exc(limit=5)
            tuning_search_complete = False
        for precision in CANDIDATES:
            try:
                x, w, x_scale, w_scale, indices, reference = prepare_lane(torch, precision, args.gpu)
                if torch_tunable_error is None:
                    print(f"tune {precision}/{CANDIDATES[precision][0]}: PyTorch TunableOp", flush=True)
                    torch_tuning[precision] = tune_torch_baseline(
                        torch, precision, x, w, x_scale, w_scale, indices, reference,
                        build_calls, accuracy, args,
                        args.output_dir / f"tuning_torch_{precision}.json",
                    )
                else:
                    torch_tuning[precision] = {
                        "candidate": CANDIDATES[precision][0], "status": "failed",
                        "error": torch_tunable_error}
                    (args.output_dir / f"tuning_torch_{precision}.json").write_text(
                        json.dumps(torch_tuning[precision], indent=2) + "\n", encoding="utf-8")
                selections[precision] = tune_lane(
                    torch, precision, x, w, x_scale, w_scale, indices, reference,
                    accuracy, measure, args,
                    args.output_dir / f"tuning_{precision}.json",
                )
                del x, w, x_scale, w_scale, indices, reference
                torch.cuda.empty_cache()
            except Exception:
                tuning_search_complete = False
                selections[precision] = {}
                state_path = args.output_dir / f"tuning_{precision}.json"
                state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
                    "precision": precision, "rows": [], "selected": {}}
                state.update(phase="failed", error=traceback.format_exc(limit=5))
                state_path.write_text(
                    json.dumps(state, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
        if torch_tunable_error is None:
            try:
                if tunable.tuning_is_enabled():
                    raise RuntimeError("PyTorch TunableOp tuning remained enabled after search")
            except Exception:
                tuning_search_complete = False
                metadata["torch_tuning_shutdown_error"] = traceback.format_exc(limit=5)
        metadata["tuning_search_complete"] = tuning_search_complete
        metadata["tuning_selected"] = selections
        metadata["torch_tuning"] = torch_tuning
        write_results(args.output_dir, metadata, rows)
        if not tuning_search_complete:
            print("Tuning search aborted; official race was not started. See tuning_*.json.",
                  file=sys.stderr, flush=True)
            return 1
    print("Phase 2/2: independent correctness and latency race", flush=True)
    for precision in CANDIDATES:
        try:
            run_lane(torch, precision, args, args.output_dir, metadata, rows,
                     selections[precision], torch_tuning.get(precision))
        except Exception:
            rows.append({"precision": precision, "candidate": "lane_setup",
                         "status": "failed", "error": traceback.format_exc(limit=5)})
            write_results(args.output_dir, metadata, rows)
    print(f"Results: {args.output_dir.resolve()}")
    return 0 if len(winners(rows)) == len(CANDIDATES) else 1


if __name__ == "__main__":
    sys.exit(main())
