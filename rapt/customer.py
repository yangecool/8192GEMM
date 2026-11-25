"""Customer inputs, validation and GPU-event timing for the RAPT GEMM."""
import statistics

import torch
import benchmark_8192_gemm_gfx1201 as protocol
import rapt_gemm as rapt


def contract_probes():
    # FP16 would overflow on store. BF16 represents the exact result.
    x = torch.full((256, 512), 16., device="cuda", dtype=torch.float16)
    y = rapt.linear(x, rapt.pack_weight(x))
    overflow = y.dtype == torch.bfloat16 and bool((y == 131072.).all())
    # Distinct, exactly representable scales reveal omitted or swapped axes.
    qx = torch.ones((256, 512), device="cuda").to(rapt.FP8)
    qw = torch.ones((512, 512), device="cuda").to(rapt.FP8)
    sx = 1. + torch.arange(256, device="cuda") % 3
    sw = 1. + torch.arange(512, device="cuda") % 7
    out = rapt.linear(qx, rapt.pack_weight(qw), scale_a=sx, scale_b=sw)
    ref = (512. * sx[:, None] * sw[None, :]).to(torch.bfloat16)
    return {"fp16_large_output": overflow, "fp8_independent_row_scales": torch.equal(out, ref) and out.dtype == torch.bfloat16}


def measure_lane(precision):
    x, w, sx, sw, indices, reference = protocol.prepare_lane(torch, precision, 0)
    packed = rapt.pack_weight(w)
    out = torch.empty((8192, 8192), device=x.device, dtype=torch.bfloat16)

    def gemm():
        return rapt.linear(x, packed, out=out, scale_a=sx, scale_b=sw)

    def check():
        metrics = protocol.accuracy(torch, out, indices, reference)
        metrics["all_finite"] = bool(torch.isfinite(out).all())
        max_abs = max(1., .2 * metrics["reference_rms"])
        if not (metrics["all_finite"] and metrics["nrmse"] <= .02 and metrics["max_abs"] <= max_abs):
            raise ValueError(f"Customer accuracy check failed: {metrics}")
        return metrics

    gemm()
    before = check()
    for _ in range(5):
        gemm()
    torch.cuda.synchronize()
    times = [protocol.measure(torch, gemm, 3) for _ in range(11)]
    after = check()
    latency = statistics.median(times)
    return dict(status="passed", median_ms=latency, tflops=protocol.tflops(latency), samples_ms=times, before=before, after=after)
