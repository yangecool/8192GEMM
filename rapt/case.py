"""Execute one customer or upstream Quick case on the selected card."""
import contextlib
import io
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import bench
import board
import customer


def quick(precision, output):
    dtype, _, scaling = precision.partition("-")
    arguments = ["bench.py", "--dtypes", dtype]
    if scaling:
        arguments += ["--fp8-scales", scaling]
    times = []
    original_timer = bench.time_fn

    def record(fn, warmup, iters, gap):
        ms = original_timer(fn, warmup, iters, gap)
        times.append(ms)
        return ms

    log = io.StringIO()
    bench.time_fn = record
    previous_args = sys.argv
    try:
        sys.argv = arguments
        with contextlib.redirect_stdout(log):
            bench.main()
    finally:
        bench.time_fn = original_timer
        sys.argv = previous_args
        (output / "upstream.log").write_text(log.getvalue())

    row, = [line for line in log.getvalue().splitlines() if line.lstrip().startswith("8192x8192x8192 ")]
    shape_dtype, rapt, _, _, check, _ = (field.strip() for field in row.split("|"))
    if check != "PASSED" or shape_dtype.split()[-1] != dtype or len(times) != 4:
        raise ValueError(f"Unexpected upstream result: {row}")
    # The two rounds time RAPT, torch, torch, RAPT in that order.
    ms = statistics.mean((times[0], times[3]))
    throughput = customer.protocol.tflops(ms)
    if rapt.split() != [f"{ms:.3f}", f"{throughput:.1f}"]:
        raise ValueError(f"Upstream timing disagrees with the recorded events: {row}")
    return dict(status="passed", command=arguments, mean_ms=ms, tflops=throughput, round_times_ms=times)


def main():
    output = Path("/out")
    config = json.loads((output / "config.json").read_text())
    card = board.identify()
    if card["uuid"] != config["uuid"] or not customer.rapt.device_supported("cuda:0"):
        raise RuntimeError(f"Unexpected or unsupported GPU: {card}")
    for _ in range(3):
        try:
            start_temp = board.temperature()
        except OSError:
            board.identify()
        else:
            if start_temp <= config["max_start_C"]:
                break
        time.sleep(0.3)
    else:
        return 75  # Retry after the card has cooled.
    family, precision = config["case"].split("-", 1)
    start = time.time()
    if family == "customer":
        torch.set_float32_matmul_precision("highest")
        probes = customer.contract_probes()
        if not all(probes.values()):
            raise ValueError(f"RAPT contract checks failed: {probes}")
        result = dict(**customer.measure_lane(precision), contract_probes=probes)
    else:
        result = quick(precision, output)
    data = dict(config, card=card, prestart_C=start_temp, started=start,
                finished=time.time(), result=result)
    (output / "case.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"{config['case']}: {result['tflops']:.3f} TFLOP/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
