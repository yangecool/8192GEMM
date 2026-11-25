"""Run isolated 8192³ RAPT benchmarks in local ROCm PyTorch images."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UPSTREAM = ROOT / "third_party/RAPT-Kernels/Benchmarks/BF16-FP16-FP8 GEMM"
IMAGES = dict(line.split() for line in (HERE / "images.txt").read_text().splitlines())
CASES = ("quick-fp8-tensor", "customer-bf16", "quick-fp16", "customer-fp8",
         "quick-bf16", "customer-fp16", "quick-fp8-row")
LIBDRM = "/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/rocm_sysdeps/lib/librocm_sysdeps_drm_amdgpu.so.1"
IMAGE_102 = IMAGES["rocm10.2.0_ubuntu24.04_py3.12_pytorch_release_2.15.0"]
RAPT_COMMIT = "81d5d9282c8ed447a207b7d62bfc9871d72a7237"


def arguments():
    p = argparse.ArgumentParser(description=__doc__, epilog="Deprecated PyTorch/AITER entry: bash run.sh --legacy GPU")
    p.add_argument("gpus", nargs="?", default="0", help="comma-separated render-node indices (0 = renderD128)")
    p.add_argument("--matrix", action="store_true", help="use the four recorded image tags and both methods")
    select = p.add_mutually_exclusive_group()
    select.add_argument("--method", choices=["customer", "quick", "all"], help="run all cases for a method")
    select.add_argument("--case", choices=CASES, help="run a single case in each selected image")
    p.add_argument("--dry-run", action="store_true", default=os.getenv("DRY_RUN") == "1", help="print the plan offline")
    args = p.parse_args()
    try:
        cards = [int(x) for x in args.gpus.split(",")]
    except ValueError:
        p.error("select numeric GPU indices, e.g. 0 or 0,1,2,3")
    if min(cards) < 0 or len(set(cards)) != len(cards):
        p.error("GPU indices must be distinct and nonnegative")
    prefix = os.getenv("IMAGE_REPO", "rocm/pytorch")
    tags = list(IMAGES)
    if args.matrix:
        if os.getenv("IMAGE"):
            p.error("use IMAGE_REPO to select a registry for --matrix")
        images = [f"{prefix}:{tag}" for tag in tags]
    else:
        images = [os.getenv("IMAGE", f"{prefix}:{tags[1]}")]
    method = args.method or ("all" if args.matrix else "customer")
    cases = [m for m in CASES if method == "all" or m.startswith(method + "-")]
    return cards, images, [args.case] if args.case else cases, args.dry_run


def plan(cards, images, cases):
    # Interleave versions and precisions, then rotate the starting point by card.
    pairs = [(image, case) for case in cases for image in images]
    if len(images) == 4 and len(cases) == 7:
        pairs = [(images[(i + 3) % 4], cases[i % 7]) for i in range(28)]
    for turn in range(len(pairs)):
        for index, gpu in enumerate(cards):
            offset = 10 if index == 7 else index
            image, case = pairs[(turn + offset) % len(pairs)]
            yield gpu, image, case


def write_json(path, data):
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(data, indent=2) + "\n")
    partial.replace(path)


def verified_sources():
    digests = {}
    for line in (HERE / "RAPT.sha256").read_text().splitlines():
        expected, name = line.split()
        path = UPSTREAM / name
        if not path.is_file():
            sys.exit("Initialize RAPT with: git submodule update --init third_party/RAPT-Kernels")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            sys.exit(f"RAPT checksum mismatch: {path}")
        digests[name] = expected
    return digests


def libdrm_name_copy(image_id, scratch):
    # Precisely the tested ROCm 10.2 runtime and string table; mounted read-only.
    data = subprocess.check_output(["docker", "run", "--rm", "--network=none", "--entrypoint=cat", image_id, LIBDRM])
    old = b"AMD Radeon AI PRO R9600 Series\0"
    new = b"AMD Radeon AI PRO R9600D\0"
    if hashlib.sha256(data).hexdigest() != "a5c477a608e41e02b4172e275551ef423fbb0228f0ccfe7a321d88abac48fa60":
        sys.exit("Unexpected libdrm build in ROCm 10.2")
    corrected = data.replace(old, new.ljust(len(old), b"\0"), 1)
    if hashlib.sha256(corrected).hexdigest() != "64a88de62af602dac3a15709e4e5e129260f90bccb074e2915ad08717df8a795":
        sys.exit("Unexpected R9600D name table")
    path = scratch / "libdrm-r9600d"
    path.write_bytes(corrected)
    return path


def docker_args(gpu, image_id, alias):
    node = Path(f"/dev/dri/renderD{128 + gpu}")
    board = (Path("/sys/class/drm") / node.name / "device").resolve(strict=True)
    cmd = ["docker", "run", "--rm", "--network=none", "--device=/dev/kfd", f"--device={node}",
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
           "-e", "HIP_VISIBLE_DEVICES=0", "-e", f"PCI={board.name}", "-e", "PYTHONDONTWRITEBYTECODE=1",
           "-e", "PYTHONPATH=/upstream:/protocol", "-v", f"{UPSTREAM}:/upstream:ro",
           "-v", f"{HERE}:/runner:ro", "-v", f"{ROOT}/vendor/gemm8192:/protocol:ro", "--mount", f"type=bind,src={board},dst=/board,readonly"]
    for gid in {node.stat().st_gid, Path("/dev/kfd").stat().st_gid}:
        cmd.extend(["--group-add", str(gid)])
    if image_id == IMAGE_102:
        identity = tuple((board / k).read_text().strip() for k in
                         ("vendor", "device", "revision", "subsystem_vendor", "subsystem_device"))
        if identity != ("0x1002", "0x7551", "0xc8", "0x1002", "0x0e11"):
            sys.exit(f"Unknown board identity for the ROCm 10.2 name correction: {identity}")
        cmd.extend(["-v", f"{alias}:{LIBDRM}:ro"])
    return cmd


def run(cards, images, tasks, dest, scratch, digests):
    image_ids = {img: subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", img], text=True).strip() for img in images}
    alias = libdrm_name_copy(IMAGE_102, scratch) if IMAGE_102 in image_ids.values() else None
    commands = {(g, img): docker_args(g, image_ids[img], alias) for g in cards for img in images}
    package = dict(rapt_commit=RAPT_COMMIT, sha256=digests, images=image_ids,
                   libdrm_name_correction=[img for img in images if image_ids[img] == IMAGE_102])
    write_json(dest / "provenance.json", package)
    last = {}
    boards = {}
    for gpu in cards:
        image = images[0]
        cmd = commands[gpu, image] + ["--entrypoint=python3", image_ids[image], "/runner/board.py"]
        boards[gpu] = json.loads(subprocess.check_output(cmd, text=True))
        last[gpu] = time.monotonic()
    if len({b["uuid"] for b in boards.values()}) != len(cards):
        raise RuntimeError("Two GPU indices selected the same physical card")
    write_json(dest / "baselines.json", boards)
    results = defaultdict(dict)
    for index, (gpu, image, case) in enumerate(tasks, 1):
        folder = dest / f"{index:03}-gpu{gpu}-{case}"
        folder.mkdir()
        board = boards[gpu]
        cmd = commands[gpu, image] + ["-v", f"{folder}:/out", "--entrypoint=python3", image_ids[image], "/runner/case.py"]
        time.sleep(max(0, last[gpu] + 60 - time.monotonic()))
        deadline = time.monotonic() + 600
        while True:
            config = dict(case=case, gpu=gpu, uuid=board["uuid"], image=image, image_id=image_ids[image],
                          max_start_C=min(30, board["baseline_C"] + 4), idle_s=time.monotonic() - last[gpu])
            write_json(folder / "config.json", config)
            with (folder / "run.log").open("a") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode != 75:
                if proc.returncode:
                    sys.exit(f"{case} failed (exit {proc.returncode}); see {folder}/run.log")
                break
            if time.monotonic() >= deadline:
                sys.exit(f"GPU {gpu} did not reach {config['max_start_C']} °C within ten minutes")
            time.sleep(22)
        last[gpu] = time.monotonic()
        value = json.loads((folder / "case.json").read_text())["result"]["tflops"]
        results[image, case][gpu] = value
        print(f"[{index}/{len(tasks)}] GPU {gpu} {image} {case}: {value:.3f} TFLOP/s", flush=True)
    summary = []
    for (image, case), scores in results.items():
        best = max(scores.values())
        summary.append(dict(image=image, case=case, tflops_by_gpu=scores, median_tflops=statistics.median(scores.values()),
                            max_tflops=best, max_gpu_ids=[g for g, value in scores.items() if value == best]))
    write_json(dest / "summary.json", summary)


def main():
    cards, images, cases, dry_run = arguments()
    tasks = list(plan(cards, images, cases))
    if dry_run:
        print(json.dumps(tasks, indent=2))
        return
    digests = verified_sources()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = Path(os.getenv("RESULT_DIR", ROOT / "results" / f"rapt-{stamp}-{os.getpid()}")).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    print(f"Results: {dest}", flush=True)
    write_json(dest / "plan.json", tasks)
    with tempfile.TemporaryDirectory() as scratch:
        run(cards, images, tasks, dest, Path(scratch), digests)


if __name__ == "__main__":
    main()
