# 8192³ GEMM on AMD Radeon AI PRO R9600D

The default benchmark runs the customer workload through AMD's RAPT v1.1 shared library. Each selected GPU computes the full `Y = X @ W.T` with M = N = K = 8192. BF16, FP16 and FP8 E4M3 inputs use FP32 accumulation and write BF16 directly. A second entry runs AMD's Quick benchmark with its own inputs and timing.

## Run on one GPU

Use a Linux host with an R9600D, the amdgpu driver, Docker, and Python 3.8 or newer. The runner reads the card's hotspot sensor and gives each container one render node. GPU index 0 selects `/dev/dri/renderD128`, index 1 selects `renderD129`, and so on.

```bash
git clone --recurse-submodules https://github.com/yangecool/8192GEMM.git
cd 8192GEMM
docker pull rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0

./run.sh 0                          # Customer BF16, FP16 and per-row FP8
./run.sh 0 --method quick           # AMD Quick, all four modes
./run.sh 0 --case customer-fp8      # One precision
```

For an existing clone, run `git pull --ff-only` followed by `git submodule update --init`. The submodule pins the upstream wrapper, shared library and Quick script to commit `81d5d928`. The launcher verifies all three files against [their SHA-256 digests](rapt/RAPT.sha256). [Upstream RAPT release](https://github.com/jundali77/RAPT-Kernels/tree/81d5d9282c8ed447a207b7d62bfc9871d72a7237/Benchmarks/BF16-FP16-FP8%20GEMM).

## Measurement protocols

| | Customer (default) | AMD Quick |
|---|---|---|
| Input | Uniform in [−1, 1), with seeds 8192 for X and 9600 for W | Upstream `torch.rand` in [0, 1), seed 0 |
| FP8 | Independently quantize each row of X and W to E4M3 using its FP32 absmax divided by 448, with a scale floor of 1e−8. The kernel applies both scales before the BF16 store. | Cast random values to E4M3. Row mode supplies independent random scales in [0.5, 1.5). Tensor mode uses 0.75 for X and 1.25 for W. |
| Timed work | Packed weights, prepared inputs, and a preallocated BF16 output. Five warm-ups, then 11 HIP-event samples of three GEMMs each. Use the median time per GEMM. | Original `bench.py` defaults. Each round runs 50 warm-ups and 200 timed calls per implementation. Two rounds alternate RAPT and torch order, with zero gap. Use the mean of the two RAPT times. |
| Validation | Compare 16 full output rows to a sampled FP32 reference rounded to BF16 and check the entire output is finite, before and after timing. Additional probes exercise FP16 input producing 131072 in BF16 and independent FP8 row scales. | Original full-output `torch.allclose` against the BF16-output torch reference. The launcher requires an explicit upstream `PASSED` before accepting a result. |

Each score is `2 × 8192³ / (time in seconds) / 10¹²` TFLOP/s. Input preparation, weight packing and correctness checks run outside the event timer. The customer script reuses this repository's original input, reference and timing helpers. [RAPT customer benchmark](rapt/customer.py) · [Single-case runner](rapt/case.py)

## Reproduce the four-image, eight-card tables

Preload the four ROCm PyTorch images listed in [rapt/images.txt](rapt/images.txt). The second field records each image ID used for the published measurements. Set `IMAGE_REPO` to use a local mirror with the same tags. For a single image, set `IMAGE` to the complete image name.

```bash
while read -r tag image_id; do docker pull "rocm/pytorch:$tag"; done < rapt/images.txt
./run.sh 0,1,2,3,4,5,6,7 --matrix --dry-run
./run.sh 0,1,2,3,4,5,6,7 --matrix
```

Start with idle cards and run one launcher on the host. The matrix covers eight cards, four images and seven cases (224 single-GPU measurements). The launcher interleaves cards, images and precisions and waits for each container to finish. Each card rests for at least 60 seconds between cases. Its starting hotspot must be at most 30 °C and within 4 °C above its recorded idle baseline. A failed check stops the run before aggregation.

The tested ROCm 10.2.0 image reports the R9600D as “R9600 Series”. For that exact image ID and PCI board revision, the launcher mounts a temporary, checksum-verified libdrm copy with the R9600D product name. Each run records the applicable image ID and name correction in `provenance.json`. All four images load the same verified RAPT binary.

Results go under `results/rapt-*`, or a new directory specified by `RESULT_DIR`. Each case preserves its GPU UUID, image ID, starting temperature, raw timings and validation results. Quick cases also retain upstream stdout. In `summary.json` each image and case has all individual GPU scores, their median, the maximum and the card indices attaining it. Table cells use the median of the unrounded single-GPU scores.

## Deprecated PyTorch / AITER benchmark

The original nine-candidate method, reference results, and validation records remain available for historical reproduction. Select it explicitly with `./run.sh --legacy 0`. Its FP16 path rounds to FP16 before converting to BF16. [Original procedure and results](LEGACY.md) · [Historical test guide](TEST_GUIDE.md)
