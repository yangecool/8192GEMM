# Test guide

## 1. Reference and measurement boundary

The reference archive is `8192-gemm-20261016T062511Z-2376509.tar.gz`, SHA-256 `637d143efd497b439a3491f2c42e1abf68c3a749928b56f599fae926c0432a0e`. `reference/results.json` is copied from that archive. All three scores in the README belong to that same run and image.

| Property | Setting |
|---|---|
| Shape | M=N=K=8192 |
| Operation | X[M,K] × W[N,K]ᵀ → Y[M,N] |
| Memory layout | Contiguous row-major X, W and Y; W.T is a view |
| Input types | BF16, FP16, FP8 E4M3FN; matched types for X/W |
| Accumulation requirement | FP32 |
| Output | BF16 |
| FP8 scales | One FP32 scale per X row and W row |
| Bias / activation | None |
| Parallelism | One complete GEMM per GPU |
| Timed work | GEMM, output conversion and FP8 row scaling |
| Preparation | Inputs, quantization, allocations, JIT and tuning completed before measurement |

BF16/FP16 input tensors are filled in 128-row chunks using `uniform_(-1, 1)` in their input dtype. FP8 inputs start from FP32 uniform values; each row uses `scale = max(amax(abs(row))/448, 1e-8)`, followed by clamping to [-448,448] and conversion to E4M3FN. Seeds are 8192 for X and 9600 for W. The benchmark is not using the default [0,1) distribution of `torch.rand`.

The FP16 PyTorch path writes an FP16 intermediate output, then performs a timed copy/conversion to BF16. The original benchmark retains PyTorch's backend defaults; see the accumulation qualification in section 3. The AITER FP8 path applies X and W scales in its epilogue. Its fixed parameters are BM=128, BN=256, BK=64, GROUP=16, num_warps=8 and num_stages=2, with one full K reduction.

For comparison with `hipblas-bench`, align input/compute/output types, transposes and leading dimensions, row-scale handling, output conversion, input distribution and the timing boundary. The supplied scripts are the reference for these settings; a raw GEMM with a different epilogue measures a different path.

## 2. Original tuning and timing procedure

The original scripts are preserved in `vendor/gemm8192/`. Phase 1 attempts PyTorch TunableOp selection and searches the AITER candidates for all three precisions. PyTorch tuning is then disabled. The reference run returned `library_dispatch` for PyTorch and did not export a fixed BLAS algorithm ID.

The measured candidates are:

| Input | Candidates |
|---|---|
| BF16 | PyTorch mm; AITER Triton; AITER gfx1201 Gluon |
| FP16 | PyTorch mm plus BF16 conversion; AITER Triton |
| FP8 | PyTorch scaled mm; AITER Triton; AITER CK; AITER gfx1201 mm8_out |

Tuning uses one warmup and five timing samples per explicit candidate. PyTorch TunableOp limits are 100 ms and 100 iterations. AITER searches tile/warp/stage choices and split-K configurations; Gluon searches its five variants and M-group settings; CK enumerates its kernel catalog. The fixed `mm8_out` implementation is recorded as `tunable=false`.

Phase 2 checks outputs, performs five warmup calls per valid candidate, and collects **11 GPU-event samples**, with **three GEMM calls per sample**. Candidate order alternates forward/reverse between samples. Each sample is divided by three to obtain time per GEMM; the reported latency is the median of all 11 samples. Search minima and first-call JIT wall time are not used as the reported score. Keeping all nine candidates preserves this original interleaving procedure.

Each GEMM performs `2 × 8192³ = 1,099,511,627,776` FLOPs. Effective throughput is `1099.511627776 / latency_ms` TFLOP/s. This is throughput for the measured path, including the documented epilogue work.

## 3. Numerical checks

Sixteen evenly spaced output rows are checked against a full-K FP32 reference rounded to BF16. The original gate requires finite sampled output, NRMSE ≤ 0.02, and maximum absolute error ≤ `max(1, 0.2 × reference_RMS)`. This is the original sampled operator check, not a full-output proof or a model-quality metric.

For historical fidelity, the benchmark preserves the original image's PyTorch backend defaults; it does not add new reduction settings or change tolerances. FP32 is the stated accumulation requirement. The original result archive did not contain a BLAS instruction audit; numerical acceptance alone does not establish the instructions executed by a library kernel. AITER's FP8 implementation uses an FP32 accumulator and FP32 epilogue scales. An additional ISA audit should be reported separately from this historical reproduction.

## 4. Running and checking output

Preview the Docker command without running a GPU test:

```bash
DRY_RUN=1 bash run.sh 0
```

Choose a fresh output directory or another physical GPU:

```bash
RESULT_DIR="$PWD/results/r9600-repeat-01" bash run.sh 1
```

Use the original measured image through the same standalone launcher:

```bash
IMAGE=compiler.hvat-ai.com:8001/lightx2vrocm/lightx2v-rocm:rocm10.0.0_py3.14_pytorch2.12.0_amd-gfx1201-r9600-bf16-r16-20261015 \
  bash run.sh 0
```

The launcher verifies the package checksums. Before tuning, the original benchmark checks the AITER version and records the installed module paths. No host AITER checkout or separate dependency installation is needed.

| File | Purpose |
|---|---|
| `benchmark/results.json` / `results.csv` / `summary.md` | Raw per-candidate samples, numerical metrics and medians |
| `benchmark/tuning_*.json` | Tuning attempts and selected configurations |
| `benchmark/aiter-image-preflight.json` | AITER paths, version and source hashes |
| `image-inspect.json` | Actual image identity and configuration |
| `reproduction-check.json` | Complete candidate coverage and comparison with the three cited HVAT paths |
| `package-provenance.json`, `package-sha256sums.txt` | Original source and archive lineage |
| `docker.log`, `exit-status.txt` | Execution log and final status |

Exit 0 means the full measurement and numerical checks completed. A changed score is reported as measured; matching the historical latency is not an exit condition. GPU frequency, power limits, driver and system activity can affect timing. Preserve the actual GPU model and all raw samples when comparing runs. The new package is intentionally runnable without Git metadata, so the original driver's `tune_commit` may be null; `package-provenance.json` identifies its exact source revision.

## 5. Image preparation

The GEMM base was built on CKS2U from the original ROCm/PyTorch base:

```text
compiler.hvat-ai.com:8001/lightx2vrocm/lightx2v-base:rocm10.0.0_py3.14_pytorch2.12.0-r3
```

PyTorch is retained at `2.12.0+rocm10.0.0`. AITER is installed at `0.1.22+amd.gfx1201.r9600.rocm10.gfc1a037a6f`, with dependencies resolved by its installer. The delivered image is ready to run the supplied scripts.

For offline transport, use `docker save` on a host with the image and `docker load` on the test host. Registry credentials are not included in this package.

## 6. Provenance and validation scope

`PROVENANCE.json` identifies the original source revision and result-archive/member hashes. `evidence/new-image.json` records the new image; `evidence/validation.json` lists the checks actually executed for this delivery. Build, import and dry-run checks do not constitute a new GPU performance measurement. GPU scores in `reference/results.json` remain the historical R9600D measurements.
