# Deprecated HVAT R9600 8192 GEMM reproduction

The current RAPT benchmark is documented in [README.md](README.md).

This package reproduces the test procedure behind the HVAT results cited in the AMD email:

| Input | Selected implementation | Median latency | Effective throughput |
|---|---|---:|---:|
| BF16 | PyTorch `torch.mm` | 14.992744 ms | 73.336248 TFLOP/s |
| FP16 | PyTorch `torch.mm`, followed by BF16 conversion | 16.562397 ms | 66.386021 TFLOP/s |
| FP8 E4M3FN | AITER gfx1201 `mm8_out` | 8.321006 ms | 132.136871 TFLOP/s |

These measurements were collected on one **AMD Radeon AI PRO R9600 (gfx1201)** on 16 October 2026. The package contains the original benchmark and tuning scripts from aiter-tune commit `b2be4a659f73c46d1346d5cd531eab710a1ae686`, copied without changes. Reference measurements and source checksums accompany the scripts.

## Image

The standalone environment is:

```text
misyangekkk9/hvatgemmbase:rocm10.0.0_py3.14_pytorch2.12.0_aiter0.1.22_gfx1201-20261103
```

| Component | Version / identity |
|---|---|
| Python | 3.14.7 |
| PyTorch | `2.12.0+rocm10.0.0` |
| AMD Triton distribution | `3.8.0+git4cff872c.rocm10.0.0` |
| AITER | `0.1.22+amd.gfx1201.r9600.rocm10.gfc1a037a6f` |
| AITER source revision | `fc1a037a6fb90235f6c947226e781b2a3fea685b` |

The image provides the PyTorch and AITER versions used for the reference measurements, together with the dependencies and compiler tools required by the scripts. No additional Python packages or model weights are needed. Image identities and validation records are available in `PROVENANCE.json` and `evidence/`.

## Quick start

Use a Linux x86-64 host with an operational AMD driver, Docker, and an available gfx1201 GPU. Pull the image from Docker Hub, or transfer it with `docker save` / `docker load`. Host Python and an aiter-tune checkout are not needed.

```bash
git clone https://github.com/yangecool/8192GEMM.git
cd 8192GEMM
sha256sum --quiet -c SHA256SUMS
docker pull misyangekkk9/hvatgemmbase:rocm10.0.0_py3.14_pytorch2.12.0_aiter0.1.22_gfx1201-20261103
bash run.sh --legacy 0
```

For the original standalone archive, use its original command `bash run.sh 0`. Extract the supplied `hvat-r9600-gemm8192-repro-20261103.tar.gz`, enter that directory, and use the checksum and image-pull commands above. This route does not require Git on the host.

`0` selects the physical GPU exposed through `HIP_VISIBLE_DEVICES`; the process uses logical device 0 inside its container. Run with the selected GPU available for the test. The launcher records the actual model, so runs on an R9700 are identifiable separately.

The legacy script performs the original tuning stage, then independently measures all nine candidates. It uses a read-only, network-disabled container and writable result/cache mounts. The first run compiles any required JIT kernels before timing.

The final lines identify the result directory, its `.tar.gz` archive and the exit status. Return the complete archive, including `benchmark/results.json`, tuning records, image inspection and `reproduction-check.json`.

## Workload

`Y = X @ W.T`, with M=N=K=8192 and contiguous row-major X and W. Output is BF16; FP32 accumulation is the specified compute requirement. FP8 uses E4M3FN and FP32 per-row scales on both operands. Each GPU executes the whole GEMM independently.

The original input generator uses uniform values in **[-1, 1)** with seeds 8192 and 9600. FP8 quantization occurs before timing. The measured FP16 path writes an intermediate FP16 result and then converts it to BF16. The FP8 path applies both row scales before BF16 rounding. These details matter when comparing with a `hipblas-bench` result.

The timing procedure, numerical checks and interpretation of results are in [TEST_GUIDE.md](TEST_GUIDE.md).

## Files

- `run_legacy.sh`: standalone container launcher and result packaging.
- `vendor/`: original benchmark, tuning script and container helper, unchanged.
- `reference/`: original measurements and image identity.
- `verify_run.py`: measurement completeness and numerical checks.
- `PROVENANCE.json`, `evidence/`: source lineage and validation record.
