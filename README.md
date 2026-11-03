# HVAT R9600D 8192 GEMM reproduction

This package reproduces the test procedure behind the HVAT results cited in the AMD email:

| Input | Selected implementation | Median latency | Effective throughput |
|---|---|---:|---:|
| BF16 | PyTorch `torch.mm` | 14.992744 ms | 73.336248 TFLOP/s |
| FP16 | PyTorch `torch.mm`, followed by BF16 conversion | 16.562397 ms | 66.386021 TFLOP/s |
| FP8 E4M3FN | AITER gfx1201 `mm8_out` | 8.321006 ms | 132.136871 TFLOP/s |

These measurements were collected on one **AMD Radeon AI PRO R9600D (gfx1201)** on 16 October 2026. The package contains the original benchmark and tuning scripts from aiter-tune commit `b2be4a659f73c46d1346d5cd531eab710a1ae686`, copied without changes. The original result JSON and source checksums accompany the scripts.

## Image

The standalone environment is:

```text
compiler.hvat-ai.com:8001/rocm/gemm-base:rocm10.0.0_py3.14_pytorch2.12.0_aiter0.1.22_gfx1201-20261103
```

| Component | Version / identity |
|---|---|
| Python | 3.14.7 |
| PyTorch | `2.12.0+rocm10.0.0` |
| AMD Triton distribution | `3.8.0+git4cff872c.rocm10.0.0` |
| AITER | `0.1.22+amd.gfx1201.r9600.rocm10.gfc1a037a6f` |
| AITER source revision | `fc1a037a6fb90235f6c947226e781b2a3fea685b` |

Built on CKS2U, this image uses the original ROCm/PyTorch base with the same AITER version as the measured image. AITER installs its dependencies automatically. The image retains the compiler and SDK needed for JIT compilation; it does not require the LightX2V application or model weights.

The recorded scores came from the original image:

```text
compiler.hvat-ai.com:8001/lightx2vrocm/lightx2v-rocm:rocm10.0.0_py3.14_pytorch2.12.0_amd-gfx1201-r9600-bf16-r16-20261015
sha256:f315a34ab1c06f177211fdbd9bcf7e89b13d812bf8566ea7c1ae19a08977191e
```

The new GEMM base provides the matching compute stack for reproduction. Its validation record is in `evidence/validation.json`; the historical scores remain attributed to the original image.

## Quick start

Use a Linux x86-64 host with an operational AMD driver, Docker, and an available gfx1201 GPU. Registry access is required to pull the image; an image can also be supplied through `docker save` / `docker load`. Host Python and an aiter-tune checkout are not needed.

```bash
git clone https://github.com/yangecool/8192GEMM.git
cd 8192GEMM
sha256sum --quiet -c SHA256SUMS
docker pull compiler.hvat-ai.com:8001/rocm/gemm-base:rocm10.0.0_py3.14_pytorch2.12.0_aiter0.1.22_gfx1201-20261103
bash run.sh 0
```

Alternatively, extract the supplied `hvat-r9600-gemm8192-repro-20261103.tar.gz`, enter that directory, and run the same checksum, image-pull and test commands. This route does not require Git on the host.

`0` selects the physical GPU exposed through `HIP_VISIBLE_DEVICES`; the process uses logical device 0 inside its container. Run with the selected GPU available for the test. The launcher records the actual model, so runs on an R9700 are identifiable separately.

The script performs the original tuning stage, then independently measures all nine candidates. It uses a read-only, network-disabled container and writable result/cache mounts. The first run compiles any required JIT kernels before timing.

The final lines identify the result directory, its `.tar.gz` archive and the exit status. Return the complete archive, including `benchmark/results.json`, tuning records, image inspection and `reproduction-check.json`.

## Workload

`Y = X @ W.T`, with M=N=K=8192 and contiguous row-major X and W. Output is BF16; FP32 accumulation is the specified compute requirement. FP8 uses E4M3FN and FP32 per-row scales on both operands. Each GPU executes the whole GEMM independently.

The original input generator uses uniform values in **[-1, 1)** with seeds 8192 and 9600. FP8 quantization occurs before timing. The measured FP16 path writes an intermediate FP16 result and then converts it to BF16. The FP8 path applies both row scales before BF16 rounding. These details matter when comparing with a `hipblas-bench` result.

The timing procedure, numerical checks and interpretation of results are in [TEST_GUIDE.md](TEST_GUIDE.md).

## Files

- `run.sh`: standalone container launcher and result packaging.
- `vendor/`: original benchmark, tuning script and container helper, unchanged.
- `reference/`: original measurements and image identity.
- `verify_run.py`: measurement completeness and numerical checks.
- `PROVENANCE.json`, `evidence/`: source lineage and validation record.
