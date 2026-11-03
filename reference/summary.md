# 8192 GEMM 算子比赛

单卡、本地完整 8192³；GPU event 时间；输入已量化/就绪，未计量化与预打包。

| 输入 | 候选 | 状态 | 中位数 ms | TFLOP/s | NRMSE |
|---|---|---|---:|---:|---:|
| bf16 | torch_mm | ok | 14.9927 | 73.3362 | 0.0001 |
| bf16 | aiter_triton | ok | 16.6977 | 65.8481 | 0.0001 |
| bf16 | aiter_gluon | ok | 16.6133 | 66.1826 | 0.0001 |
| fp16 | torch_mm_then_bf16 | ok | 16.5624 | 66.3860 | 0.0014 |
| fp16 | aiter_triton | ok | 17.1791 | 64.0028 | 0.0002 |
| fp8 | torch_scaled_mm | ok | 11.8609 | 92.7008 | 0.0001 |
| fp8 | aiter_triton | ok | 8.7010 | 126.3662 | 0.0001 |
| fp8 | aiter_ck | ok | 9.8448 | 111.6849 | 0.0001 |
| fp8 | aiter_gfx1201_mm8 | ok | 8.3210 | 132.1369 | 0.0001 |

各精度胜者：bf16: torch_mm；fp16: torch_mm_then_bf16；fp8: aiter_gfx1201_mm8
