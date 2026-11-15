# 9B results, 2026-10-11

Median step time over steps 6–50; tok/s/GPU = real tokens per second per GPU.

| Run | GPUs | Megatron step | rdsp step | Result | Megatron tok/s/GPU | rdsp tok/s/GPU |
|---|---|---|---|---|---|---|
| Megatron-Bridge: TP4 × PP2 × DP1 | 8 | 18.24 s | 14.63 s | rdsp 20% faster | 445 | 555 |
| Megatron-Bridge: TP2 × PP2 × DP2 | 8 | 9.09 s | 7.62 s | rdsp 16% faster | 894 | 1,065 |
| Megatron-Bridge: TP2 × DP4, no pipeline | 8 | 6.45 s | 5.84 s | rdsp 10% faster | 1,258 | 1,390 |
| MegatronMIMO: vision 1 GPU + TP2 × PP2 | 5 | 9.52 s | 9.11 s | rdsp 4% faster | 1,364 | 1,426 |
| MegatronMIMO: vision 1 GPU + TP2 × DP2 | 5 | 8.11 s | 8.22 s | about even | 1,603 | 1,580 |
