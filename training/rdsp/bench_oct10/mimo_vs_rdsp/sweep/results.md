# Vision sweep: MIMO's fastest vision layout (2026-10-11)

MegatronMIMO only, 15 steps each, median of steps 6–15. The 1-GPU vision rows come from the 50-step runs of a and b (same steps 6–15).

| Language grid | Vision layout | GPUs | Median step |
|---|---|---|---|
| TP2×DP2 | **1 GPU** | 5 | **8.21 s** |
| TP2×DP2 | DP2 | 6 | 8.40 s |
| TP2×DP2 | TP2 | 6 | 8.69 s |
| TP2×DP2 | TP2×DP2 | 8 | 8.79 s |
| TP2×PP2×DP1 | **1 GPU** | 5 | **10.00 s** |
| TP2×PP2×DP1 | TP2 | 6 | 11.32 s |
| TP2×PP2×DP1 | TP4 | 8 | 12.59 s |

More vision GPUs made MIMO slower in every case, so its best vision layout is 1 GPU for both language grids. That makes c the same as a, and d the same as b; they were not run separately.

Logs: `9b_<vision>_<language>.log`; the picks: `best_<language>.txt`.
