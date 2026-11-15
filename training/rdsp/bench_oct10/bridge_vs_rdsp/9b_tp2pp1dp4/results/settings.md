# 9B TP2×PP1×DP4 (no pipeline): settings

Same as `../../9b_tp4pp2dp1/results/settings.md` (model, data, 64 rows per step, every row padded to 2,048, optimizer, precision, loss, no recompute, eager vision encoder in rdsp), except:

| Setting | Megatron-Bridge | rdsp |
|---|---|---|
| Layout | TP2, DP4, no pipeline (`runs.sh tp2dp4-megatron`) | 1 stage of TP2 × DP4 |
| Microbatches | 16, 1 row per DP copy | 16, 4 rows (1 per DP copy) |
| Layer split | none (one stage) | none (one stage) |
| Optimizer state | split over the 4 DP copies (distributed optimizer) | split over the 4 DP copies (ZeRO-1) |
| Padding to 2,048 | `dataset.pad_to_max_length=true` (without a pipeline Bridge does not force it) | `--pad-multiple 0` |
