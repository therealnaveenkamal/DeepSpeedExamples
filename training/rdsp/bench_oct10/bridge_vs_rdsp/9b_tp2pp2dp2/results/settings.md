# 9B TP2×PP2×DP2: settings

Same as `../../9b_tp4pp2dp1/results/settings.md` (model, data, 64 rows per step, every row padded to 2,048, optimizer, precision, loss, no recompute, eager vision encoder in rdsp), except:

| Setting | Megatron-Bridge | rdsp |
|---|---|---|
| Layout | TP2, PP2, DP2 (`runs.sh shared-megatron`) | 2 stages of TP2 × DP2 |
| Microbatches | 32, 1 row per DP copy | 32, 2 rows (1 per DP copy) |
| Layer split | even 16 \| 16 | `--cuts balanced` |
| Optimizer state | split over the 2 DP copies (distributed optimizer) | split over the 2 DP copies (ZeRO-1) |
| Padding to 2,048 | forced by Bridge under PP | `--pad-multiple 0` |
