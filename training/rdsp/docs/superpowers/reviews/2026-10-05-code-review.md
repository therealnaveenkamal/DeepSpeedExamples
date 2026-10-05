# rdsp code review (332a673..HEAD)

Two axes, read-only sub-agents. Standards: `docs/CODING_STANDARDS.md` + smell baseline. Spec: `docs/ENGINEERING.md`, README promises.

## Spec axis

| # | Finding | Where | Verdict |
|---|---|---|---|
| S1 | Prefetch: read-ahead step survives `load_checkpoint` and failed steps; every later `train_batch` raises "another iterator" | coordinator.py:74,195-202,265 | fixed |
| S2 | Prefetch with the engine-owned loader: checkpoint counts the read-ahead step; `eval_batch` reads the same iterator as the reader thread | data.py:47, engine.py:71, coordinator.py:205 | fixed |
| S3 | README: colocated encoder state "replicated", code splits it | README | fix in Task 8/9 |
| S4 | ENGINEERING "never had to patch": adapter patches one DeepSpeed optimizer method, writes private fields in `reset()` | deepspeed_adapter.py:135-150,470-480 | fix in Task 9 (honest edges) |
| S5 | README engine overrides list incomplete (`immediate_grad_update`, `train_batch_size`) | README | fix in Task 8 |
| S6 | Explicit `gradient_clipping` rejected only when stages > 1; one stage forces 0 silently | compiler.py:69,171 | fixed |
| S7 | Plan hash covers runtime-only settings; toggling prefetch/compile/recompute rejects a valid checkpoint | plan.py, compiler.py | fixed |
| S8 | README `train_vl` example assumes untied embeddings | README | fix in Task 8 |
| S9 | Undocumented features: prefetch, colocated compile / per-microbatch encoding / uneven share, compile + compile_vision, split-vocabulary loss, `TokenMeanLoss.per_microbatch`, per-microbatch shapes, `vision_token_ratio`, freed bf16 grads, profiling env vars, `train_vl` flags | — | fix in Tasks 3, 5, 8, 9 |
| S10 | Vision-injection check is a heuristic (`deepstack_visual_indexes`) | compiler.py:138-144 | skip: documented as a limit in CONTRACTS |

## Standards axis

| # | Finding | Where | Verdict |
|---|---|---|---|
| T1 | Project-history references (phase ids, plan section, "rerun after fix", dated evidence) | support_matrix.py | fixed |
| T2 | "v1" wording, `UnsupportedInV1` | api.py, compiler.py, config.py, data.py | skip: names the API version, not history; renaming a public error type is churn |
| T3 | Built-in exceptions / asserts for user errors | deepspeed_adapter.py:106,276,446,456; p2p.py:43,187; coordinator.py:64 | fixed |
| T4 | Options that do nothing: `CheckpointPolicy.save_optimizer_state`, `ConnectionOverride.buffer_limit` | config.py:53,113 | fixed |
| T5 | Internal plan fields never read (`process_group_tag`, `FailureSpec`, `ScheduleSpec`) | plan.py | skip: internal plan schema, part of the hash, no user surface |
| T6 | Unreachable: `p2p.send_sized`/`recv_sized`, `ColocatedVisionEngine.save` | p2p.py:142-158, vision.py:386 | fixed |
| T7 | Test-only seams (`protocols.Coordinator`, `COMMAND_KINDS`, `vision.waiting()`, `has_loss_fn`) | — | skip: test seams |
| T8 | Missing public exports `StageOverride`, `ConnectionOverride`, `CheckpointPolicy` | __init__.py | fix in Task 3 |
| T9 | Vision settings shipped as a positional 4-tuple | stage_group.py:272, stage_worker.py:146 | fixed |
| T10 | `per_microbatch` means three things | config.py:104, train_vl.py:121 | fixed |
| T11 | `vision_config` lookup duplicated 4× | partition.py, compiler.py | fixed |
| T12 | `train_vl` loss sums duplicate `next_token_loss_sum`; pad round-up written twice | train_vl.py:137-176 | fixed |
| T13 | Example reaches `engine._coordinator._plan` | train_vl.py:376 | fixed |
| T14 | `import os` inside `main` | train_vl.py:360 | fixed |
| T15 | Tuple-shaped `route_images` / `_peers` results; long `_run_step` | vision.py, stage_worker.py | skip: internal; refactor without GPU validation is risk with no user gain |
| T16 | TP style table in two modules | partition.py:478, deepspeed_adapter.py:19 | skip: partition validates names without importing DeepSpeed |
| T17 | Profiler path hard-coded under /tmp | stage_worker.py:280 | skip: debug-only env var, documented |
| T18 | Middle man `StageGroupClient.reconnect_p2p` | stage_group.py:154 | skip: coordinator's seam for fake workers |
