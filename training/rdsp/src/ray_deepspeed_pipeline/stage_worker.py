"""The Ray actor behind one rank of one pipeline stage.

Each actor owns one GPU, one DeepSpeed engine (through DeepSpeedStageAdapter)
and its links to neighbouring stages. A training step arrives as one call,
run_step(), carrying the rank's whole 1F1B op list; activations and gradients
then move rank to rank over the p2p groups, never through the driver.
"""

import os
import socket

import torch

from ray_deepspeed_pipeline import checkpoint as ckpt
from ray_deepspeed_pipeline.boundary import (
    Grid,
    assemble,
    gradient_scale,
    p2p_destinations,
    p2p_sources,
    rank_cell,
)
from ray_deepspeed_pipeline.deepspeed_adapter import DeepSpeedStageAdapter, observed_mesh
from ray_deepspeed_pipeline.errors import ValidationError


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class StageWorkerActor:
    """One rank of one stage-local world. Construction is cheap; bootstrap()
    joins the stage world once every rank exists."""

    def __init__(self, stage_module, ds_config_json: str, *, stage: int, rank: int,
                 grid: Grid, backend: str, n_microbatches: int,
                 is_first: bool, is_last: bool, loss_fn=None, engine_factory=None):
        self.stage, self.rank, self.grid = stage, rank, grid
        self.p2p = None
        self._p2p_store = None
        self.cell = rank_cell(grid, rank)
        self._init = dict(stage_module=stage_module, ds_config_json=ds_config_json,
                          backend=backend, n_microbatches=n_microbatches,
                          is_first=is_first, is_last=is_last, loss_fn=loss_fn,
                          engine_factory=engine_factory)
        self.adapter = None
        self.executed = []  # command ids in execution order
        self.generation = None

    def address(self) -> tuple[str, int, str]:
        """(host, free port, node id) for this stage's rendezvous; called on rank 0."""
        import ray
        return (ray.util.get_node_ip_address(), _free_port(),
                ray.get_runtime_context().get_node_id())

    def bootstrap(self, master_addr: str, master_port: int) -> dict | None:
        """Join the stage world and build the engine; returns observed_mesh()."""
        init = self._init
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = "0"  # Ray sets CUDA_VISIBLE_DEVICES per actor
        os.environ["WORLD_SIZE"] = str(self.grid.world)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        module = init["stage_module"]
        if init["backend"] == "nccl":
            torch.cuda.set_device(0)
            module = module.to("cuda")
        torch.distributed.init_process_group(
            init["backend"], rank=self.rank, world_size=self.grid.world,
            init_method=f"tcp://{master_addr}:{master_port}")
        self.adapter = DeepSpeedStageAdapter(
            module, init["ds_config_json"], init["n_microbatches"],
            init["is_first"], init["is_last"], loss_fn=init["loss_fn"],
            engine_factory=init["engine_factory"])
        self._init = None  # drop the driver-built module copy
        return observed_mesh(self.adapter.engine)

    def ping(self) -> bool:
        return True

    def has_loss_fn(self) -> bool:
        return self.adapter.loss_fn is not None

    def executed_commands(self):
        return list(self.executed)

    def named_parameters_numpy(self):
        return self.adapter.named_parameters_numpy()

    def _enter_generation(self, generation, *, begin: bool = True) -> None:
        # a new generation discards whatever an abandoned one left behind
        if generation == self.generation:
            return
        self.generation = generation
        if begin:
            self.adapter.begin_generation()

    def _forward(self, kind: str, mb: int, x, labels):
        offset = self.cell.sp_index * x.shape[1] if self.grid.sp > 1 else None
        run = self.adapter.forward if kind == "forward" else self.adapter.eval_forward
        return run(mb, x, labels=labels, position_offset=offset)

    def execute(self, command_id: str, kind: str, generation=None, control=None):
        """Run one control command: apply, save or load (verify or load)."""
        self.executed.append(command_id)
        self._enter_generation(generation, begin=kind != "load")
        if kind == "apply":
            return self.adapter.apply()
        if kind == "save":
            return self._save(control["root"], control["tag"])
        if kind == "load" and "verify" in control:
            return self._verify(control["root"], control["tag"], control["verify"])
        if kind == "load":
            return self._load(control["root"], control["tag"],
                              control.get("load_optimizer_states", True),
                              control.get("load_lr_scheduler_states", True))
        raise ValidationError(f"unknown command kind {kind!r}")

    # -- the step ----------------------------------------------------------------

    def init_p2p(self, host: str, port: int, epoch: int, grids: list,
                 timeout_s: float = 300.0) -> bool:
        """Join fresh cross-stage groups for `epoch` and warm up every
        neighbour link, so no connection setup happens inside a step."""
        from ray_deepspeed_pipeline.p2p import PipelineP2P, make_store
        grids = [Grid(*grid) for grid in grids]
        offsets = [0]
        for grid in grids:
            offsets.append(offsets[-1] + grid.world)
        self._grids, self._offsets = grids, offsets
        global_rank = offsets[self.stage] + self.rank
        if self._p2p_store is None:
            self._p2p_store = make_store(host, port, offsets[-1], global_rank == 0, timeout_s)
        if self.p2p is not None:
            # release unmatched sends of the failed step before NCCL's watchdog does
            self.p2p.abort()
        self.p2p = PipelineP2P(self._p2p_store, global_rank, offsets[-1], epoch, timeout_s,
                               self.adapter.device)
        self._warm_up_links()
        return True

    def _peers(self):
        """(prev, next) neighbour links in global ranks; None at a pipeline end.
        prev = (activation senders as (rank, cell), gradient receivers);
        next = (activation receivers, gradient senders as (rank, cell))."""
        stage, grids, offsets = self.stage, self._grids, self._offsets
        prev = next_ = None
        if stage > 0:
            up, base = grids[stage - 1], offsets[stage - 1]
            prev = ([(base + r, cell) for r, cell in p2p_sources(up, self.grid, self.rank)],
                    [base + r for r in p2p_destinations(self.grid, up, self.rank)])
        if stage < len(grids) - 1:
            down, base = grids[stage + 1], offsets[stage + 1]
            next_ = ([base + r for r in p2p_destinations(self.grid, down, self.rank)],
                     [(base + r, cell) for r, cell in p2p_sources(down, self.grid, self.rank)])
        return prev, next_

    def _warm_up_links(self):
        prev, next_ = self._peers()
        dummy = torch.zeros(1, device=self.adapter.device)
        self.p2p.begin_step()
        if next_:
            for peer in next_[0]:
                self.p2p.send("fwd", dummy, peer, 0)
        if prev:
            for peer, _ in prev[0]:
                self.p2p.recv("fwd", peer, 0)
            for peer in prev[1]:
                self.p2p.send("bwd", dummy, peer, 0)
        if next_:
            for peer, _ in next_[1]:
                self.p2p.recv("bwd", peer, 0)
        self.p2p.end_step()

    def run_step(self, generation: int, ops: list, train: bool, inputs=None,
                 labels=None):
        """Run this rank's whole step. ops: [(kind, mb, command_id)] in 1F1B
        order. inputs / labels: this rank's cell of every microbatch (first /
        last stage only)."""
        self._enter_generation(generation)
        prev, next_ = self._peers()
        adapter, p2p = self.adapter, self.p2p
        # DeepSpeed averages gradients over DP and sums over SP, so the gradient
        # entering this stage must be dp_s x dL/d(output): scale by dp_s/dp_{s+1}
        scale = gradient_scale(self._grids[self.stage + 1].dp, self.grid.dp) if next_ else 1.0
        losses = {}
        p2p.begin_step()
        try:
            ready = True
            for kind, mb, command_id in ops:
                self.executed.append(command_id)
                if kind == "ready":
                    ready = adapter.ready()
                    continue
                if kind in ("forward", "eval"):
                    if prev:
                        x = assemble([(cell, p2p.recv("fwd", peer, mb))
                                      for peer, cell in prev[0]], self.cell)
                    else:
                        x = inputs[mb]
                    mb_labels = labels[mb] if labels is not None else None
                    out = self._forward(kind, mb, x, mb_labels)
                    if next_:
                        for peer in next_[0]:
                            p2p.send("fwd", out, peer, mb)
                    else:
                        losses[mb] = out
                else:  # backward
                    grad = None
                    if next_:
                        grad = assemble([(cell, p2p.recv("bwd", peer, mb))
                                         for peer, cell in next_[1]], self.cell)
                        if scale != 1.0:
                            grad = grad * scale
                    input_grad = adapter.backward(mb, grad=grad)
                    if prev:
                        for peer in prev[1]:
                            p2p.send("bwd", input_grad, peer, mb)
            p2p.end_step()
        finally:
            p2p.begin_step()  # drop per-step shape records either way
        return {"losses": [losses[k] for k in sorted(losses)] if adapter.is_last else None,
                "ready": ready}

    # -- checkpoints -------------------------------------------------------------

    def _save(self, root: str, tag: str) -> dict:
        stage_dir = ckpt.stage_dir(root, tag, self.stage)
        self.adapter.save_shard(stage_dir, tag)
        self.adapter.save_rng(ckpt.rng_path(root, tag, self.stage, self.rank))
        # every rank's files are complete before rank 0 fingerprints the stage
        torch.distributed.barrier()
        record = {"rank": self.rank}
        if self.rank == 0:
            record["files"] = ckpt.digest_tree(stage_dir)
        return record

    def _verify(self, root: str, tag: str, files_by_stage: dict):
        """Rank 0 re-checks this stage's shard digests on its own node.
        Returns True or the rejection text; never raises, so the reason
        reaches the driver without a transport-specific wrapper."""
        if self.rank != 0:
            return True
        files = files_by_stage.get(self.stage, files_by_stage.get(str(self.stage)))
        try:
            ckpt.verify_stage_files(ckpt.stage_dir(root, tag, self.stage),
                                    self.stage, files)
        except Exception as e:
            return str(e)
        return True

    def _load(self, root: str, tag: str, load_optimizer_states: bool,
              load_lr_scheduler_states: bool) -> bool:
        self.adapter.load_shard(
            ckpt.stage_dir(root, tag, self.stage), tag,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states)
        self.adapter.load_rng(ckpt.rng_path(root, tag, self.stage, self.rank))
        return True
