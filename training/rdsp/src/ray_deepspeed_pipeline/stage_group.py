"""Driver-side handles to the stage actor groups, and their start-up.

create_stage_clients() places each stage on one node, starts one
StageWorkerActor per GPU, forms every stage's DeepSpeed world and opens the
cross-stage p2p groups. The coordinator talks to each stage through a
StageGroupClient; Ray ObjectRefs never leave this module unresolved.
"""

import os

import torch

from ray_deepspeed_pipeline.boundary import (
    Grid,
    cell_slice,
    rank_cell,
    rank_coords,
    representatives,
    slice_inputs,
)
from ray_deepspeed_pipeline.errors import ValidationError
from ray_deepspeed_pipeline.partition import select_stage_builder
from ray_deepspeed_pipeline.protocols import Command
from ray_deepspeed_pipeline.stage_worker import StageWorkerActor

_PLACEMENT_TIMEOUT_S = float(os.environ.get("RDSP_PLACEMENT_TIMEOUT_S", "600"))


def _p2p_timeout_s() -> float:
    """How long a stage waits on a neighbour before the step fails."""
    return float(os.environ.get("RDSP_P2P_TIMEOUT_S", "300"))


def _alive_timeout_s() -> float:
    return float(os.environ.get("RDSP_ALIVE_TIMEOUT_S", "60"))


class _GroupHandle:
    """Future-like handle for protocols.resolve(): resolves every rank of a
    stage and reduces to one stage-level value. The raw ObjectRefs never leave
    the runtime layer."""

    def __init__(self, refs, reduce):
        self.refs = refs
        self.reduce = reduce

    def result(self):
        import ray
        return self.reduce(ray.get(self.refs))


def _all_ok(results):
    """True if every rank succeeded; a rank's rejection text wins."""
    for result in results:
        if isinstance(result, str):
            return result
    return all(result is not False for result in results)


def _mean(values):
    return sum(float(v) for v in values) / len(values)


class StageGroupClient:
    """Driver-side client for one stage's actors. Every command goes to every
    rank in the same order, so the ranks' collectives line up."""

    def __init__(self, actors, stage: int = 0, grid: Grid | None = None,
                 is_last: bool = False, placement_group=None):
        self.actors = actors
        self.stage = stage
        self.grid = grid or Grid(dp=len(actors))
        self.is_last = is_last
        self.placement_group = placement_group
        self.node_id = None  # set at bootstrap; rebuilds prefer this node

    def submit(self, command: Command, *, inputs=None, labels=None, control=None):
        """Send a command to every rank; returns a handle resolving to one
        stage-level result."""
        if command.kind in ("step", "eval_step"):
            return self._submit_step(command, inputs, labels, control)
        refs = [actor.execute.remote(command.command_id, command.kind,
                                     generation=command.generation, control=control)
                for actor in self.actors]
        if command.kind == "save":
            return _GroupHandle(refs, self._stage_record)
        return _GroupHandle(refs, _all_ok)

    def _submit_step(self, command, inputs, labels, control):
        """One call per rank carrying the rank's whole op list for the step."""
        refs = []
        for rank, actor in enumerate(self.actors):
            cell = rank_cell(self.grid, rank)
            refs.append(actor.run_step.remote(
                command.generation, control["ops"], command.kind == "step",
                inputs=[slice_inputs(x, cell) for x in inputs] if inputs is not None else None,
                labels=[cell_slice(y, cell) for y in labels] if labels is not None else None))
        reps = representatives(self.grid)

        def reduce(results):
            out = {"ready": all(res["ready"] for res in results)}
            if self.is_last:  # per microbatch: mean over the data-parallel cells
                per_rep = [results[r]["losses"] for r in reps]
                out["losses"] = [_mean(v) for v in zip(*per_rep)]
            return out
        return _GroupHandle(refs, reduce)

    def init_p2p(self, host: str, port: int, epoch: int, grids: list):
        return [actor.init_p2p.remote(host, port, epoch, grids, _p2p_timeout_s())
                for actor in self.actors]

    @staticmethod
    def reconnect_p2p(clients, epoch: int) -> None:
        connect_p2p(clients, epoch)

    def _stage_record(self, rank_records):
        files = next(r["files"] for r in rank_records if r["rank"] == 0)
        return {"index": self.stage, "world_size": len(self.actors),
                "ranks": [{"rank": r["rank"]} for r in rank_records],
                "files": files}

    def alive(self, timeout: float | None = None) -> bool:
        import ray
        timeout = _alive_timeout_s() if timeout is None else timeout
        try:
            ray.get([actor.ping.remote() for actor in self.actors], timeout=timeout)
            return True
        except Exception:
            return False

    def shutdown(self) -> None:
        """Best effort: kill the actors and release the placement group."""
        import ray
        for actor in self.actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        if self.placement_group is not None:
            try:
                ray.util.remove_placement_group(self.placement_group)
            except Exception:
                pass


def check_mesh(stage: int, grid: Grid, observed: list[dict | None]) -> None:
    """Check that DeepSpeed placed every rank where the plan's grid says, with
    each cell covered exactly once. Ranks reporting None (non-DeepSpeed stub
    engines) are checked only for count."""
    if len(observed) != grid.world:
        raise ValidationError(f"stage {stage}: {len(observed)} ranks bootstrapped, "
                              f"grid needs {grid.world}")
    seen = set()
    for rank, mesh in enumerate(observed):
        if mesh is None:
            continue
        d, q, t = rank_coords(grid, rank)
        expected = {"dp_rank": d, "dp_world": grid.dp, "tp_rank": t,
                    "tp_world": grid.tp, "sp_rank": q, "sp_world": grid.sp}
        for key, want in expected.items():
            got = mesh.get(key)
            if got is not None and got != want:
                raise ValidationError(
                    f"stage {stage} rank {rank}: DeepSpeed places it at "
                    f"{key}={got}, the plan's grid expects {want} (mesh mismatch)")
        coords = (mesh.get("dp_rank", d), mesh.get("sp_rank", q), mesh.get("tp_rank", t))
        if coords in seen:
            raise ValidationError(f"stage {stage}: duplicate rank at grid position {coords}")
        seen.add(coords)


def connect_p2p(clients, epoch: int) -> None:
    """(Re)build the cross-stage groups on every rank of every stage. The
    TCPStore lives in stage 0 rank 0; each epoch uses fresh groups under a new
    key prefix, because a failed step leaves the old ones unusable."""
    import ray
    if epoch == 0:
        host, port, _ = ray.get(clients[0].actors[0].address.remote())
        for client in clients:
            client.p2p_address = (host, port)
    host, port = clients[0].p2p_address
    grids = [(client.grid.dp, client.grid.sp, client.grid.tp) for client in clients]
    ray.get([ref for client in clients for ref in client.init_p2p(host, port, epoch, grids)])


def create_stage_clients(model, plan, loss_fn, *, engine_factory=None,
                         use_gpu: bool | None = None, stage_builder=None,
                         prefer_nodes: list | None = None, weights: str | None = None):
    """Start one actor group per plan stage and return their clients.

    Needs an attached Ray context. Each stage gets its own STRICT_PACK
    placement group (all its GPUs on one node, so TP/EP/SP collectives stay on
    the node's interconnect and its checkpoint shards land on one disk), its
    own rendezvous and its own stage-local world. prefer_nodes: per-stage node
    ids to reuse on a rebuild, so node-local shards stay reachable after an
    actor (not node) failure. weights: HF checkpoint dir each stage loads its
    meta-device parameters from. On any failure every started stage is shut
    down.
    """
    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    if not ray.is_initialized():
        raise ValidationError(
            "no Ray context: attach with ray.init(...) before rdsp.initialize() "
            "(the package uses the current context and never creates one)")
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    backend = "nccl" if use_gpu else "gloo"
    actor_cls = ray.remote(StageWorkerActor)

    clients = []
    n_stages = len(plan.stages)
    # chosen from the model itself, so rdsp.initialize() (which passes no
    # builder) and recovery rebuilds get the right one too
    builder = stage_builder or select_stage_builder(model)
    try:
        for spec in plan.stages:
            grid = Grid(dp=spec.dp, sp=spec.sp, tp=spec.tp)
            module = builder(model, spec.block_start, spec.block_stop,
                             spec.parameter_names)
            is_last = spec.index == n_stages - 1
            pg = None
            options = {"num_gpus": 1 if use_gpu else 0}
            if use_gpu:
                strategy = "STRICT_PACK"
                target = prefer_nodes[spec.index] if prefer_nodes else None
                pg = placement_group([{"GPU": 1, "CPU": 1}] * spec.num_gpus,
                                     strategy=strategy, _soft_target_node_id=target)
                try:
                    ray.get(pg.ready(), timeout=_PLACEMENT_TIMEOUT_S)
                except Exception as e:
                    ray.util.remove_placement_group(pg)
                    raise ValidationError(
                        f"stage {spec.index}: cannot place {spec.num_gpus} GPUs "
                        f"({strategy}) in this Ray cluster: {e}") from e
            module_ref = ray.put(module)
            actors = []
            for rank in range(spec.num_gpus):
                rank_options = dict(options)
                if pg is not None:
                    rank_options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=rank)
                actors.append(actor_cls.options(**rank_options).remote(
                    module_ref, spec.ds_config_json,
                    stage=spec.index, rank=rank, grid=grid,
                    backend=backend, n_microbatches=plan.global_microbatches,
                    is_first=spec.index == 0, is_last=is_last,
                    loss_fn=loss_fn if is_last else None,
                    engine_factory=engine_factory, weights=weights))
            clients.append(StageGroupClient(actors, stage=spec.index, grid=grid,
                                            is_last=is_last, placement_group=pg))

        # bootstrap every stage's world concurrently
        boots = []
        for client in clients:
            addr, port, client.node_id = ray.get(client.actors[0].address.remote())
            boots.append([actor.bootstrap.remote(addr, port) for actor in client.actors])
        for client, refs in zip(clients, boots):
            check_mesh(client.stage, client.grid, ray.get(refs))
        connect_p2p(clients, epoch=0)
    except Exception:
        for client in clients:
            client.shutdown()
        raise
    return clients
