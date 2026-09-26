"""Direct stage-to-stage tensor exchange for stage-local dispatch.

Each rank joins two process groups spanning every rank of every stage: one
carries activations downstream, the other gradients upstream. They are
standalone groups built from a TCPStore, so the default torch.distributed
world stays the stage-local world DeepSpeed owns.

Why two groups: NCCL runs a communicator's point-to-point ops in issue order,
so with one shared group a stage's gradient receive could queue behind its
own activation send (and its neighbour's behind theirs) and deadlock. One
direction per group keeps each group's dependencies acyclic. Sends are async
and kept alive until the step ends; receives block, and 1F1B order
guarantees each one's matching send is eventually issued.

Microbatches are equal-sized, so each (direction, peer) pair sends its shape
header once per step; both ends walk microbatches in the same order, so they
agree on which message carries it.
"""

import datetime

import torch
import torch.distributed as dist

_DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.float64, torch.int64]
_HEADER_LEN = 10  # dtype code, ndim, up to 8 dims


def _encode(t: torch.Tensor) -> torch.Tensor:
    header = [_DTYPES.index(t.dtype), t.dim(), *t.shape]
    return torch.tensor(header + [0] * (_HEADER_LEN - len(header)), dtype=torch.int64)


def _decode(header: torch.Tensor) -> tuple[torch.dtype, tuple[int, ...]]:
    vals = header.tolist()
    return _DTYPES[vals[0]], tuple(vals[2:2 + vals[1]])


class PipelineP2P:
    """One rank's two cross-stage groups, plus per-step bookkeeping."""

    def __init__(self, store, rank: int, world: int, epoch: int, timeout_s: float,
                 device: torch.device):
        self.rank, self.world, self.device = rank, world, device
        timeout = datetime.timedelta(seconds=timeout_s)
        self.fwd = self._group(dist.PrefixStore(f"e{epoch}/fwd", store), timeout)
        self.bwd = self._group(dist.PrefixStore(f"e{epoch}/bwd", store), timeout)
        self._pending = []       # (work, tensor) sends in flight this step
        self._shapes = {}        # (direction, peer) -> (dtype, shape) received this step
        self._announced = set()  # (direction, peer) already sent a header this step

    def _group(self, store, timeout):
        if self.device.type == "cuda":
            opts = dist.ProcessGroupNCCL.Options()
            opts._timeout = timeout
            return dist.ProcessGroupNCCL(store, self.rank, self.world, opts)
        return dist.ProcessGroupGloo(store, self.rank, self.world, timeout)

    def abort(self) -> None:
        """Tear down both groups now. After a failed step an unmatched send can
        sit on an NCCL stream; aborting releases it before the watchdog would
        kill the process."""
        self._pending.clear()
        for pg in (self.fwd, self.bwd):
            for name in ("abort", "shutdown"):
                fn = getattr(pg, name, None)
                if fn is not None:
                    try:
                        fn()
                        break
                    except Exception:
                        pass

    # -- per step --------------------------------------------------------------

    def begin_step(self) -> None:
        self._shapes.clear()
        self._announced.clear()

    def end_step(self) -> None:
        """Wait for every send of this step, so nothing crosses into the next."""
        pending, self._pending = self._pending, []
        for work, _ in pending:
            work.wait()

    # -- transfers -------------------------------------------------------------

    def _pg(self, direction: str):
        return self.fwd if direction == "fwd" else self.bwd

    def send(self, direction: str, t: torch.Tensor, peer: int, mb: int) -> None:
        pg = self._pg(direction)
        t = t.detach().contiguous()
        if (direction, peer) not in self._announced:
            self._announced.add((direction, peer))
            header = _encode(t).to(self.device)
            self._pending.append((pg.send([header], peer, 2 * mb), header))
        self._pending.append((pg.send([t], peer, 2 * mb + 1), t))

    def recv(self, direction: str, peer: int, mb: int) -> torch.Tensor:
        pg = self._pg(direction)
        key = (direction, peer)
        if key not in self._shapes:
            header = torch.empty(_HEADER_LEN, dtype=torch.int64, device=self.device)
            pg.recv([header], peer, 2 * mb).wait()
            self._shapes[key] = _decode(header.cpu())
        dtype, shape = self._shapes[key]
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        pg.recv([buf], peer, 2 * mb + 1).wait()
        return buf


def make_store(host: str, port: int, world: int, is_master: bool, timeout_s: float):
    return dist.TCPStore(host, port, world, is_master,
                         timeout=datetime.timedelta(seconds=timeout_s),
                         wait_for_workers=False)
