"""Colocated vision: the vision encoder runs on every rank of every stage.

The pipeline's stages hold only the language model. Each step, every rank
encodes a share of the step's images with its own copy of the encoder, as one
batch; the features go to the first stage's ranks whose rows hold those
images, and stand in for the pixels there. After the pipeline's backward the
first stage returns each image's feature gradient to the rank that encoded
it, every rank backpropagates through its encoder, and the encoder gradients
are summed over all ranks. The encoder thus runs data parallel over the whole
pipeline instead of on the first stage's GPUs only.

Images are routed per row: a first-stage input gives `pixel_values` and
`image_grid_thw` as per-row lists (boundary.slice_inputs). An image here is
a row's images together.
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from ray_deepspeed_pipeline.boundary import Grid, rank_coords
from ray_deepspeed_pipeline.errors import StepFailed, ValidationError

# attribute names Hugging Face vision-language models give their vision encoder
_ENCODER_NAMES = ("visual", "vision_tower", "vision_model")


def find_vision_encoder(model: nn.Module) -> str:
    """Module path of the model's vision encoder (e.g. "model.visual")."""
    for name, module in model.named_modules():
        has_params = next(module.parameters(), None) is not None
        if name.rsplit(".", 1)[-1] in _ENCODER_NAMES and has_params:
            return name
    raise ValidationError(
        f"{type(model).__name__} has no vision encoder (a module named one of "
        f"{_ENCODER_NAMES}); colocated vision needs one")


class VisionTower(nn.Module):
    """One rank's copy of the vision encoder. `path` is where the encoder
    sits in the full model, which names its tensors in the checkpoint."""

    def __init__(self, module: nn.Module, path: str):
        super().__init__()
        self.module = module
        self.path = path

    def checkpoint_key(self, name: str) -> str:
        return self.path + name.removeprefix("module")

    def local_blocks(self):
        """The encoder's block lists' blocks (recompute wraps them)."""
        blocks = []
        for module in self.module.modules():
            if (isinstance(module, nn.ModuleList) and len(module) >= 2
                    and len({type(b) for b in module}) == 1):
                blocks.extend(module)
        return blocks

    def forward(self, pixel_values, grid_thw):
        out = self.module(pixel_values, grid_thw=grid_thw, return_dict=True)
        return out.pooler_output  # merged: one feature per image token


def build_vision_tower(model: nn.Module, path: str) -> VisionTower:
    return VisionTower(copy.deepcopy(model.get_submodule(path)), path)


class PrecomputedVision(nn.Module):
    """Stands in for the vision encoder on the first stage: the model's own
    code calls it with `pixel_values`, which there already holds the encoded
    features, and gets them back as the encoder's output."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.spatial_merge_size = getattr(encoder, "spatial_merge_size", None)
        self._dtype = next(encoder.parameters()).dtype

    @property
    def dtype(self):
        return self._dtype

    def forward(self, pixel_values, grid_thw=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        return BaseModelOutputWithPooling(pooler_output=pixel_values)


# --- routing -------------------------------------------------------------------

@dataclass(frozen=True)
class VisionLayout:
    """first: the first stage's rank grid; world: ranks over all stages
    (global rank = stage offset + stage-local rank, the first stage first)."""

    first: Grid
    world: int


# share of a fair split each first-stage cell encodes itself: enough to start
# the pipeline at once, while the other stages encode the rest in its fill time
_FIRST_STAGE_SHARE = 0.5


def _owners(images: list, layout: VisionLayout, cell_rows: int, cell_ranks) -> dict:
    """image -> encoding rank. Each first-stage cell's TP rank 0 encodes the
    cell's first images (the ones its first microbatches need, so the
    pipeline starts without waiting); the remaining images, in microbatch
    order, are dealt in contiguous runs to the ranks of the other stages,
    which are idle while the pipeline fills, the last stage's ranks first."""
    # last stage first: it finishes its pipeline work first, and the first
    # microbatches' feature gradients are the first to come back
    others = list(range(layout.world - 1, layout.first.world - 1, -1))
    own_first = int(len(images) / layout.world * _FIRST_STAGE_SHARE) if others else len(images)
    owner, taken, rest = {}, {}, []
    for image, _, row in images:
        rep = cell_ranks(row)[0]
        if taken.get(rep, 0) < max(own_first, 1 if not others else 0):
            owner[image] = rep
            taken[rep] = taken.get(rep, 0) + 1
        else:
            rest.append(image)
    for i, image in enumerate(rest):
        owner[image] = others[i * len(others) // len(rest)]
    return owner


def route_images(inputs: list, layout: VisionLayout, rank: int) -> dict:
    """What global `rank` does with the step's images. inputs: the step's
    first-stage inputs, one per microbatch. An image is a (microbatch, row)
    with any image, numbered microbatch-major (see _owners for who encodes
    which). Returns
      own:  [(image, pixel_values, grid_thw, destinations, gradient source)]
            to encode; destinations are the first-stage ranks of the row's
            cell (every TP rank), the gradient comes from its TP rank 0
      need: [(microbatch, image, owner)] features this first-stage rank
            receives, in row order."""
    images = []  # (image, microbatch, row)
    for mb, entry in enumerate(inputs):
        pixels, grids = entry.get("pixel_values"), entry.get("image_grid_thw")
        if not isinstance(pixels, list) or not isinstance(grids, list):
            raise StepFailed("colocated vision needs pixel_values and image_grid_thw "
                             "given per row (lists)")
        rows = len(pixels)
        images += [(mb * rows + row, mb, row) for row in range(rows) if grids[row].numel()]
    first = layout.first
    cell_rows = rows // first.dp

    def cell_ranks(row):
        dp_index = row // cell_rows
        return [r for r in range(first.world) if rank_coords(first, r)[0] == dp_index]

    owner = _owners(images, layout, cell_rows, cell_ranks)

    own, need = [], []
    for image, mb, row in images:
        if owner[image] == rank:
            dests = cell_ranks(row)
            own.append((image, inputs[mb]["pixel_values"][row], inputs[mb]["image_grid_thw"][row],
                        dests, dests[0]))
        if rank in cell_ranks(row):
            need.append((mb, image, owner[image]))
    return {"own": own, "need": need}


# --- the per-rank engine ---------------------------------------------------------

VISION_OPTIMIZERS = ("adam", "adamw", "sgd")


def _optimizer(params, ds_config: dict):
    """The DeepSpeed config's optimizer as a torch optimizer (Adam, AdamW or
    SGD; DeepSpeed's Adam is AdamW unless adam_w_mode is off), plus its LR
    scheduler as DeepSpeed would build it."""
    opt = ds_config.get("optimizer", {})
    kind, args = opt.get("type", "AdamW"), dict(opt.get("params", {}))
    if kind.lower() not in VISION_OPTIMIZERS or args.pop("adam_w_mode", True) is False:
        raise ValidationError(f"colocated vision supports Adam, AdamW or SGD, got {kind!r}")
    args.pop("torch_adam", None)
    optimizer = (torch.optim.SGD if kind.lower() == "sgd" else torch.optim.AdamW)(params, **args)
    scheduler = None
    if "scheduler" in ds_config:
        from deepspeed.runtime import lr_schedules
        sched = ds_config["scheduler"]
        scheduler = getattr(lr_schedules, sched["type"])(optimizer, **sched.get("params", {}))
    return optimizer, scheduler


class ColocatedVisionEngine:
    """One rank's vision encoder with its own optimizer. forward() encodes
    this rank's images for the whole step; backward() takes their feature
    gradients; reduce_gradients() sums the encoder gradients over all ranks;
    apply() steps the optimizer with the pipeline's."""

    def __init__(self, tower: VisionTower, ds_config: dict, device: torch.device):
        self.tower = tower.to(device)
        self.device = device
        bf16 = ds_config.get("bf16", {}).get("enabled", False)
        self.autocast = torch.bfloat16 if bf16 else None
        self.optimizer, self.scheduler = _optimizer(self.tower.parameters(), ds_config)
        self._outputs = {}  # image -> features with their graph (training)

    def forward(self, images: list, train: bool) -> dict:
        """images: [(image, pixel_values, grid_thw)] -> {image: features}."""
        if not images:
            return {}
        pixels = torch.cat([p for _, p, _ in images]).to(self.device)
        grids = torch.cat([g for _, _, g in images]).to(self.device)
        merge = getattr(self.tower.module, "spatial_merge_size", 1) ** 2
        sizes = [int(g.prod(-1).sum()) // merge for _, _, g in images]
        with torch.set_grad_enabled(train), torch.autocast(
                self.device.type, dtype=self.autocast, enabled=self.autocast is not None):
            features = self.tower(pixels, grids)
        out = dict(zip((i for i, _, _ in images), features.split(sizes)))
        if train:
            self._outputs = out
        return {i: f.detach() for i, f in out.items()}

    def backward(self, grads: dict) -> None:
        if not grads:
            return
        images = sorted(grads)
        torch.autograd.backward([self._outputs[i] for i in images],
                                [grads[i].to(self._outputs[i].dtype) for i in images])
        self._outputs = {}

    def reduce_gradients(self, group, divide_by: float) -> None:
        """Sum over `group` (a raw process group over all ranks; None: this
        rank alone), then divide: the first stage's feature gradients carry
        its data-parallel degree (boundary.gradient_scale)."""
        params = [p for p in self.tower.parameters() if p.requires_grad]
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in params])
        if group is not None:
            group.allreduce([flat]).wait()
        flat /= divide_by
        offset = 0
        for p in params:
            p.grad = flat[offset:offset + p.numel()].view_as(p).clone()
            offset += p.numel()

    def apply(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()

    def reset(self) -> None:
        """Drop an abandoned step's graph and gradients."""
        self._outputs = {}
        self.optimizer.zero_grad()

    def save(self, path: str) -> None:
        torch.save({"module": self.tower.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict() if self.scheduler else None}, path)

    def load(self, path: str, optimizer: bool = True, scheduler: bool = True) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.tower.module.load_state_dict(state["module"])
        if optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
        if scheduler and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.reset()
