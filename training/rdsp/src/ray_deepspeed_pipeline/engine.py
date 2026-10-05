"""RayPipelineEngine: the blocking driver-side facade over the coordinator."""

import torch

from ray_deepspeed_pipeline.data import CountingIterator, data_position, resume_loader_iter
from ray_deepspeed_pipeline.errors import UnsupportedEngineMethod, ValidationError
from ray_deepspeed_pipeline.protocols import resolve

_UNSUPPORTED_MSG = (
    "{name} is not available on RayPipelineEngine: the model is partitioned "
    "across stage actors, and forward/backward/step for a pipeline are not "
    "separable driver-side operations. Use train_batch() (or eval_batch()) — "
    "one call executes one globally coordinated optimizer step. "
    "Optimizer, scheduler, parameters, and module state live inside the stage "
    "actors and are managed through the DeepSpeed config."
)


def _public_loss(value) -> torch.Tensor:
    """Mean loss as a zero-dimensional detached CPU tensor."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").reshape(()).clone()
    return torch.tensor(float(value))


class RayPipelineEngine:
    """Driver-side training client. Not a DeepSpeedEngine: the model lives in
    the stage actors, so only whole-step methods are available."""

    def __init__(self, coordinator, training_dataloader=None):
        self._coordinator = coordinator
        self._training_dataloader = training_dataloader
        self._owned_iter = None

    @property
    def global_steps(self) -> int:
        return int(self._coordinator.global_steps)

    def _data_iter(self, data_iter):
        if data_iter is not None:
            if self._training_dataloader is not None:
                raise ValidationError(
                    "this engine owns the dataloader built from training_data; "
                    "a call-specific data_iter is only accepted when "
                    "training_data=None was passed to initialize()")
            return data_iter
        if self._training_dataloader is None:
            raise ValidationError(
                "no data source: initialize() received training_data=None, so "
                "train_batch/eval_batch require an explicit data_iter argument")
        if self._owned_iter is None:
            self._owned_iter = CountingIterator(iter(self._training_dataloader))
        return self._owned_iter

    def train_batch(self, data_iter=None) -> torch.Tensor:
        source = self._data_iter(data_iter)
        return _public_loss(resolve(self._coordinator.train_batch(source)))

    def eval_batch(self, data_iter=None) -> torch.Tensor:
        source = self._data_iter(data_iter)
        return _public_loss(resolve(self._coordinator.eval_batch(source)))

    def save_checkpoint(self, save_dir, tag=None, client_state=None,
                        save_latest=True) -> bool:
        """Save every stage plus global step and the owned loader's position.

        Returns True once the manifest is committed; a failed save raises
        CheckpointError and commits nothing."""
        position = {"owner": "caller"}
        if self._training_dataloader is not None:
            consumed = self._owned_iter.consumed if self._owned_iter else 0
            position = data_position(self._training_dataloader, consumed)
        return bool(resolve(self._coordinator.save_checkpoint(
            save_dir, tag, client_state=client_state, data_position=position,
            save_latest=save_latest)))

    def load_checkpoint(self, load_dir, tag=None, load_optimizer_states=True,
                        load_lr_scheduler_states=True):
        """Restore the whole pipeline; returns (load_path, client_state) like
        DeepSpeed. Clears a poisoned pipeline."""
        result = resolve(self._coordinator.load_checkpoint(
            load_dir, tag, load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states))
        position = result.get("data_position") or {"owner": "caller"}
        if position.get("owner") == "engine":
            if self._training_dataloader is None:
                raise ValidationError(
                    "checkpoint records an engine-owned data position but this "
                    "engine was initialized with training_data=None")
            self._owned_iter = resume_loader_iter(
                self._training_dataloader, position["entries_consumed"])
        return result["path"], result["client_state"]

    # Local-engine methods that have no meaning for a partitioned model.

    def __call__(self, *args, **kwargs):
        raise UnsupportedEngineMethod(_UNSUPPORTED_MSG.format(name="engine(inputs)"))

    def forward(self, *args, **kwargs):
        raise UnsupportedEngineMethod(_UNSUPPORTED_MSG.format(name="forward()"))

    def backward(self, *args, **kwargs):
        raise UnsupportedEngineMethod(_UNSUPPORTED_MSG.format(name="backward()"))

    def step(self, *args, **kwargs):
        raise UnsupportedEngineMethod(_UNSUPPORTED_MSG.format(name="step()"))

    @property
    def module(self):
        raise UnsupportedEngineMethod(_UNSUPPORTED_MSG.format(name="engine.module"))

    def parameters(self, *args, **kwargs):
        raise UnsupportedEngineMethod(_UNSUPPORTED_MSG.format(name="parameters()"))
