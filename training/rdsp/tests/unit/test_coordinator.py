"""The one-step coordinator state machine, on fake workers."""

import pytest
from test_schedule import make_plan

from ray_deepspeed_pipeline.coordinator import PipelineCoordinator
from ray_deepspeed_pipeline.errors import PipelinePoisoned, StepFailed


class FakeStageWorker:
    """Synchronous stand-in for a stage actor group speaking the step protocol:
    a step carries the stage's op list and returns {"losses", "ready"}.
    Records every command and every op it was asked to run."""

    reconnects = 0

    def __init__(self, stage, is_terminal, fail_on=None, ready=True):
        self.stage = stage
        self.is_terminal = is_terminal
        self.fail_on = fail_on or set()  # command or op kinds that raise
        self.ready = ready
        self.commands = []
        self.ops = []
        self.saw_labels = False
        self.saw_inputs = False
        self.loss_calls = 0
        self.applies = 0

    @staticmethod
    def reconnect_p2p(clients, epoch):
        FakeStageWorker.reconnects += 1

    def prepare(self, inputs=None, labels=None):
        """A step's payload, made ready ahead of its dispatch."""
        return (inputs, labels)

    def submit(self, command, *, inputs=None, labels=None, control=None, prepared=None):
        if prepared is not None:
            inputs, labels = prepared
        if command.kind != "apply":
            self.last_inputs = inputs
        self.commands.append(command)
        if command.kind in self.fail_on:
            raise RuntimeError(f"injected failure on {command.kind}")
        if command.kind == "apply":
            self.applies += 1
            return True
        assert command.kind in ("step", "eval_step"), command.kind
        ops = control["ops"]
        self.ops.extend(ops)
        if any(kind in self.fail_on for kind, _, _ in ops):
            raise RuntimeError("injected failure inside the step")
        self.saw_inputs |= inputs is not None
        self.saw_labels |= labels is not None
        assert (inputs is not None) == (self.stage == 0), "only stage 0 gets inputs"
        assert (labels is not None) == self.is_terminal, "only the last stage gets labels"
        forwards = [mb for kind, mb, _ in ops if kind in ("forward", "eval")]
        losses = None
        if self.is_terminal:
            self.loss_calls += len(forwards)
            losses = [0.5 + mb for mb in forwards]
        return {"losses": losses, "ready": self.ready}


def build(stages=2, microbatches=4, fail_on=None):
    plan = make_plan(stages, microbatches)
    workers = [FakeStageWorker(s, s == stages - 1,
                               (fail_on or {}).get(s))
               for s in range(stages)]
    return PipelineCoordinator(plan, workers), workers


def test_one_call_one_global_step_and_mean_loss():
    coord, workers = build()
    data = [(f"x{k}", f"y{k}") for k in range(4)]
    loss = coord.train_batch(iter(data))
    assert loss == pytest.approx((0.5 + 1.5 + 2.5 + 3.5) / 4)
    assert coord.global_steps == 1
    assert all(w.applies == 1 for w in workers)


def test_only_terminal_stage_sees_labels_and_loss():
    coord, workers = build(stages=3, microbatches=6)
    coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(6)]))
    assert workers[0].saw_inputs and not workers[0].saw_labels
    assert not workers[1].saw_inputs and not workers[1].saw_labels
    assert workers[2].saw_labels and workers[2].loss_calls == 6


def test_eval_never_increments_global_step_or_applies():
    coord, workers = build()
    data = [(f"x{k}", f"y{k}") for k in range(4)]
    loss = coord.eval_batch(iter(data))
    assert loss == pytest.approx(2.0)
    assert coord.global_steps == 0
    assert all(w.applies == 0 for w in workers)
    assert all(c.kind == "eval_step" for w in workers for c in w.commands)
    assert all(kind == "eval" for w in workers for kind, _, _ in w.ops)


def test_exhausted_iterator_fails_before_any_dispatch():
    coord, workers = build()
    with pytest.raises(StepFailed):
        coord.train_batch(iter([("x0", "y0")]))
    assert coord.global_steps == 0
    assert all(w.commands == [] for w in workers), \
        "no command may be issued when consumption fails"


def test_forward_failure_aborts_without_poison_and_is_retryable():
    coord, workers = build(fail_on={1: {"forward"}})
    with pytest.raises(StepFailed, match="stage 1"):
        coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))
    assert coord.global_steps == 0
    assert all(w.applies == 0 for w in workers)
    # not poisoned: after the links are rebuilt, a fresh call succeeds
    workers[1].fail_on = set()
    reconnects = FakeStageWorker.reconnects
    assert coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)])) > 0
    assert FakeStageWorker.reconnects > reconnects
    assert coord.global_steps == 1


def test_stage_not_ready_aborts_before_apply():
    coord, workers = build()
    workers[0].ready = False
    with pytest.raises(StepFailed, match="not ready"):
        coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))
    assert coord.global_steps == 0
    assert all(w.applies == 0 for w in workers)


def test_each_stage_receives_its_1f1b_op_list():
    coord, workers = build(stages=3, microbatches=4)
    coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))
    first = [(kind, mb) for kind, mb, _ in workers[0].ops]
    assert first[:2] == [("forward", 0), ("forward", 1)]  # two warm-up forwards
    assert first[-1] == ("ready", None)
    assert all(len(w.commands) == 2 for w in workers)  # one step + one apply


def test_apply_failure_poisons_generation():
    coord, workers = build(fail_on={1: {"apply"}})
    with pytest.raises(PipelinePoisoned):
        coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))
    assert coord.global_steps == 0, "a poisoned generation never counts"
    # every subsequent call is rejected, even after workers heal
    workers[1].fail_on = set()
    with pytest.raises(PipelinePoisoned):
        coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))
    with pytest.raises(PipelinePoisoned):
        coord.eval_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))


def test_no_tensor_handles_resolved_by_driver():
    """The coordinator returns a plain float built only from terminal losses."""
    coord, _ = build()
    loss = coord.train_batch(iter([(f"x{k}", f"y{k}") for k in range(4)]))
    assert isinstance(loss, float)
    assert loss == pytest.approx(2.0)


def test_facade_integration_with_real_coordinator():
    """rdsp.initialize wired to a real PipelineCoordinator over fake workers
    honours the public engine contract."""
    import torch
    from test_partition import ToyLM

    import ray_deepspeed_pipeline as rdsp
    from ray_deepspeed_pipeline import api
    from ray_deepspeed_pipeline.compiler import lower

    model = ToyLM()

    def factory(*, model, pipeline_config, ds_config, loss_fn, weights=None):
        plan = lower(model, pipeline_config, ds_config)
        workers = [FakeStageWorker(s, s == len(plan.stages) - 1)
                   for s in range(len(plan.stages))]
        return PipelineCoordinator(plan, workers)

    original = api._coordinator_factory
    api._coordinator_factory = factory
    try:
        engine, _, _, _ = rdsp.initialize(
            model=model, config={"train_batch_size": 8},
            pipeline_config=rdsp.PipelineConfig(
                stages=2, partition=rdsp.UniformTransformerBlocks(),
                microbatches=4),
            loss_fn=lambda outputs, labels: 0.0)
        loss = engine.train_batch(data_iter=iter([(f"x{k}", f"y{k}")
                                                  for k in range(4)]))
        assert isinstance(loss, torch.Tensor) and loss.shape == ()
        assert engine.global_steps == 1
    finally:
        api._coordinator_factory = original


def build_prefetching(microbatches=2):
    import dataclasses
    plan = dataclasses.replace(make_plan(2, microbatches), prefetch=True)
    workers = [FakeStageWorker(s, s == 1) for s in range(2)]
    return PipelineCoordinator(plan, workers), workers


def test_prefetch_reads_the_next_step_during_this_one_and_keeps_order():
    coord, workers = build_prefetching()
    data = iter([(f"x{k}", f"y{k}") for k in range(6)])
    coord.train_batch(data)
    assert workers[0].last_inputs == ["x0", "x1"]
    assert next(data) == ("x4", "y4")  # x2, x3 were read ahead for step 2
    coord.train_batch(data)
    assert workers[0].last_inputs == ["x2", "x3"]


def test_prefetch_refuses_a_different_iterator_and_eval_leaves_it_alone():
    coord, workers = build_prefetching()
    data = iter([(f"x{k}", f"y{k}") for k in range(6)])
    coord.train_batch(data)
    coord.eval_batch(iter([("e0", "f0"), ("e1", "f1")]))
    assert workers[0].last_inputs == ["e0", "e1"]
    with pytest.raises(StepFailed, match="same iterator"):
        coord.train_batch(iter([("z0", "w0"), ("z1", "w1")]))
    coord.train_batch(data)
    assert workers[0].last_inputs == ["x2", "x3"]


def test_prefetch_reports_an_exhausted_iterator_on_the_step_that_needs_it():
    coord, _ = build_prefetching()
    data = iter([(f"x{k}", f"y{k}") for k in range(3)])
    coord.train_batch(data)  # reading ahead finds only one entry: not this step's problem
    with pytest.raises(StepFailed, match="exhausted"):
        coord.train_batch(data)
