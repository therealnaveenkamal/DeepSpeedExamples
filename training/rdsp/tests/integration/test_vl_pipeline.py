"""A vision-language model (tiny Qwen3-VL) through rdsp.initialize() on real
Ray CPU actors, with only the DeepSpeed engine swapped for the torch stub.

Each pipelined run must reproduce the unsplit model's losses on the same
batches: images reach the vision encoder on stage 0, and the multimodal rotary
tables it computes travel to the later stages with the hidden state."""

import os

import pytest
import torch
import torch.nn.functional as F
from test_deepspeed_adapter import stub_engine_factory

import ray_deepspeed_pipeline as rdsp
from ray_deepspeed_pipeline.config import StageOverride
from ray_deepspeed_pipeline.errors import ValidationError

ray = pytest.importorskip("ray")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
    pytest.skip("transformers without Qwen3-VL", allow_module_level=True)

VOCAB, SEQ, ROWS, N_MB = 256, 12, 2, 2
IMAGE, IMAGE_PAD, VISION_START = 250, 250, 252
GRID = (1, 4, 4)  # 16 patches, 4 tokens after 2x2 merging
PATCH_DIM = 3 * 1 * 4 * 4
DS = {"train_batch_size": N_MB * ROWS, "gradient_accumulation_steps": N_MB,
      "optimizer": {"type": "AdamW", "params": {"lr": 0.01}}}


def tiny_qwen3_vl():
    cfg = transformers.Qwen3VLConfig(
        text_config=dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, rope_scaling={"rope_type": "default",
                                                    "mrope_section": [2, 3, 3],
                                                    "mrope_interleaved": True}),
        vision_config=dict(depth=4, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=64, patch_size=4, spatial_merge_size=2,
                           temporal_patch_size=1, deepstack_visual_indexes=[1, 2],
                           num_position_embeddings=16),
        image_token_id=IMAGE_PAD, video_token_id=251, vision_start_token_id=VISION_START,
        tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    return cfg


def make_batches(seed=0):
    """N_MB microbatches of ROWS rows, one image per row at the same place, so
    every row has the same number of masked label positions."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(N_MB):
        ids = torch.randint(0, 200, (ROWS, SEQ), generator=g)
        ids[:, 2] = VISION_START
        ids[:, 3:7] = IMAGE_PAD
        labels = ids.clone()
        labels[:, 2:7] = -100
        inputs = {
            "input_ids": ids,
            "mm_token_type_ids": (ids == IMAGE_PAD).long(),
            # not row-shaped: one packed tensor per row, concatenated per rank
            "pixel_values": [torch.randn(16, PATCH_DIM, generator=g) for _ in range(ROWS)],
            "image_grid_thw": [torch.tensor([GRID]) for _ in range(ROWS)],
        }
        batches.append((inputs, labels))
    return batches


def lm_loss(logits, labels):
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), labels[:, 1:].reshape(-1))


def unsplit_losses(model, batches, steps):
    """Loss per step of the whole model in one process, same AdamW."""
    opt = torch.optim.AdamW(model.parameters(), lr=DS["optimizer"]["params"]["lr"])
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        total = 0.0
        for inputs, labels in batches:
            full = {k: torch.cat(v) if isinstance(v, list) else v for k, v in inputs.items()}
            loss = lm_loss(model(**full, use_cache=False).logits, labels) / len(batches)
            loss.backward()
            total += float(loss)
        opt.step()
        losses.append(total)
    return losses


@pytest.fixture(scope="module")
def ray_ctx():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.pathsep.join([os.path.join(here, "..", "unit"), here])  # stub engine, lm_loss
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False,
             runtime_env={"env_vars": {"PYTHONPATH": path}})
    yield
    ray.shutdown()


@pytest.fixture()
def stub_engines(monkeypatch):
    """Real runtime and default stage builder; only the engine is the CPU stub."""
    from ray_deepspeed_pipeline import stage_group

    real = stage_group.create_stage_clients
    built = []

    def cpu_clients(model, plan, loss_fn, **kw):
        clients = real(model, plan, loss_fn, engine_factory=stub_engine_factory,
                       use_gpu=False, **kw)
        built.append(clients)
        return clients

    monkeypatch.setattr(stage_group, "create_stage_clients", cpu_clients)
    yield
    for clients in built:
        for client in clients:
            client.shutdown()


def pipelined_losses(model, cuts, steps, overrides=(), weights=None):
    engine, _, _, _ = rdsp.initialize(
        model=model, config=DS, loss_fn=lm_loss, weights=weights,
        pipeline_config=rdsp.PipelineConfig(
            stages=len(cuts) + 1, partition=rdsp.ExplicitCuts(cuts),
            stage_overrides=overrides))
    return [float(engine.train_batch(data_iter=iter(make_batches()))) for _ in range(steps)]


def test_two_stages_match_unsplit_model(ray_ctx, stub_engines):
    torch.manual_seed(0)
    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference.load_state_dict(model.state_dict())

    expected = unsplit_losses(reference, make_batches(), steps=3)
    assert pipelined_losses(model, (3,), steps=3) == pytest.approx(expected, rel=1e-4)


def test_data_parallel_vision_stage_matches_unsplit_model(ray_ctx, stub_engines):
    """Stage 0 (vision encoder, blocks 0-1) on 2 data-parallel ranks: each
    rank gets its rows' images, and the next stage reassembles the rows of
    both the hidden state and the rotary tables."""
    torch.manual_seed(1)
    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference.load_state_dict(model.state_dict())

    expected = unsplit_losses(reference, make_batches(), steps=2)
    got = pipelined_losses(model, (2, 4), steps=2,
                           overrides=(StageOverride(stage=0, num_gpus=2),))
    assert got == pytest.approx(expected, rel=1e-4)


def test_meta_skeleton_loads_weights_per_stage(ray_ctx, stub_engines, tmp_path):
    """The driver holds only a meta-device skeleton; each stage reads its own
    tensors from the checkpoint directory and trains like the real model."""
    accelerate = pytest.importorskip("accelerate")
    torch.manual_seed(2)
    trained = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    trained.save_pretrained(tmp_path)
    expected = unsplit_losses(trained, make_batches(), steps=2)

    with accelerate.init_empty_weights():
        skeleton = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    assert all(p.is_meta for p in skeleton.parameters())
    got = pipelined_losses(skeleton, (3,), steps=2, weights=str(tmp_path))
    assert got == pytest.approx(expected, rel=1e-4)


def test_meta_skeleton_without_weights_is_rejected(ray_ctx, stub_engines):
    accelerate = pytest.importorskip("accelerate")
    with accelerate.init_empty_weights():
        skeleton = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl())
    with pytest.raises(Exception, match="meta device"):
        pipelined_losses(skeleton, (3,), steps=1)


def test_first_cut_before_vision_injection_is_rejected(ray_ctx, stub_engines):
    """Qwen3-VL adds vision features inside blocks 0-1 (deepstack), so those
    must stay on the first stage."""
    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl())
    with pytest.raises(ValidationError, match="first cut at 2"):
        pipelined_losses(model, (1,), steps=1)


def test_recompute_on_every_stage_matches_unsplit_model(ray_ctx, stub_engines):
    """Blocks rerun their forward during backward; the injected rotary tables
    and hidden state must be the ones the first run used."""
    torch.manual_seed(3)
    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl()).float()
    reference.load_state_dict(model.state_dict())

    expected = unsplit_losses(reference, make_batches(), steps=2)
    got = pipelined_losses(model, (2, 4), steps=2, overrides=(
        StageOverride(stage=0, num_gpus=2, recompute=True),
        StageOverride(stage=1, recompute=True), StageOverride(stage=2, recompute=True)))
    assert got == pytest.approx(expected, rel=1e-4)


def test_balanced_cuts_keep_vision_injection_on_the_first_stage():
    from ray_deepspeed_pipeline.config import BalancedTransformerBlocks
    from ray_deepspeed_pipeline.partition import partition_parameters

    model = transformers.Qwen3VLForConditionalGeneration(tiny_qwen3_vl())
    for stages in (2, 3, 4):
        parts = partition_parameters(model, BalancedTransformerBlocks(), stages)
        assert parts[0].block_stop >= 2  # blocks 0-1 receive deepstack features
