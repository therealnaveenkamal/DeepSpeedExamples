"""Stage-to-stage exchange over gloo: two ranks in one process, one thread each."""

import socket
import threading

import torch

from ray_deepspeed_pipeline.p2p import PipelineP2P, make_store


def _port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_pair(sender, receiver):
    """Run sender(p2p) on rank 0 and receiver(p2p) on rank 1; returns the
    receiver's result."""
    port, result, errors = _port(), {}, []

    def rank(r, fn):
        try:
            store = make_store("127.0.0.1", port, 2, r == 0, 30)
            p2p = PipelineP2P(store, r, 2, 0, 30, torch.device("cpu"))
            p2p.begin_step()
            out = fn(p2p)
            p2p.end_step()
            if r == 1:
                result["value"] = out
        except Exception as e:  # surfaced in the main thread
            errors.append(e)

    threads = [threading.Thread(target=rank, args=(0, sender)),
               threading.Thread(target=rank, args=(1, receiver))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors
    return result["value"]


def test_hidden_state_and_named_extras_cross_every_microbatch():
    hidden = [torch.randn(2, 5, 8) for _ in range(3)]
    extras = [{"position_embeddings.0": torch.randn(2, 5, 4),
               "position_embeddings.1": torch.randn(2, 5, 4)} for _ in range(3)]

    def send(p2p):
        for mb in range(3):
            p2p.send_boundary("fwd", hidden[mb], extras[mb], 1, mb)

    got = _run_pair(send, lambda p2p: [p2p.recv_boundary("fwd", 0, mb) for mb in range(3)])

    for mb, (h, e) in enumerate(got):
        assert torch.equal(h, hidden[mb])
        assert list(e) == list(extras[mb])
        assert all(torch.equal(e[k], extras[mb][k]) for k in e)


def test_boundary_without_extras():
    hidden = torch.randn(2, 3, 4)
    got = _run_pair(lambda p2p: p2p.send_boundary("fwd", hidden, {}, 1, 0),
                    lambda p2p: p2p.recv_boundary("fwd", 0, 0))
    assert torch.equal(got[0], hidden) and got[1] == {}


def test_bf16_crosses_the_cpu_group_as_its_bits():
    """send_cpu sends bf16 as int16 bits (gloo may lack bf16); the header
    marks it so recv_cpu views the bits back, exactly."""
    from ray_deepspeed_pipeline.p2p import _decode, _encode

    t = torch.randn(3, 5).to(torch.bfloat16)
    bits = t.view(torch.int16)
    dtype, shape, flag = _decode(_encode(bits, n_extras=1))
    assert (dtype, shape, flag) == (torch.int16, (3, 5), 1)
    assert torch.equal(bits.clone().view(torch.bfloat16), t)
