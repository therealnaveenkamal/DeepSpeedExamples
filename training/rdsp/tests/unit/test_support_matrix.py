"""The support matrix is default-deny and internally consistent."""

from ray_deepspeed_pipeline.support_matrix import FIRST_ROW, P8_ROWS, ROWS


def test_default_deny():
    # a row claims support only with a recorded passing acceptance run; the
    # first row has passed its own
    assert FIRST_ROW.supported
    for row in P8_ROWS:
        assert not row.supported or row.evidence, row.row_id


def test_first_row_definition():
    assert FIRST_ROW.row_id == "p6-first-row"
    assert (FIRST_ROW.stages, FIRST_ROW.gpus_per_stage) == (2, 1)
    assert FIRST_ROW.dtype == "bf16" and FIRST_ROW.zero_stage == 0
    assert FIRST_ROW.schedule == "1f1b" and FIRST_ROW.boundaries == "identity"
    assert (FIRST_ROW.dp, FIRST_ROW.tp, FIRST_ROW.ep, FIRST_ROW.sp) == (1, 1, 1, 1)


def test_row_ids_unique():
    ids = [r.row_id for r in ROWS]
    assert len(ids) == len(set(ids))


def test_composed_row_requires_every_prerequisite():
    by_id = {r.row_id: r for r in P8_ROWS}
    for row in P8_ROWS:
        assert all(req in by_id for req in row.requires), row.row_id
        if row.supported:
            assert all(by_id[req].supported for req in row.requires), \
                f"{row.row_id} cannot be supported before {row.requires}"
    assert set(by_id["p8-four-stage-mixed"].requires) == set(by_id) - {"p8-four-stage-mixed"}
