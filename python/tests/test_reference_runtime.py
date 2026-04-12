from __future__ import annotations

import torch

from ttt.runtime import Int5TiledRuntime, ReferenceSa2x2Device


def test_reference_runtime_tiled_matmul_matches_integer_reference() -> None:
    runtime = Int5TiledRuntime(ReferenceSa2x2Device())
    lhs = torch.tensor(
        [
            [1, -2, 3, 0],
            [2, 1, -1, 4],
            [0, 2, 3, -2],
        ],
        dtype=torch.int32,
    )
    rhs = torch.tensor(
        [
            [2, 1, -1, 0],
            [3, -2, 1, 4],
            [1, 0, 2, -3],
            [2, 1, -2, 1],
        ],
        dtype=torch.int32,
    )

    got = runtime.matmul(lhs, rhs, post_shift=0, relu=False, clamp_output_to_int5=False)
    expected = torch.matmul(lhs, rhs)
    assert torch.equal(got, expected.to(torch.int32))


def test_reference_runtime_add_and_relu_clamp_back_to_int5() -> None:
    runtime = Int5TiledRuntime(ReferenceSa2x2Device())
    lhs = torch.tensor([[7, -8], [15, -16]], dtype=torch.int32)
    rhs = torch.tensor([[5, 3], [15, 4]], dtype=torch.int32)

    summed = runtime.add(lhs, rhs, clamp_output_to_int5=True)
    relu = runtime.relu(summed, clamp_output_to_int5=True)

    assert torch.equal(summed, torch.tensor([[12, -5], [15, -12]], dtype=torch.int32))
    assert torch.equal(relu, torch.tensor([[12, 0], [15, 0]], dtype=torch.int32))
