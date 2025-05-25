import pytest
import torch


@pytest.mark.parametrize(
    ("batch_sizes", "dim", "expected_size"),
    [
        ([(1,), (3,)], 0, (2, 3)),
        ([(1,), (3,)], -1, (2, 3)),
        ([(3, 2), (5, 2)], 0, (2, 5, 2)),
        ([(2, 3), (2, 5)], -1, (2, 2, 5)),
    ],
)
def test_pad_tensors(
    batch_sizes: tuple[int], dim: int, expected_size: tuple[int]
) -> None:
    from seq_rec.data.load import pad_tensors

    batch = [torch.rand(size) for size in batch_sizes]
    padded = pad_tensors(batch, dim=dim)
    assert padded.size() == expected_size, f"{padded.size() = } != {expected_size = }"
