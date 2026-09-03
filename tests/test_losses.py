import torch

from lentoorava import PairCopyDataset
from lentoorava.losses import LossConfig, curriculum_weights, multiscale_right_loss, organism_loss


def test_curriculum_moves_from_coarse_to_fine():
    early = curriculum_weights(0.0)
    late = curriculum_weights(1.0)
    assert early[-1] > early[0]
    assert late[0] > late[-1]


def test_multiscale_supplies_long_range_gradient():
    _, target, meta = PairCopyDataset(1, seed=7)[0]
    pred = torch.zeros_like(target, requires_grad=True)
    loss, _ = multiscale_right_loss(pred[None], target[None], progress=0.0)
    loss.backward()
    y = int(meta["y"])
    x = 24
    far_y = min(target.shape[-2] - 1, y + 7)
    coarse_grad = pred.grad[:, far_y, x].abs().mean().item()
    assert coarse_grad > 1e-5


def test_organism_loss_is_finite_and_backwardable():
    _, target, _ = PairCopyDataset(1, seed=9)[0]
    pred = torch.full_like(target, 0.05, requires_grad=True)
    loss, metrics = organism_loss(
        pred[None],
        target[None],
        progress=0.5,
        cfg=LossConfig(),
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v) for v in metrics.values())
    loss.backward()
    assert torch.isfinite(pred.grad).all()
