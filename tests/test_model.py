import torch

from lentoorava import OrganismConfig, PairCopyDataset, make_variant


def test_dataset_shapes_and_hidden_half():
    observed, target, _ = PairCopyDataset(8, seed=1)[0]
    assert observed.shape == target.shape == (3, 32, 32)
    assert target[:, :, 16:].sum() > observed[:, :, 16:].sum()


def test_all_variants_forward_and_backward():
    x, target, _ = PairCopyDataset(4, seed=2)[0]
    batch = x[None].repeat(2, 1, 1, 1)

    for name in ["active", "fixed", "transport"]:
        cfg = OrganismConfig(
            channels=8,
            n_agents=4,
            hidden_dim=16,
            read_dim=16,
            steps=2,
        )
        model = make_variant(name, cfg)
        y = model(batch)
        assert y.shape == batch.shape
        assert torch.isfinite(y).all()

        loss = ((y - target[None]) ** 2).mean()
        loss.backward()
        grads = [
            p.grad
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)


def test_trace_addresses_stay_bounded():
    x, _, _ = PairCopyDataset(1, seed=3)[0]
    cfg = OrganismConfig(
        channels=8,
        n_agents=4,
        hidden_dim=16,
        read_dim=16,
        steps=3,
    )
    model = make_variant("active", cfg)
    _, trace = model(x[None], return_trace=True)
    assert trace.min() >= -1.0
    assert trace.max() <= 1.0
