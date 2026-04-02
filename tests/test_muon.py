import torch
from torch import nn

from tabicl.train.muon import Muon


def test_muon_step_updates_params_without_nan():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 4))
    optimizer = Muon(model.parameters(), lr=1e-3, weight_decay=0.01)

    x = torch.randn(32, 8)
    y = torch.randint(0, 4, (32,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()

    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    changed = [not torch.equal(a, b) for a, b in zip(before, after)]
    assert any(changed), "Muon step did not update any parameter"
    assert all(torch.isfinite(p).all() for p in after), "Found non-finite parameter after Muon step"
