import torch
from torch.nn import functional as F


def sample_time(x, positions):
    """Linear interpolation of [people,time,channels], with explicit coordinates."""
    t = x.shape[1]
    positions = positions.clamp(0, t-1)
    lo = positions.floor().long()
    hi = (lo+1).clamp(max=t-1)
    weight = (positions-lo).unsqueeze(-1)
    lo = lo.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    hi = hi.unsqueeze(-1).expand_as(lo)
    return x.gather(1, lo)*(1-weight) + x.gather(1, hi)*weight


def augment(x):
    n, t, c = x.shape
    y = x + 0.02*torch.randn_like(x)
    y = y * (1 + 0.1*torch.randn(n, 1, c, device=x.device))
    knots = 1 + 0.1*torch.randn(n, c, 6, device=x.device)
    magnitude = F.interpolate(knots, size=t, mode="linear", align_corners=True).transpose(1, 2)
    y = y*magnitude
    increments = F.interpolate(0.1*torch.randn(n, 1, 6, device=x.device),
                               size=t-1, mode="linear", align_corners=True).squeeze(1).exp()
    positions = torch.cat([torch.zeros(n, 1, device=x.device), increments.cumsum(1)], dim=1)
    positions = positions / positions[:, -1:] * (t-1)
    return sample_time(y, positions)


def speed_transform(x, rates, labels=None):
    n, t, _ = x.shape
    if labels is None:
        labels = torch.randint(len(rates), (n,), device=x.device)
    rate = x.new_tensor(rates)[labels].unsqueeze(1)
    positions = (torch.arange(t, device=x.device)[None]-(t-1)/2)*rate+(t-1)/2
    # Faster views need unavailable context; reflected boundaries are an explicit assumption.
    positions = positions.remainder(2*(t-1))
    positions = torch.minimum(positions, 2*(t-1)-positions)
    return sample_time(x, positions), labels
