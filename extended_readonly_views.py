"""Optional measurement instruments for testing known blind spots.

Only read-only observational utilities. Not a replacement, extension, or update
of frozen DNO's mathematics, dynamics, parameterization, or State.
"""
from __future__ import annotations
import numpy as np
import torch


def stable_identity_labels(state):
    """Stable *bookkeeping* labels, distinct from learned semantic identity."""
    return tuple(sorted(state.ids))


def identity_distinguished(a, b) -> bool:
    return stable_identity_labels(a) != stable_identity_labels(b)


def temporal_value_signature(model, state, *, steps=3, dt=.1):
    """Hypothetical model-only rollout. No ground truth, no history recovery.

    Returns observations at 0..steps, not writes to the original state.
    """
    if steps < 0 or dt < 0 or not np.isfinite(dt):
        raise ValueError('invalid temporal sampling')
    from multiview_observer import view
    result=[]
    s=state
    with torch.no_grad():
        for k in range(steps+1):
            result.append(view(s,'value'))
            if k!=steps:
                s=model.evolve(s,dt)
    return result


def address_signature(model, state):
    """Active model-side address probabilities, unlike passive autonomous flow."""
    with torch.no_grad():
        p=model.propose(state)
    return np.concatenate((p.source.detach().cpu().numpy(),
                           p.target.detach().cpu().numpy(),
                           p.activity.detach().cpu().reshape(1).numpy()))
