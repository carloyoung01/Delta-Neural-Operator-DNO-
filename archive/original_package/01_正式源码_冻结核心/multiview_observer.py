"""Read-only, external multi-coordinate observational audit for the frozen DNO.

Nothing here alters DNO, its weights, State or underlying dynamics.
An inability to distinguish states is NEVER interpreted as their equivalence.
The available views are tests chosen by an analyst, not ontology.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import numpy as np
import torch

VIEW_ORDER = ('global_mean', 'value', 'boundary', 'law', 'incidence', 'time', 'observer', 'future_value')


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().double().numpy().copy()


def _indices(state) -> list[int]:
    return sorted(range(len(state.ids)), key=lambda i: state.ids[i])


def _carrier_matrix(state, key: str) -> np.ndarray:
    return _np(getattr(state, key))[_indices(state)].reshape(-1)


def view(state, kind: str, *, model=None, dt: float = 0.125) -> np.ndarray:
    """Sample one view of an unchanged DNO State.

    `future_value` computes a hypothetical rollout on a NEW State object;
    it is not empirical evidence about an external physical process.
    """
    if kind == 'global_mean':
        # A deliberately lossy observation, showing why local details disappear.
        return _np(state.value.mean(dim=0)) if len(state.ids) else np.zeros(state.value.shape[1])
    if kind in ('value', 'boundary', 'law', 'support', 'age'):
        return _carrier_matrix(state, kind)
    if kind == 'observer':
        return _np(state.observer)
    if kind == 'time':
        return _carrier_matrix(state, 'age')
    if kind == 'incidence':
        idx = {ident: i for i, ident in enumerate(sorted(state.ids))}
        matrix = np.zeros((len(idx), len(idx)), dtype=np.float64)
        for a, b in state.links:
            matrix[idx[a], idx[b]] = 1.
        return matrix.ravel()
    if kind == 'future_value':
        if model is None:
            raise ValueError('future_value requires a frozen model reference')
        with torch.no_grad():
            later = model.evolve(state, dt)
        return np.concatenate((np.array(sorted(later.ids), dtype=np.float64),
                               _carrier_matrix(later, 'value')))
    if kind == 'full':
        return np.concatenate([view(state, k) for k in
             ('value', 'boundary', 'law', 'support', 'age', 'observer', 'incidence')])
    raise ValueError(f'unknown observation kind: {kind}')


@dataclass(frozen=True)
class ViewEvidence:
    name: str
    finding: str      # 'difference' | 'not_distinguished' | 'unavailable'
    magnitude: Optional[float] = None
    reason: str = ''


@dataclass(frozen=True)
class Audit:
    status: str       # 'difference_found' | 'undetermined'
    trace: tuple[ViewEvidence, ...]
    first_distinguishing_view: Optional[str]
    # This result is never a judgement that the generator is wrong.


def compare(a, b, *, model=None, views: Sequence[str] = VIEW_ORDER,
            tolerance: float = 1e-7, dt: float = 0.125, stop_on_difference: bool = True) -> Audit:
    """Switch viewpoints to answer a *specific distinction* query.

    The criterion is empirical difference in a specified view, not a claim of
    object identity, cause, complete knowledge, or global model adequacy.
    """
    if tolerance < 0:
        raise ValueError('tolerance must be nonnegative')
    trace: list[ViewEvidence] = []
    first: Optional[str] = None
    for name in views:
        try:
            aa = view(a, name, model=model, dt=dt)
            bb = view(b, name, model=model, dt=dt)
        except (ValueError, RuntimeError) as ex:
            trace.append(ViewEvidence(name, 'unavailable', reason=str(ex)))
            continue
        if aa.shape != bb.shape:
            # Different counts are distinguishable in this view.
            magnitude = float('inf')
        elif aa.size:
            magnitude = float(np.linalg.norm(aa-bb))
        else:
            magnitude = 0.
        distinct = magnitude > tolerance
        trace.append(ViewEvidence(name, 'difference' if distinct else 'not_distinguished', magnitude))
        if distinct and first is None:
            first = name
            if stop_on_difference:
                break
    return Audit('difference_found' if first is not None else 'undetermined', tuple(trace), first)


def orthogonal_coordinate_check(a: np.ndarray, b: np.ndarray, Q: np.ndarray,
                                *, tolerance: float = 1e-9) -> dict:
    """Under an orthogonal change of coordinates, Euclidean distance is invariant.

    Important: this transforms ONLY observation vectors. It does not assume
    the trained DNO generator is equivariant under such transformations.
    """
    a, b, Q = np.asarray(a), np.asarray(b), np.asarray(Q)
    if a.ndim != 1 or b.shape != a.shape or Q.shape != (len(a), len(a)):
        raise ValueError('coordinate shapes disagree')
    if not np.allclose(Q.T @ Q, np.eye(len(a)), atol=tolerance, rtol=0):
        raise ValueError('Q must be orthogonal')
    original = float(np.linalg.norm(a-b))
    transformed = float(np.linalg.norm(Q @ a-Q @ b))
    return {'original': original, 'transformed': transformed,
            'preserved': abs(original-transformed) <= tolerance}


def snapshot_hash(state, model=None) -> str:
    """Audit no mutation via exact tensor bytes + connectivity / IDs / weights."""
    import hashlib
    h = hashlib.sha256()
    for k in ('value','boundary','law','support','age','observer'):
        x = getattr(state,k).detach().cpu().contiguous()
        h.update(str((k, tuple(x.shape), str(x.dtype))).encode())
        h.update(x.numpy().tobytes())
    h.update(repr((state.ids,state.links,state.next_id)).encode())
    if model is not None:
        for k, w in model.state_dict().items():
            h.update(k.encode())
            h.update(w.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
