"""Delta DNO: compressed current-state generative kernel (finite research model).

The model has no event taxonomy, no pair candidate table, no history archive,
no compulsory event scheduler, and no language-specific head. Every effective
unit—including a relation created by interaction—is a carrier with the same
fields. One shared generator is used for autonomous evolution and interaction.

This is an executable hypothesis, NOT a proof of a universal generative theory.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Mapping
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _net(input_dim: int, output_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(),
                         nn.Linear(hidden, output_dim))


@dataclass(frozen=True)
class State:
    value: Tensor          # [N,D]: current effective content, not prior observations
    boundary: Tensor       # [N,D]: mutable current boundary features
    law: Tensor            # [N,D]: mutable effective local law, NOT shared parameters
    support: Tensor        # [N,1]: nonnegative existence support
    age: Tensor            # [N,1]: local elapsed time
    ids: tuple[int, ...]   # stable identities; no historical event list
    links: tuple[tuple[int, int], ...]  # present directed incidence, not messages
    observer: Tensor       # [D]: current perspective; may act as endogenous input
    next_id: int = 0       # only an identity allocator; never model input

    def validate(self) -> None:
        if self.value.ndim != 2:
            raise ValueError('value must have shape [N,D]')
        n, d = self.value.shape
        if any(v.shape != (n, d) for v in (self.boundary, self.law)):
            raise ValueError('current feature dimensions disagree')
        if self.support.shape != (n, 1) or self.age.shape != (n, 1):
            raise ValueError('support / age dimensions disagree')
        if self.observer.shape != (d,) or len(self.ids) != n:
            raise ValueError('observer / identity dimensions disagree')
        if len(set(self.ids)) != n or any(i < 0 for i in self.ids):
            raise ValueError('identities must be unique nonnegative integers')
        if self.next_id < (max(self.ids) + 1 if self.ids else 0):
            raise ValueError('next identity would collide')
        if len(set(self.links)) != len(self.links) or any(
                a not in self.ids or b not in self.ids for a, b in self.links):
            raise ValueError('invalid current incidence')
        if bool((self.support < 0).any()) or bool((self.age < 0).any()):
            raise ValueError('negative support / time')
        for v in (self.value, self.boundary, self.law,
                  self.support, self.age, self.observer):
            if not bool(torch.isfinite(v).all()):
                raise ValueError('nonfinite current state')

    def detach(self) -> 'State':
        names = ('value', 'boundary', 'law', 'support', 'age', 'observer')
        return replace(self, **{n: getattr(self, n).detach() for n in names})


@dataclass(frozen=True)
class Opportunity:
    source: Tensor       # probability over CURRENT carrier identities
    target: Tensor
    activity: Tensor     # >=0: zero means no action at this observation
    source_logits: Tensor
    target_logits: Tensor


class DNO(nn.Module):
    """Finite shared law + current carriers + conditional generic composition.

    Complexity of one `propose`: O(N+E) carrier/link visits, not an N*N list.
    `compose` may add one carrier, which can itself later take part in composition.
    The caller/runtime controls *when* to commit; the core has no event clock.
    """
    def __init__(self, d: int = 8, hidden: int = 32):
        super().__init__()
        if d <= 0 or hidden <= 0:
            raise ValueError('d and hidden must be positive')
        self.d, self.hidden = d, hidden
        self.seed_boundary = nn.Linear(d, d)
        self.seed_law = nn.Linear(d, d)
        p = 3 * d + 2  # content, boundary, effective law, support, local time
        self.generator = _net(2 * p + 2 * d + 1, 3 * (3 * d + 1), hidden)
        self.observer_flow = _net(3 * d, d, hidden)
        self.key = _net(p, hidden, hidden)
        self.link_context = nn.Linear(hidden, hidden, bias=False)
        self.source_query = _net(2 * d + hidden, hidden, hidden)
        self.target_query = _net(2 * d + 2 * hidden, hidden, hidden)
        self.activity = _net(2 * d + 3 * hidden, 1, hidden)
        self.observe_query = nn.Linear(2 * d, hidden)
        self.observe_out = nn.Linear(d + hidden, d)

    def blank(self) -> State:
        z = next(self.parameters()).new_zeros
        return State(z((0, self.d)), z((0, self.d)), z((0, self.d)),
                     z((0, 1)), z((0, 1)), (), (), z((self.d,)))

    def introduce(self, s: State, values: Tensor) -> State:
        """An encountered independent entity enters current state; not an event kind."""
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.d:
            raise ValueError('values must be [N,D]')
        if values.device != s.value.device or values.dtype != s.value.dtype:
            raise ValueError('device/dtype mismatch')
        k = values.shape[0]
        if not k:
            return s
        z = values.new_zeros((k, 1))
        return replace(s, value=torch.cat((s.value, values)),
                       boundary=torch.cat((s.boundary, self.seed_boundary(values).tanh())),
                       law=torch.cat((s.law, self.seed_law(values).tanh())),
                       support=torch.cat((s.support, z + 1)),
                       age=torch.cat((s.age, z)),
                       ids=s.ids + tuple(range(s.next_id, s.next_id + k)),
                       next_id=s.next_id + k)

    @staticmethod
    def _pack(s: State) -> Tensor:
        return torch.cat((s.value, s.boundary, s.law, s.support, s.age), -1)

    def _call(self, a: Tensor, b: Tensor, observer: Tensor,
              cue: Tensor, elapsed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raw = self.generator(torch.cat((a, b, observer, cue, elapsed), -1))
        return raw.chunk(3, -1)

    def evolve(self, s: State, dt: float, *,
               local: Mapping[int, Tensor] | None = None,
               external: Tensor | None = None) -> State:
        """Autonomous change, no implicit link messages. No external input required.

        External input changes the observer; local cues go only to addressed IDs.
        """
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('invalid elapsed time')
        if dt == 0:
            return s
        n = len(s.ids)
        d = self.d
        if external is not None and (external.shape != (d,) or
                                     external.device != s.value.device):
            raise ValueError('external observation has wrong dimensions/device')
        if local is not None and not set(local).issubset(s.ids):
            raise KeyError('local cue references absent identity')
        x = s.value.new_zeros((d,)) if external is None else external
        if n:
            cues = torch.stack([local.get(uid, s.value.new_zeros(d))
                                if local else s.value.new_zeros(d) for uid in s.ids])
            if cues.shape != (n, d):
                raise ValueError('invalid local cue dimension')
            z = self._pack(s)
            zero = torch.zeros_like(z)
            delta, _, _ = self._call(z, zero, torch.zeros_like(s.value),
                                     cues, z.new_full((n, 1), dt))
            new = torch.cat((z[:, :3 * d] + dt * delta[:, :3 * d].tanh(),
                             F.relu(s.support + dt * delta[:, 3 * d:3 * d + 1].tanh())), -1)
            value, boundary, law, support = torch.split(new, (d, d, d, 1), -1)
            age = s.age + dt
        else:
            value, boundary, law, support, age = (s.value, s.boundary, s.law,
                                                   s.support, s.age)
        # Autonomous observation-mode evolution does NOT silently inspect carriers.
        # Reading the present structure is a separate, optional act (attend).
        observer = s.observer + dt * self.observer_flow(
            torch.cat((s.observer, s.observer.new_zeros(d), x))).tanh()
        return self.compact(replace(s, value=value, boundary=boundary, law=law,
                                    support=support, age=age, observer=observer))

    def _keys(self, s: State) -> Tensor:
        """Incidence participates only in active reading/planning, not passive flow."""
        keys = self.key(self._pack(s)).tanh()
        if not s.links:
            return keys
        id_pos = {uid: pos for pos, uid in enumerate(s.ids)}
        src = torch.tensor([id_pos[a] for a, _ in s.links], device=keys.device)
        dst = torch.tensor([id_pos[b] for _, b in s.links], device=keys.device)
        context = torch.zeros_like(keys).index_add(
            0, dst, keys[src] * s.support[src])
        return torch.tanh(keys + self.link_context(context))

    def propose(self, s: State, *, external: Tensor | None = None) -> Opportunity:
        """Unfold a compressed *opportunity*, without a pair/event catalogue."""
        if not s.ids or not bool((s.support > 0).any()):
            raise ValueError('cannot address an empty current state')
        x = s.observer.new_zeros(self.d) if external is None else external
        if x.shape != (self.d,) or x.device != s.value.device:
            raise ValueError('invalid external cue')
        keys = self._keys(s)
        alive = s.support.flatten() > 0
        center = keys[alive].mean(0)
        source_query = self.source_query(torch.cat((s.observer, center, x))).tanh()
        src_logits = (keys * source_query).sum(-1) / math.sqrt(self.hidden)
        src_logits = src_logits.masked_fill(~alive, -torch.inf)
        src = src_logits.softmax(-1)
        selected = (src[:, None] * keys).sum(0)
        target_query = self.target_query(
            torch.cat((s.observer, center, x, selected))).tanh()
        dst_logits = (keys * target_query).sum(-1) / math.sqrt(self.hidden)
        dst_logits = dst_logits.masked_fill(~alive, -torch.inf)
        dst = dst_logits.softmax(-1)
        target = (dst[:, None] * keys).sum(0)
        activity = F.relu(self.activity(torch.cat((s.observer, center, x,
                                                    selected, target))))[0]
        return Opportunity(src, dst, activity, src_logits, dst_logits)

    def compose(self, s: State, source_id: int, target_id: int, *,
                cue: Tensor | None = None, strength: float = 1.0) -> State:
        """One generic rewrite: alter two carriers; optional positive-support child.

        A newly produced child is itself a carrier, with incidence A→child→B.
        No 'birth', 'join', 'relation', 'remember', or 'release' event taxonomy.
        """
        if not math.isfinite(strength) or strength < 0:
            raise ValueError('invalid interaction strength')
        if strength == 0:
            return s
        positions = {uid: i for i, uid in enumerate(s.ids)}
        if source_id not in positions or target_id not in positions:
            raise KeyError('contact identity not present')
        cue = s.observer.new_zeros(self.d) if cue is None else cue
        if cue.shape != (self.d,) or cue.device != s.value.device:
            raise ValueError('invalid interaction cue')
        i, j = positions[source_id], positions[target_id]
        if not bool((s.support[[i, j]] > 0).all()):
            raise ValueError('cannot compose nonexistent support')
        z = self._pack(s)
        a, b, child = self._call(z[i:i + 1], z[j:j + 1],
                                s.observer[None, :], cue[None, :],
                                z.new_zeros((1, 1)))
        d = self.d
        xy = torch.cat((s.value, s.boundary, s.law), -1)
        ij = torch.tensor([i, j], device=xy.device)
        changes = torch.cat((a[:, :3 * d], b[:, :3 * d]), 0).tanh() * strength
        xy = xy.index_add(0, ij, changes)
        dm = torch.cat((a[:, 3 * d:], b[:, 3 * d:]), 0).tanh() * strength
        mass = F.relu(s.support.index_add(0, ij, dm))
        v, boundary, law = xy.split(d, -1)
        new_support = F.relu(child[:, 3 * d:]) * strength
        if float(new_support.detach().item()) == 0.0:
            return self.compact(replace(s, value=v, boundary=boundary,
                                        law=law, support=mass))
        new_id = s.next_id
        return self.compact(replace(s,
            value=torch.cat((v, child[:, :d].tanh()), 0),
            boundary=torch.cat((boundary, child[:, d:2 * d].tanh()), 0),
            law=torch.cat((law, child[:, 2 * d:3 * d].tanh()), 0),
            support=torch.cat((mass, new_support), 0),
            age=torch.cat((s.age, s.age.new_zeros((1, 1))), 0),
            ids=s.ids + (new_id,),
            links=s.links + ((source_id, new_id), (new_id, target_id)),
            next_id=new_id + 1))

    def compact(self, s: State) -> State:
        """Release ONLY exact-zero support. No fixed age, memory or value threshold."""
        keep = [i for i, mass in enumerate(s.support.detach().flatten()) if float(mass) > 0]
        ids = tuple(s.ids[i] for i in keep)
        available = set(ids)
        return replace(s, value=s.value[keep], boundary=s.boundary[keep],
                       law=s.law[keep], support=s.support[keep], age=s.age[keep],
                       ids=ids, links=tuple((a, b) for a, b in s.links
                                              if a in available and b in available))

    def attend(self, s: State, *, external: Tensor | None = None) -> State:
        """Explicitly couple observation to current structure, not every tick."""
        x = s.observer.new_zeros(self.d) if external is None else external
        if x.shape != (self.d,) or x.device != s.value.device:
            raise ValueError('invalid observation cue')
        if s.ids:
            keys = self._keys(s)
            q = self.observe_query(torch.cat((s.observer, x))).tanh()
            p = ((keys * q).sum(-1) / math.sqrt(self.hidden)).softmax(-1)
            content = (p[:, None] * s.value).sum(0)
        else:
            content = s.observer.new_zeros(self.d)
        updated = s.observer + self.observer_flow(
            torch.cat((s.observer, content, x))).tanh()
        return replace(s, observer=updated)

    def observe(self, s: State, *, external: Tensor | None = None) -> Tensor:
        """Optional observation: read current structure, not a text-token history."""
        x = s.observer.new_zeros(self.d) if external is None else external
        if x.shape != (self.d,):
            raise ValueError('invalid observation cue')
        q = self.observe_query(torch.cat((s.observer, x))).tanh()
        if s.ids:
            keys = self._keys(s)
            p = ((keys * q).sum(-1) / math.sqrt(self.hidden)).softmax(-1)
            content = (p[:, None] * s.value).sum(0)
        else:
            content = s.observer.new_zeros(self.d)
        return self.observe_out(torch.cat((content, q)))
