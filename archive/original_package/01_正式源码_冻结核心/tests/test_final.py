"""Executable mathematical counterfactuals, not proofs of universal capability."""
from dataclasses import replace
from pathlib import Path
import inspect
import math
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dno import DNO


def make(n=3, seed=9, d=8):
    torch.manual_seed(seed)
    model = DNO(d=d, hidden=24)
    state = model.introduce(model.blank(), torch.randn(n, d) * .4)
    state.validate()
    return model, state


def force_child(model, positive=True):
    with torch.no_grad():
        head = model.generator[-1]
        head.weight[-1].zero_()
        head.bias[-1] = 1.0 if positive else -1.0


def test_blank_and_input_validation():
    m, s = make(0)
    s.validate()
    with pytest.raises(ValueError):
        m.propose(s)
    with pytest.raises(ValueError):
        m.introduce(s, torch.zeros(4, 4))


def test_input_not_a_history_log():
    m, s = make(2)
    new = m.evolve(s, .5, local={s.ids[0]: torch.ones(m.d)})
    new.validate()
    assert not hasattr(new, 'history')
    assert not hasattr(new, 'events')
    assert new.value.shape == s.value.shape


def test_deterministic_current_state_sufficiency():
    m, s = make(3)
    a = m.evolve(s, .08)
    b = m.evolve(s, .08)
    assert torch.equal(a.value, b.value)
    assert torch.equal(a.law, b.law)
    assert torch.equal(a.observer, b.observer)


def test_no_external_input_still_changes_state():
    m, s = make(3)
    after = m.evolve(s, .4)
    assert not torch.equal(after.value, s.value)
    assert not torch.equal(after.law, s.law)
    assert torch.equal(after.age, s.age + .4)


def test_invalid_time_rejected():
    m, s = make(2)
    for dt in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            m.evolve(s, dt)


def test_boundary_is_causal_condition():
    m, s = make(3)
    other = replace(s, boundary=s.boundary + .5)
    assert (m.evolve(s, .2).value - m.evolve(other, .2).value).norm().item() > 1e-6


def test_local_law_is_causal_condition():
    m, s = make(3)
    other = replace(s, law=s.law + .5)
    assert (m.evolve(s, .2).value - m.evolve(other, .2).value).norm().item() > 1e-6


def test_cue_is_local_not_broadcast():
    m, s = make(3)
    cue = {s.ids[0]: torch.ones(m.d)}
    a = m.evolve(s, .2)
    b = m.evolve(s, .2, local=cue)
    assert (a.value[0] - b.value[0]).abs().max() > 0
    assert torch.equal(a.value[1:], b.value[1:])


def test_link_has_no_auto_message_in_intrinsic_flow():
    m, s = make(2)
    with_link = replace(s, links=((s.ids[0], s.ids[1]),))
    a, b = m.evolve(s, .1), m.evolve(with_link, .1)
    assert torch.equal(a.value, b.value)
    assert torch.equal(a.law, b.law)
    assert torch.equal(a.boundary, b.boundary)


def test_compose_can_generate_real_new_carrier():
    m, s = make(2)
    force_child(m, True)
    after = m.compose(s, s.ids[0], s.ids[1])
    after.validate()
    assert len(after.ids) == 3
    assert after.links == ((s.ids[0], 2), (2, s.ids[1]))
    assert after.support[-1].item() > 0
    assert not torch.equal(after.value[:2], s.value)


def test_generic_child_itself_can_participate():
    m, s = make(2)
    force_child(m, True)
    a = m.compose(s, s.ids[0], s.ids[1])
    b = m.compose(a, a.ids[-1], a.ids[0])
    b.validate()
    assert len(b.ids) == 4
    assert (2, 3) in b.links


def test_no_new_child_is_possible():
    m, s = make(2)
    force_child(m, False)
    after = m.compose(s, s.ids[0], s.ids[1])
    after.validate()
    assert len(after.ids) == 2
    assert not after.links
    assert not torch.equal(after.value, s.value)


def test_zero_strength_no_action():
    m, s = make(2)
    a = m.compose(s, s.ids[0], s.ids[1], strength=0)
    assert a is s


def test_stable_identity_after_compression():
    m, s = make(3)
    force_child(m, True)
    s = m.compose(s, 0, 1)
    zero = torch.cat((s.support[:1], s.support.new_zeros((1, 1)), s.support[2:]))
    after = m.compact(replace(s, support=zero))
    after.validate()
    assert after.ids == (0, 2, 3)
    assert all(1 not in link for link in after.links)
    more = m.introduce(after, torch.zeros(m.d))
    assert more.ids == (0, 2, 3, 4)


def test_existing_structure_changes_future_address():
    m, s = make(3)
    force_child(m, True)
    s = m.compose(s, 0, 1)
    other = replace(s, value=torch.cat((s.value[:-1], s.value[-1:] + 2.)),
                    law=torch.cat((s.law[:-1], s.law[-1:] - 1.)))
    p, q = m.propose(s), m.propose(other)
    assert (p.source - q.source).norm().item() > 1e-7 or (p.target - q.target).norm().item() > 1e-7


def test_incidence_changes_future_address():
    m, s = make(3)
    q = replace(s, links=((0, 1),))
    a, b = m.propose(s), m.propose(q)
    assert (a.source - b.source).norm().item() > 1e-7


def test_current_observer_is_endogenous_input():
    m, s = make(3)
    q = replace(s, observer=s.observer + torch.linspace(-1, 1, m.d))
    a, b = m.propose(s), m.propose(q)
    assert (a.source - b.source).norm().item() > 1e-7


def test_no_compulsory_contact():
    m, s = make(3)
    after = m.evolve(s, .2)
    assert after.ids == s.ids and after.links == s.links


def test_zero_and_positive_activity_both_possible():
    m, s = make(3)
    with torch.no_grad():
        head = m.activity[-1]
        head.weight.zero_()
        head.bias.fill_(-1.)
    assert m.propose(s).activity.item() == 0
    with torch.no_grad():
        head.bias.fill_(1.)
    assert m.propose(s).activity.item() == 1


def test_compressed_possibility_differs_by_current_choice():
    m, s = make(3)
    force_child(m, True)
    a = m.compose(s, 0, 1)
    b = m.compose(s, 0, 2)
    assert a.links != b.links
    assert len(s.links) == 0
    assert len(a.ids) == len(b.ids) == 4


def test_optional_observation_is_not_a_required_step():
    m, s = make(2)
    after = m.evolve(s, .2)
    assert after.value.shape == s.value.shape
    y = m.observe(after)
    assert y.shape == (m.d,)
    assert bool(torch.isfinite(y).all())


def test_autograd_through_shared_generator():
    m, s = make(3)
    after = m.evolve(s, .2, local={0: torch.ones(m.d)})
    loss = (after.value ** 2).sum() + (after.law ** 2).sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0
               for p in m.generator.parameters())


def test_support_zero_compacts_without_age_threshold():
    m, s = make(2)
    s = replace(s, support=s.support * 0, age=s.age + 5000)
    after = m.compact(s)
    assert len(after.ids) == 0
    assert after.next_id == s.next_id


def test_errors_dont_silently_change_identities():
    m, s = make(3)
    with pytest.raises(KeyError):
        m.compose(s, 0, 999)
    with pytest.raises(KeyError):
        m.evolve(s, .2, local={999: torch.ones(m.d)})
    with pytest.raises(ValueError):
        m.compose(s, 0, 1, strength=float('nan'))


def test_internal_features_remain_finite_short_trajectory():
    m, s = make(3)
    force_child(m, False)
    for _ in range(45):
        s = m.evolve(s, .05)
        s.validate()
    assert (s.age > 2).all()


def test_no_event_names_or_all_pairs_catalog():
    source = inspect.getsource(DNO)
    assert 'EVENT_KINDS' not in source
    assert '_proposals' not in source
    assert 'for i in range(n)' not in source
    assert 'for j in range(n)' not in source


def test_state_is_nonhistorical_but_has_a_model_state():
    m, s = make(2)
    history_a = m.evolve(s, .1)
    # A new call is based on the current state, without receiving either input log.
    next_a = m.evolve(history_a, .2)
    next_b = m.evolve(history_a, .2)
    assert torch.equal(next_a.value, next_b.value)
    assert len(history_a.ids) == 2


def test_not_observed_means_no_global_state_leak():
    m, s = make(3)
    alternative = replace(s, value=torch.cat((s.value[:2], s.value[2:] + 5), dim=0))
    a, b = m.evolve(s, .2), m.evolve(alternative, .2)
    assert torch.equal(a.value[:2], b.value[:2])
    assert torch.equal(a.observer, b.observer)


def test_explicit_observation_can_modify_future_address():
    m, s = make(3)
    a = m.attend(s)
    b = m.attend(replace(s, value=s.value + 1.5))
    assert not torch.equal(a.observer, b.observer)
    assert (m.propose(a).source - m.propose(b).source).abs().max() > 1e-7


def test_automatic_exact_zero_compression_after_contact():
    m, s = make(2)
    force_child(m, False)
    with torch.no_grad():
        head = m.generator[-1]
        # Parent support updates are outputs 3D and (3D+1)+3D.
        for ix in (3*m.d, 2*(3*m.d+1)-1):
            head.weight[ix].zero_()
            head.bias[ix] = -10
    # Large strength makes both existing carriers lose all support.
    after = m.compose(s, 0, 1, strength=3.)
    assert not after.ids and not after.links
    after.validate()


def test_zero_support_excluded_from_address_before_gc():
    m, s = make(3)
    dead = replace(s, support=torch.cat((s.support[:2], s.support.new_zeros((1,1)))))
    plan = m.propose(dead)
    assert plan.source[-1].item() == plan.target[-1].item() == 0
    assert plan.source[:2].sum().item() > .999999


def test_relation_carrier_can_evolve_without_touching_its_endpoints():
    m, s = make(2)
    force_child(m, True)
    s = m.compose(s, 0, 1)
    altered = replace(s, value=torch.cat((s.value[:2], s.value[2:] + 3), 0))
    a, b = m.evolve(s, .05), m.evolve(altered, .05)
    assert torch.equal(a.value[:2], b.value[:2])
    assert not torch.equal(a.value[2:], b.value[2:])
