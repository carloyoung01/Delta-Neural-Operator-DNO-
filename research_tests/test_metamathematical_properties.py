"""Metamorphic checks: observational invariances, approximate time scale, limitations."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import torch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'frozen')]
from dno import DNO
from multiview_observer import view,snapshot_hash,compare

@pytest.mark.parametrize('seed',range(20))
def test_id_preserving_permutation_preserves_model_address_distribution(seed):
    torch.manual_seed(seed+501)
    m=DNO(d=3,hidden=13).eval()
    s=m.introduce(m.blank(),torch.randn(3,3))
    s=replace(s,links=((0,1),(1,2)))
    ix=[2,0,1]
    t=replace(s,ids=tuple(s.ids[i] for i in ix), value=s.value[ix], law=s.law[ix],
              boundary=s.boundary[ix], support=s.support[ix],age=s.age[ix])
    with torch.no_grad():
        p=m.propose(s);q=m.propose(t)
    # probability coordinates follow array rows; compare by identity
    inverse=[t.ids.index(uid) for uid in s.ids]
    assert torch.allclose(p.source,q.source[inverse],atol=2e-7)
    assert torch.allclose(p.target,q.target[inverse],atol=2e-7)
    assert abs(p.activity.item()-q.activity.item())<2e-7

@pytest.mark.parametrize('seed',range(10))
def test_dno_one_large_step_vs_two_small_steps_not_identical(seed):
    torch.manual_seed(seed+701)
    m=DNO(d=3,hidden=15).eval()
    s=m.introduce(m.blank(),torch.randn(2,3)*.4)
    with torch.no_grad():
        large=m.evolve(s,.4)
        small=m.evolve(m.evolve(s,.2),.2)
    assert float(torch.norm(large.value-small.value))>0
    # A time resolution change can change the observed result; it is not a proof
    # of new physics or of defective mathematics.

@pytest.mark.parametrize('seed',range(10))
def test_passive_flow_does_not_automatically_read_new_links(seed):
    torch.manual_seed(seed+801)
    m=DNO(d=3,hidden=15).eval()
    s=m.introduce(m.blank(),torch.randn(2,3)*.4)
    b=replace(s,links=((0,1),))
    with torch.no_grad():
        a2=m.evolve(s,.3);b2=m.evolve(b,.3)
    assert torch.equal(a2.value,b2.value)
    assert torch.equal(a2.law,b2.law)
    assert torch.equal(a2.boundary,b2.boundary)

@pytest.mark.parametrize('seed',range(10))
def test_passive_flow_does_not_change_observer_on_carrier_only_difference(seed):
    torch.manual_seed(seed+901)
    m=DNO(d=3,hidden=15).eval()
    s=m.introduce(m.blank(),torch.randn(3,3)*.4)
    b=replace(s,value=s.value+.8)
    with torch.no_grad():
        a2=m.evolve(s,.4);b2=m.evolve(b,.4)
    assert torch.equal(a2.observer,b2.observer)
    assert not torch.equal(a2.value,b2.value)
