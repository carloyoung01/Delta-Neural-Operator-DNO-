"""Independent observation extension for rigorous blind-spot testing."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import torch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'frozen')]
from dno import DNO
from multiview_observer import compare,snapshot_hash
from extended_readonly_views import identity_distinguished,temporal_value_signature,address_signature

@pytest.fixture
def sample():
    torch.manual_seed(910)
    m=DNO(d=3,hidden=14).eval()
    s=m.introduce(m.blank(),torch.tensor([[.1,.2,-.3],[.4,-.2,.7]]))
    return m,s

def test_identity_view_detects_original_omission(sample):
    m,s=sample
    b=replace(s,ids=(11,12),next_id=13)
    assert compare(s,b,views=['full']).status=='undetermined'
    assert identity_distinguished(s,b)

def test_identity_measure_ignores_reordering(sample):
    _,s=sample
    ix=[1,0]
    b=replace(s,ids=s.ids[::-1],value=s.value[ix],law=s.law[ix],boundary=s.boundary[ix],support=s.support[ix],age=s.age[ix])
    assert not identity_distinguished(s,b)

def test_next_id_is_allocator_not_effective_carrier_label(sample):
    _,s=sample
    b=replace(s,next_id=100)
    assert not identity_distinguished(s,b)

def test_temporal_law_difference_emerges_without_modifying_state(sample):
    m,s=sample
    b=replace(s,law=s.law+.6)
    hash_before=(snapshot_hash(s,m),snapshot_hash(b))
    x=temporal_value_signature(m,s,steps=4,dt=.12)
    y=temporal_value_signature(m,b,steps=4,dt=.12)
    np.testing.assert_array_equal(x[0],y[0])
    assert any(np.linalg.norm(a-b)>1e-6 for a,b in zip(x[1:],y[1:]))
    assert hash_before==(snapshot_hash(s,m),snapshot_hash(b))

def test_zero_time_samples_cannot_reveal_law_with_value_only(sample):
    m,s=sample
    b=replace(s,law=s.law+.6)
    x=temporal_value_signature(m,s,steps=4,dt=0)
    y=temporal_value_signature(m,b,steps=4,dt=0)
    assert all(np.array_equal(a,b) for a,b in zip(x,y))

def test_missing_history_cannot_be_recovered_by_reformatting_one_snapshot(sample):
    _,s=sample
    # Two distinct trajectories may end in the exact same provided State.
    # A deterministic function of the final snapshot must give the same output.
    assert not identity_distinguished(s,s)
    assert np.array_equal(s.value.detach().numpy(),s.value.detach().numpy())

def test_link_is_inert_during_passive_evolve_but_active_during_address(sample):
    m,s=sample
    b=replace(s,links=((0,1),))
    with torch.no_grad():
        a_future=m.evolve(s,.1);b_future=m.evolve(b,.1)
    assert torch.equal(a_future.value,b_future.value)
    assert np.linalg.norm(address_signature(m,s)-address_signature(m,b))>1e-7

def test_view_switch_is_not_an_exogenous_world_measurement(sample):
    m,s=sample
    before=snapshot_hash(s,m)
    temporal_value_signature(m,s,steps=3)
    address_signature(m,s)
    assert snapshot_hash(s,m)==before

def test_invalid_temporal_samples_rejected(sample):
    m,s=sample
    for steps,dt in ((-1,.1),(1,-1),(1,float('nan'))):
        with pytest.raises(ValueError): temporal_value_signature(m,s,steps=steps,dt=dt)
