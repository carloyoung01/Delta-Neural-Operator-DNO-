"""Additional rigorous, read-only tests; not a new DNO mechanism."""
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'frozen')]
from dno import DNO
from multiview_observer import compare, view, snapshot_hash, orthogonal_coordinate_check

@pytest.fixture
def fixture():
    torch.manual_seed(903)
    m=DNO(d=4,hidden=12).eval()
    s=m.introduce(m.blank(),torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]],dtype=torch.float32))
    return m,s

def alter(s,kind):
    if kind=='value_swap': return replace(s,value=s.value[[1,0,2]])
    if kind=='balanced_value':
        delta=torch.zeros_like(s.value);delta[0,0]=.5;delta[1,0]=-.5
        return replace(s,value=s.value+delta)
    if kind=='law': return replace(s,law=s.law+.13)
    if kind=='boundary': return replace(s,boundary=s.boundary+.13)
    if kind=='observer': return replace(s,observer=s.observer+.13)
    if kind=='age': return replace(s,age=s.age+.13)
    if kind=='links': return replace(s,links=((s.ids[0],s.ids[1]),))
    if kind=='support': return replace(s,support=s.support+.13)
    raise ValueError(kind)

@pytest.mark.parametrize('kind,expected',[('value_swap','value'),('balanced_value','value'),('law','law'),('boundary','boundary'),('observer','observer'),('age','time'),('links','incidence'),('support','support')])
def test_switch_detects_hidden_difference(fixture,kind,expected):
    m,s=fixture;b=alter(s,kind)
    assert compare(s,b,views=['global_mean']).status=='undetermined' or kind=='value_swap'
    assert compare(s,b,views=['global_mean',expected]).first_distinguishing_view==expected or kind in ['observer','age','links','support'] and compare(s,b,views=[expected]).first_distinguishing_view==expected

@pytest.mark.parametrize('kind',[ 'value_swap','balanced_value','law','boundary','observer','age','links','support'])
def test_full_view_detects_all_existing_fields(fixture,kind):
    _,s=fixture
    assert compare(s,alter(s,kind),views=['full']).status=='difference_found'

def test_empty_list_not_equivalence(fixture):
    _,s=fixture
    a=compare(s,s,views=[])
    assert a.status=='undetermined' and len(a.trace)==0

def test_future_view_can_distinguish_hidden_law(fixture):
    m,s=fixture
    b=alter(s,'law')
    assert compare(s,b,views=['value']).status=='undetermined'
    assert compare(s,b,views=['future_value'],model=m,dt=.2).status=='difference_found'

def test_future_is_hypothetical_and_readonly(fixture):
    m,s=fixture;b=alter(s,'law')
    h=(snapshot_hash(s,m),snapshot_hash(b))
    compare(s,b,views=['future_value'],model=m,dt=.5)
    assert h==(snapshot_hash(s,m),snapshot_hash(b))

def test_zero_dt_future_may_not_unmask_law(fixture):
    m,s=fixture;b=alter(s,'law')
    # No time means value remains identical, so this future-view is insufficient.
    assert compare(s,b,views=['future_value'],model=m,dt=0).status=='undetermined'

def test_tolerance_hides_real_small_difference(fixture):
    _,s=fixture
    b=replace(s,value=s.value+1e-5)
    assert compare(s,b,views=['value'],tolerance=.1).status=='undetermined'
    assert compare(s,b,views=['value'],tolerance=1e-7).status=='difference_found'

def test_actual_equivalence_cannot_be_concluded(fixture):
    _,s=fixture
    b=replace(s,law=s.law+2)
    assert compare(s,b,views=['global_mean','value']).status=='undetermined'
    assert not torch.equal(s.law,b.law)

def test_relabelled_ids_is_a_known_blind_spot(fixture):
    _,s=fixture
    # Same row-wise values but different IDs; no directed links.
    # The OLD 'full' view does not include IDs and returns undetermined.
    b=replace(s,ids=(10,11,12),next_id=13)
    b.validate()
    assert s.ids!=b.ids
    assert compare(s,b,views=['full']).status=='undetermined'

def test_same_local_data_but_different_routing_links(fixture):
    _,s=fixture
    b=alter(s,'links')
    assert compare(s,b,views=['value','law','boundary']).status=='undetermined'
    assert compare(s,b,views=['incidence']).status=='difference_found'

def test_global_mean_is_order_invariant_but_local_ids_matter(fixture):
    _,s=fixture
    b=alter(s,'value_swap')
    assert np.array_equal(view(s,'global_mean'),view(b,'global_mean'))
    assert not np.array_equal(view(s,'value'),view(b,'value'))

def test_projection_collision_is_mathematically_unavoidable():
    a=np.array([1.,0.]);b=np.array([1.,5.]);P=np.array([[1.,0.]])
    assert np.array_equal(P@a,P@b)
    assert not np.array_equal(a,b)
    # Any function of the single observed coordinate must agree on both points.
    assert (P@a).sum()**3 == (P@b).sum()**3

def test_invertible_transform_preserves_distinguishability():
    A=np.array([[2.,1.],[1.,1.]])
    x=np.array([1.,2.]);y=np.array([1.,3.])
    assert np.linalg.det(A)!=0
    assert np.linalg.norm(A@x-A@y)>0
    np.testing.assert_allclose(np.linalg.solve(A,A@x),x)

def test_orthogonal_invariance_does_not_imply_model_equivariance(fixture):
    m,s=fixture
    torch.manual_seed(4)
    Q,_=torch.linalg.qr(torch.randn(4,4))
    s_rot=replace(s,value=s.value@Q.T,boundary=s.boundary@Q.T,law=s.law@Q.T,observer=s.observer@Q.T)
    after=m.evolve(s,.1).value@Q.T
    rotated_after=m.evolve(s_rot,.1).value
    assert torch.norm(after-rotated_after).item()>1e-5
    assert orthogonal_coordinate_check(np.array([1.,2.]),np.zeros(2),np.array([[0.,-1.],[1.,0.]]))['preserved']

def test_permuting_rows_with_ids_preserved_is_invisible(fixture):
    _,s=fixture
    ix=[2,0,1]
    b=replace(s,ids=tuple(s.ids[i] for i in ix),value=s.value[ix],boundary=s.boundary[ix],law=s.law[ix],support=s.support[ix],age=s.age[ix])
    b.validate()
    assert compare(s,b,views=['full']).status=='undetermined'

def test_link_direction_is_measured(fixture):
    _,s=fixture
    a=replace(s,links=((0,1),));b=replace(s,links=((1,0),))
    assert compare(a,b,views=['incidence']).status=='difference_found'

def test_invalid_missing_future_is_unavailable_not_evidence(fixture):
    _,s=fixture
    result=compare(s,alter(s,'law'),views=['future_value'])
    assert result.status=='undetermined' and result.trace[0].finding=='unavailable'

def test_hashes_unchanged_under_audit_and_rollouts(fixture):
    m,s=fixture;b=alter(s,'law')
    baseline=(snapshot_hash(s,m),snapshot_hash(b))
    for _ in range(5): compare(s,b,model=m,views=['full','future_value'],stop_on_difference=False)
    assert baseline==(snapshot_hash(s,m),snapshot_hash(b))

def test_view_order_changes_first_detection_not_state(fixture):
    m,s=fixture
    b=replace(s,value=s.value+.2,law=s.law+.3)
    first=compare(s,b,views=['law','value']).first_distinguishing_view
    second=compare(s,b,views=['value','law']).first_distinguishing_view
    assert first=='law' and second=='value'

def test_no_metric_means_no_global_error_claim(fixture):
    _,s=fixture
    result=compare(s,alter(s,'law'),views=['value'])
    assert result.status=='undetermined' and result.first_distinguishing_view is None
    assert not hasattr(result,'model_is_wrong')

@pytest.mark.parametrize('seed',[0,1,2,3,4])
def test_multiple_initializations_generalize_the_observation_logic(seed):
    torch.manual_seed(seed)
    model=DNO(d=5,hidden=12).eval()
    s=model.introduce(model.blank(),torch.randn(4,5))
    b=replace(s,value=s.value[[1,0,2,3]])
    assert compare(s,b,views=['global_mean']).status=='undetermined'
    assert compare(s,b,views=['global_mean','value']).first_distinguishing_view=='value'
