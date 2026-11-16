import sys
from pathlib import Path
from dataclasses import replace
import unittest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'frozen'))
from dno import DNO
from multiview_observer import compare, view, orthogonal_coordinate_check, snapshot_hash

class TestPerspective(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = DNO(d=3, hidden=12).eval()
        with torch.no_grad():
            self.a = self.model.introduce(self.model.blank(), torch.tensor([[1.,0.,0.],[0.,1.,0.]]))
    def test_local_differences_hidden_by_global_view(self):
        b=replace(self.a, value=self.a.value.flip(0))
        self.assertEqual(compare(self.a,b,views=['global_mean']).status, 'undetermined')
        t=compare(self.a,b,views=['global_mean','value'])
        self.assertEqual(t.first_distinguishing_view,'value')
    def test_law_difference_requires_switch(self):
        b=replace(self.a, law=self.a.law+0.1)
        t=compare(self.a,b,views=['value','law'])
        self.assertEqual(t.first_distinguishing_view,'law')
    def test_boundary_difference_requires_switch(self):
        b=replace(self.a, boundary=self.a.boundary+0.2)
        t=compare(self.a,b,views=['value','boundary'])
        self.assertEqual(t.first_distinguishing_view,'boundary')
    def test_age_and_time(self):
        b=replace(self.a, age=self.a.age+1)
        self.assertEqual(compare(self.a,b,views=['value','time']).first_distinguishing_view,'time')
    def test_observer(self):
        b=replace(self.a, observer=self.a.observer+1)
        self.assertEqual(compare(self.a,b,views=['value','observer']).first_distinguishing_view,'observer')
    def test_incidence(self):
        b=replace(self.a, links=((self.a.ids[0],self.a.ids[1]),))
        self.assertEqual(compare(self.a,b,views=['value','incidence']).first_distinguishing_view,'incidence')
    def test_canonical_id_order(self):
        ix=[1,0]
        b=replace(self.a, value=self.a.value[ix], law=self.a.law[ix], boundary=self.a.boundary[ix], support=self.a.support[ix], age=self.a.age[ix], ids=self.a.ids[::-1])
        for k in ('value','law','boundary','incidence','full'):
            np.testing.assert_allclose(view(self.a,k),view(b,k),atol=0)
    def test_orthogonal_coordinates_do_not_change_distance(self):
        a=np.array([1.,2.]);b=np.array([2.,-3.]);phi=0.73
        q=np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
        self.assertTrue(orthogonal_coordinate_check(a,b,q)['preserved'])
    def test_non_orthogonal_rejected(self):
        with self.assertRaises(ValueError): orthogonal_coordinate_check(np.ones(2),np.zeros(2),np.diag([1.,2.]))
    def test_never_claim_equivalence_from_matching_views(self):
        t=compare(self.a,self.a,views=['value','law','boundary'])
        self.assertEqual(t.status,'undetermined')
    def test_empty_future_without_model_abstains(self):
        t=compare(self.a,self.a,views=['future_value'])
        self.assertEqual(t.status,'undetermined')
        self.assertEqual(t.trace[0].finding,'unavailable')
    def test_genuine_changes_are_not_hidden(self):
        b=replace(self.a,value=self.a.value+0.5)
        self.assertEqual(compare(self.a,b,views=['full']).status,'difference_found')
    def test_dno_readonly(self):
        b=replace(self.a,law=self.a.law+.3)
        before=(snapshot_hash(self.a,self.model),snapshot_hash(b))
        t=compare(self.a,b,model=self.model,stop_on_difference=False)
        self.assertTrue(len(t.trace)>=5)
        after=(snapshot_hash(self.a,self.model),snapshot_hash(b))
        self.assertEqual(before,after)
    def test_future_view_returns_finite(self):
        v=view(self.a,'future_value',model=self.model)
        self.assertTrue(np.isfinite(v).all())
    def test_unknown_view_abstains(self):
        t=compare(self.a,self.a,views=['nonsense'])
        self.assertEqual(t.trace[0].finding,'unavailable')
    def test_empty_view_sequence_abstains(self):
        self.assertEqual(compare(self.a,self.a,views=[]).status,'undetermined')

if __name__=='__main__':unittest.main()
