"""Reproducible observational stress: NO updates to model, state, or weights."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import torch
from frozen.dno import DNO
from multiview_observer import compare, snapshot_hash


def audit(n_seeds: int = 160):
    results={"seeds":n_seeds,"cases_per_seed":8,"positive_cases":0,"positive_detected":0,
             "negative_cases":0,"negative_false_positive":0,"global_hidden_cases":0,
             "global_hidden_success":0,"identity_only_cases":0,"identity_only_missed_by_prior_full":0,
             "future_law_cases":0,"future_law_detected":0,"mutations":0,
             "numerical_only_changes_on_balanced_add":0,"large_tolerance_hidden":0,
             "per_field":{},"notes":[]}
    fields=['value','law','boundary','observer','age','support','links','identity']
    for k in fields:results['per_field'][k]={"cases":0,"found_by_relevant_view":0,"found_by_original_full":0}
    for seed in range(n_seeds):
        torch.manual_seed(seed+1200)
        m=DNO(d=4,hidden=10).eval()
        base=m.introduce(m.blank(),torch.randn(2,4)*.4)
        # snapshot readout must not change either model or state
        pre=snapshot_hash(base,m)
        scenarios={
            'value':replace(base,value=base.value.flip(0)),
            'law':replace(base,law=base.law+.21),
            'boundary':replace(base,boundary=base.boundary+.21),
            'observer':replace(base,observer=base.observer+.21),
            'age':replace(base,age=base.age+.21),
            'support':replace(base,support=base.support+.21),
            'links':replace(base,links=((0,1),)),
            'identity':replace(base,ids=(100,101),next_id=102),
        }
        focus={'value':'value','law':'law','boundary':'boundary','observer':'observer','age':'time',
               'support':'support','links':'incidence','identity':'full'}
        for key,changed in scenarios.items():
            results['positive_cases']+=1
            results['per_field'][key]['cases']+=1
            detected=compare(base,changed,views=[focus[key]],tolerance=1e-6).status=='difference_found'
            results['per_field'][key]['found_by_relevant_view']+=int(detected)
            original=compare(base,changed,views=['full'],tolerance=1e-6).status=='difference_found'
            results['per_field'][key]['found_by_original_full']+=int(original)
            results['positive_detected']+=int(detected)
            if key=='identity':
                results['identity_only_cases']+=1
                results['identity_only_missed_by_prior_full']+=int(not original)
            if key=='value':
                results['global_hidden_cases']+=1
                hidden=compare(base,changed,views=['global_mean'],tolerance=1e-6).status=='undetermined'
                results['global_hidden_success']+=int(hidden)
        results['negative_cases']+=2
        results['negative_false_positive']+=int(compare(base,base,views=['full']).status=='difference_found')
        # Relabel tensor rows and ID list consistently: semantically same carriers
        rearranged=replace(base,ids=base.ids[::-1],value=base.value.flip(0),boundary=base.boundary.flip(0),
                           law=base.law.flip(0),support=base.support.flip(0),age=base.age.flip(0))
        results['negative_false_positive']+=int(compare(base,rearranged,views=['full'],tolerance=1e-6).status=='difference_found')
        result=compare(base,scenarios['law'],views=['future_value'],model=m,dt=.1,tolerance=1e-7)
        results['future_law_cases']+=1
        results['future_law_detected']+=int(result.status=='difference_found')
        results['mutations']+=int(snapshot_hash(base,m)!=pre)
        # Nominally balanced perturbations can accumulate rounding noise under float32.
        balanced=torch.zeros_like(base.value)
        balanced[0,0]=1;balanced[1,0]=-1
        adjust=replace(base,value=base.value+balanced)
        f=compare(base,adjust,views=['global_mean'],tolerance=0)
        results['numerical_only_changes_on_balanced_add']+=int(f.status=='difference_found')
        tiny=replace(base,value=base.value+1e-5)
        results['large_tolerance_hidden']+=int(compare(base,tiny,views=['value'],tolerance=.01).status=='undetermined')
    results['notes']=[
     'Synthetic perturbations test the observer against known changes; they do not establish learned semantics or causal validity.',
     'A matching set of partial views cannot establish equivalence under all possible views.',
     'The original full view omits stable identity labels, which is measured as an observational blind spot, not a model defect.',
     'Future measurements use the frozen randomly initialized DNO dynamics, not environmental ground truth.',
     'Finite floating precision and tolerance change detection decisions.'
    ]
    return results

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=160);p.add_argument('--output',type=Path)
    a=p.parse_args();r=audit(a.seeds)
    text=json.dumps(r,ensure_ascii=False,indent=2)
    if a.output: a.output.write_text(text,encoding='utf-8')
    print(text)
