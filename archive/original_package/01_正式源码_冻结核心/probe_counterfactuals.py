"""Reproducible counterfactuals on untrained shared laws, not trained semantics."""
from dataclasses import replace
import json
from pathlib import Path
import torch
from dno import DNO


def main():
    torch.set_num_threads(1)
    data = []
    for seed in range(64):
        torch.manual_seed(seed)
        m = DNO(8, 24)
        s = m.introduce(m.blank(), torch.randn(3, 8)*.4)
        act = m.compose(s, 0, 1)
        newborn = len(act.ids) - 3
        if newborn:
            mass = act.support.clone()
            mass[-1] = 0
            now = m.compact(replace(act, support=mass))
        else:
            now = act
        # The original contact has been released as an explicit link/carrier,
        # but its consequences in the parents' present values remain.
        p = m.evolve(now, .08)
        q = m.evolve(s, .08)
        difference = (p.value[:3]-q.value).norm().item()
        # Truly uncontacted relations do not continually message to participants.
        if newborn:
            modified = replace(act, value=torch.cat((act.value[:3], act.value[-1:]+2), 0))
            own_a, own_b = m.evolve(act, .03), m.evolve(modified, .03)
            silent = (own_a.value[:3] - own_b.value[:3]).abs().max().item()
            relation_effect = (m.propose(act).source - m.propose(modified).source).norm().item()
        else:
            silent, relation_effect = None, None
        data.append({'seed': seed, 'new_carrier_created': bool(newborn),
                     'effect_after_relation_removed': difference,
                     'silent_flow_effect': silent,
                     'relation_changes_future_address': relation_effect})
    summary = {
        'seeds': len(data),
        'created_new_carrier': sum(t['new_carrier_created'] for t in data),
        'history_effect_nonzero': sum(t['effect_after_relation_removed'] > 1e-8 for t in data),
        'silent_flow_exact_zero': sum(t['silent_flow_effect'] == 0 for t in data if t['silent_flow_effect'] is not None),
        'current_relation_changes_address': sum((t['relation_changes_future_address'] or 0) > 1e-8 for t in data),
        'details': data,
    }
    path = Path(__file__).resolve().parent/'evidence'/'counterfactuals.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='details'},ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
