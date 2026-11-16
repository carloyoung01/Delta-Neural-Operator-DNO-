"""Read-only external experiment. NOT a model module or a new DNO operator."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from dataclasses import replace
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dno import DNO
from multiview_observer import compare, snapshot_hash
from extended_readonly_views import identity_distinguished


def main():
    torch.manual_seed(913)
    model = DNO(d=4, hidden=12).eval()
    with torch.no_grad():
        state = model.introduce(model.blank(), torch.tensor(
            [[1., 0., 0., 0.], [0., 1., 0., 0.]]))
        switched = replace(state, value=state.value[[1, 0]])
        law_changed = replace(state, law=state.law + .3)
        renumbered = replace(state, ids=(9, 10), next_id=11)
        before = snapshot_hash(state, model)
        records = {
            'global_only': compare(state, switched, views=['global_mean']).status,
            'global_then_local': compare(state, switched,
                views=['global_mean', 'value']).first_distinguishing_view,
            'content_only': compare(state, law_changed, views=['value']).status,
            'content_then_law': compare(state, law_changed,
                views=['value', 'law']).first_distinguishing_view,
            'old_full_misses_ids': compare(state, renumbered,
                views=['full']).status,
            'identity_observer_finds_ids': identity_distinguished(state, renumbered),
            'all_checked_and_same': compare(state, state,
                views=['global_mean', 'value', 'law', 'boundary', 'incidence']).status,
            'read_only': before == snapshot_hash(state, model),
        }
    print(json.dumps(records, ensure_ascii=False, indent=2))
    assert records['global_only'] == 'undetermined'
    assert records['global_then_local'] == 'value'
    assert records['content_only'] == 'undetermined'
    assert records['content_then_law'] == 'law'
    assert records['old_full_misses_ids'] == 'undetermined'
    assert records['identity_observer_finds_ids']
    assert records['all_checked_and_same'] == 'undetermined'
    assert records['read_only']
    return records


if __name__ == '__main__':
    main()
