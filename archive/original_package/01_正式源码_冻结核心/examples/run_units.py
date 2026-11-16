"""Small execution audit across media; NO cross-modal performance benchmark.
Run: python examples/run_units.py --out evidence/multimodal_demo.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dno import DNO
from examples.units import image_units, video_units, audio_units, vector_units


def demo(seed: int = 2026):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    m = DNO(d=8, hidden=24)
    # Tiny procedural probes; the task data is not photographs or real recordings.
    yy,xx = torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')
    img = (xx*0.07+yy*0.11).sin()
    vid = torch.stack([(img + (t/7.) * (xx/8.)).sin() for t in range(4)], dim=0)
    t = torch.arange(256)/128.
    sound = (2*torch.pi*7*t).sin() + .2*(2*torch.pi*15*t).sin()
    symbolic = torch.stack([torch.linspace(-1,1,8).roll(i) for i in range(4)])
    vectors = {
        'image_patches':image_units(img),
        'video_cubes':video_units(vid),
        'audio_frames':audio_units(sound),
        'symbolic_vectors':vector_units(symbolic),
    }
    records = {}
    for name, values in vectors.items():
        s = m.introduce(m.blank(),values)
        before = s.value.clone()
        after = m.evolve(s,.1)
        after.validate()
        attended = m.attend(after)
        plan = m.propose(attended)
        reading = m.observe(attended)
        records[name] = dict(
            input_units=int(values.shape[0]), input_width=int(values.shape[1]),
            finite=bool(torch.isfinite(reading).all()),
            autonomous_value_change_l2=float((after.value-before).norm().detach().item()),
            observation_width=int(reading.numel()),
            proposed_source_probability_sum=float(plan.source.sum().detach().item()),
            proposed_target_probability_sum=float(plan.target.sum().detach().item()),
            # proposal is NOT automatically executed by the runtime.
            actual_interaction_committed=False,
        )
    return {
        'kind':'synthetic interface execution only; no training, no real-world quality estimate',
        'seed':seed, 'core_parameter_count':sum(x.numel() for x in m.parameters()),
        'same_untrained_model_parameters_for_all_inputs':True,
        'results':records,
    }


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='evidence/multimodal_demo.json')
    args=ap.parse_args()
    r=demo()
    p=Path(args.out);p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(r,indent=2,ensure_ascii=False))
