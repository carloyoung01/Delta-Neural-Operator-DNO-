"""Independently reload all saved probes and evaluate freshly sampled controls."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from dno import DNO
from train_synthetic import data_pair, pair_forward, sequence_batch, history_forward


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent / 'evidence'
    checks = {'pair': [], 'history': []}
    for path in sorted(root.glob('pair_*.pt')):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        torch.manual_seed(int(checkpoint['result']['seed']) + 30000)
        model = DNO(8, 40)
        model.load_state_dict(checkpoint['core'])
        model.eval()
        with torch.no_grad():
            x, q, y, ia, ib = data_pair(1024)
            result, pa, pb = pair_forward(model, x, q)
            mse = float(F.mse_loss(result, y))
            src = float((pa.argmax(-1) == ia).float().mean())
            dst = float((pb.argmax(-1) == ib).float().mean())
        assert mse < .05 and src > .97 and dst > .97, (path, mse, src, dst)
        checks['pair'].append({'checkpoint': path.name, 'fresh_mse': mse,
                               'source_acc': src, 'target_acc': dst})
    for path in sorted(root.glob('history_*.pt')):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        torch.manual_seed(int(checkpoint['result']['seed']) + 40000)
        model = DNO(8, 40)
        head = nn.Linear(3*8 + 2, 2)
        model.load_state_dict(checkpoint['core'])
        head.load_state_dict(checkpoint['head'])
        with torch.no_grad():
            y, first, nuisance = sequence_batch(1024, 10)
            a = float((history_forward(model, head, first, nuisance).argmax(-1)==y).float().mean())
            b = float((history_forward(model, head, torch.zeros_like(first), nuisance).argmax(-1)==y).float().mean())
        assert a > .95 and .4 < b < .6, (path, a, b)
        checks['history'].append({'checkpoint': path.name, 'fresh_acc': a,
                                  'no_first_input_acc': b})
    assert len(checks['pair']) == 2 and len(checks['history']) == 2
    (root / 'checkpoint_verification.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
