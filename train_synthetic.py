"""Small supervised probes of the same compressed DNO core (not LLM training).

Result-only task 1: discover which two current carriers a query refers to.
Result-only task 2: keep one bit across distractors in the current state.
No event, address, memory-anchor, or history-log labels in the training losses.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from dno import DNO


def data_pair(batch: int, *, device='cpu'):
    """Four permuted symbolic identities; random numeric values; two queries."""
    b, n, d = batch, 4, 8
    permutations = torch.rand(b, n, device=device).argsort(-1)
    v = torch.randn((b, n), device=device)
    x = torch.zeros(b, n, d, device=device)
    x[:, :, :4] = F.one_hot(permutations, 4).float()
    x[:, :, 4] = v
    keys = torch.randint(n, (b, 2), device=device)
    q = torch.zeros(b, d, device=device)
    q[:, :4] = F.one_hot(keys[:, 0], 4).float()
    q[:, 4:] = F.one_hot(keys[:, 1], 4).float()
    values_at_key = torch.zeros(b, n, device=device).scatter(1, permutations, v)
    y = values_at_key.gather(1, keys[:, 1:2]).squeeze(1) - values_at_key.gather(1, keys[:, 0:1]).squeeze(1)
    expected_source = (permutations == keys[:, 0:1]).float().argmax(-1)
    expected_target = (permutations == keys[:, 1:2]).float().argmax(-1)
    return x, q, y, expected_source, expected_target


def pair_forward(m: DNO, x: torch.Tensor, q: torch.Tensor):
    """Same key & query weights as propose(), vectorized for independent samples."""
    b, n, d = x.shape
    z = torch.cat((x, m.seed_boundary(x).tanh(), m.seed_law(x).tanh(),
                   x.new_ones((b, n, 1)), x.new_zeros((b, n, 1))), -1)
    keys = m.key(z).tanh()
    center = keys.mean(-2)
    o = q.new_zeros((b, d))
    sq = m.source_query(torch.cat((o, center, q), -1)).tanh()
    sp = ((keys * sq[:, None]).sum(-1) / (m.hidden ** .5)).softmax(-1)
    sk = (keys * sp[..., None]).sum(-2)
    tq = m.target_query(torch.cat((o, center, q, sk), -1)).tanh()
    tp = ((keys * tq[:, None]).sum(-1) / (m.hidden ** .5)).softmax(-1)
    values = x[:, :, 4]
    predicted = (values * tp).sum(-1) - (values * sp).sum(-1)
    return predicted, sp, tp


def train_pair(seed: int, steps: int, out: Path):
    torch.manual_seed(seed)
    m = DNO(8, 40)
    optimizer = torch.optim.Adam(m.parameters(), lr=.008)
    for _ in range(steps):
        x, q, y, _, _ = data_pair(128)
        prediction, _, _ = pair_forward(m, x, q)
        loss = F.mse_loss(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.manual_seed(seed + 10000)
    with torch.no_grad():
        x, q, y, a, b = data_pair(2048)
        prediction, sp, tp = pair_forward(m, x, q)
        result = {'seed': seed, 'steps': steps,
                  'mse': float(F.mse_loss(prediction, y)),
                  'source_acc': float((sp.argmax(-1) == a).float().mean()),
                  'target_acc': float((tp.argmax(-1) == b).float().mean())}
        # Removing the query destroys the task reference, not the current carriers.
        no_query, _, _ = pair_forward(m, x, torch.zeros_like(q))
        result['no_query_mse'] = float(F.mse_loss(no_query, y))
    torch.save({'core': m.state_dict(), 'result': result}, out / f'pair_{seed}.pt')
    return result


def sequence_batch(batch: int, delay: int):
    label = torch.randint(2, (batch,))
    first = torch.zeros((batch, 8))
    first.scatter_(1, label[:, None], 1.)
    nuisance = torch.zeros((delay, batch, 8))
    nuisance[:, :, 2:6] = torch.randn((delay, batch, 4)) * .7
    return label, first, nuisance


def history_forward(m: DNO, classifier: nn.Module, first: torch.Tensor,
                    nuisance: torch.Tensor):
    b, d = first.shape
    x = first.new_zeros((b, d))
    boundary = m.seed_boundary(x).tanh()
    law = m.seed_law(x).tanh()
    support = first.new_ones((b, 1))
    age = first.new_zeros((b, 1))
    cues = [first] + list(nuisance.unbind(0))
    dt = .22
    for cue in cues:
        z = torch.cat((x, boundary, law, support, age), -1)
        delta, _, _ = m._call(z, torch.zeros_like(z), torch.zeros_like(x),
                              cue, z.new_full((b, 1), dt))
        x = x + dt * delta[:, :d].tanh()
        boundary = boundary + dt * delta[:, d:2*d].tanh()
        law = law + dt * delta[:, 2*d:3*d].tanh()
        support = F.relu(support + dt * delta[:, 3*d:].tanh())
        age = age + dt
    return classifier(torch.cat((x, boundary, law, support, age), -1))


def train_history(seed: int, steps: int, out: Path):
    torch.manual_seed(seed)
    m = DNO(8, 40)
    classifier = nn.Linear(3*8 + 2, 2)
    optimizer = torch.optim.Adam(list(m.parameters()) + list(classifier.parameters()), lr=.006)
    for step in range(steps):
        label, first, nuisance = sequence_batch(128, 2 + step % 6)
        logit = history_forward(m, classifier, first, nuisance)
        loss = F.cross_entropy(logit, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.manual_seed(seed + 20000)
    result = {'seed': seed, 'steps': steps, 'test_accuracy': {}}
    with torch.no_grad():
        for delay in (3, 7, 15):
            label, first, nuisance = sequence_batch(1024, delay)
            outp = history_forward(m, classifier, first, nuisance)
            result['test_accuracy'][str(delay)] = float((outp.argmax(-1)==label).float().mean())
        label, first, nuisance = sequence_batch(1024, 7)
        logits = history_forward(m, classifier, torch.zeros_like(first), nuisance)
        result['removed_history_accuracy'] = float((logits.argmax(-1)==label).float().mean())
    torch.save({'core': m.state_dict(), 'head': classifier.state_dict(),
                'result': result}, out / f'history_{seed}.pt')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='evidence')
    parser.add_argument('--pair-steps', type=int, default=400)
    parser.add_argument('--history-steps', type=int, default=300)
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {'pair': [], 'history': []}
    for seed in (13, 29):
        r = train_pair(seed, args.pair_steps, out)
        results['pair'].append(r)
        print('pair', r, flush=True)
    for seed in (17, 31):
        r = train_history(seed, args.history_steps, out)
        results['history'].append(r)
        print('history', r, flush=True)
    (out / 'training_results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
