"""Minimal non-LLM demonstration of present state and composition."""
import torch
from dno import DNO


def main():
    torch.manual_seed(11)
    model = DNO(8, 24)
    # These labels are merely an application interpretation, not core tokens.
    state = model.introduce(model.blank(), torch.eye(8)[:3])
    print('current IDs:', state.ids)
    next_state = model.evolve(state, .2)
    print('without new external input, current state changed:',
          bool((next_state.value-state.value).abs().max() > 0))
    opportunity = model.propose(next_state)
    print('currently unfolded opportunity:',
          tuple(next_state.ids[i] for i in
                (int(opportunity.source.argmax()), int(opportunity.target.argmax()))))
    # This is an actual chosen action, not a simulated learned outcome.
    with torch.no_grad():
        model.generator[-1].weight[-1].zero_()
        model.generator[-1].bias[-1].fill_(1)
    combined = model.compose(next_state, 0, 1)
    combined.validate()
    print('after one generic composition: IDs', combined.ids, 'incidence', combined.links)


if __name__ == '__main__':
    main()
