# Code-to-principle and evidence map

| Research statement | Code | Current evidence | Interpretation |
|---|---|---|---|
| Current state carries effective history | `State`, `evolve()`, `compose()` | `tests/test_final.py`, `evidence/counterfactuals.json` | Present-state consequences, not full event logs |
| Relative autonomous evolution | `evolve()`, `generator` | Core tests | A call can advance without new external input; not background computation |
| Shared generative law | `_call()` / `generator` | Core tests | Both autonomous evolution and two-carrier composition use shared parameters |
| Interaction changes participants | `compose()` | 64-seed counterfactuals | Contact can leave future-relevant state differences |
| New carrier may participate again | `compose()` and common state fields | Core tests | Actual structure formation, not a predeclared relationship taxonomy |
| Retention/release | `support`, `compact()` | Core tests | Precise zero support triggers release; no global optimality claim |
| Observation and non-observation | `observer`, `evolve()`, `attend()`, `observe()` | Core tests | Observation state evolves; explicit observation is optional |
| Opportunity need not execute | `propose()` versus `compose()` | Core tests | Learned addressing is distinct from committed interaction |
| Potential successors need not be enumerated | Current state + operations | Formal description | No exhaustive prospective state store exists in current code |
| Conditional addressing can be trained | `train_synthetic.py` / `pair_forward()` | New pair checkpoints, fresh metrics | Outcome-supervised synthetic numerical task |
| Current state can preserve a simple delayed signal | `history_forward()` and DNO generator | New history checkpoints, fresh metrics | Controlled one-bit delay task |
| General future research | `docs/RESEARCH_OUTLOOK.md` | Hypotheses, not results | No new algorithm/module required by the proposed directions |

The code pathways in current tasks are **not** an end-to-end training of every available operation. The test suites and training probes address different behaviors and are labeled accordingly.
