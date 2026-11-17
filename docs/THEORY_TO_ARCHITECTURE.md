# Delta Neural Operator (DNO): Theory-to-Architecture Mapping

**Author:** Carlo Young  
**Project:** Delta Neural Operator (DNO)  
**License:** Apache License 2.0  
**Status:** Research documentation accompanying the DNO 1.0 series; not a new implementation or verification run.

## 1. Scope and positioning

Delta Unified Recursive Generative Theory is a mathematical and conceptual research framework for effective state, interactions, relations, structural formation, historical consequences, and changing conditions of future generation.

**DNO is a concrete, trainable experimental neural architecture exploring a subset of these principles.** It is not a pretrained commercial large language model. Current tests do not establish general intelligence, universal self-modification, or performance superiority over Transformers. The repository's [architecture specification](ARCHITECTURE.md) remains authoritative for actual implementation.

## 2. General Delta formulation

The abstract system state is

$$
\Omega_n=(U_n,R_n,S_n,X_n),
$$

where the terms represent units, relations, organization, and effective current state. A broader conceptual description is

$$
\Xi_n=(\Omega_n,P_n,O_n,\Theta_n),\qquad
\Xi_{n+1}\in\mathfrak D(\Xi_n,E_n).
$$

Here $P_n$ describes conditional successor possibilities, $O_n$ observation conditions, and $\Theta_n$ effective generative conditions. This is **theoretical notation**: DNO does not necessarily store these as separate runtime objects.

Four perspectives—**carrying, relating, transforming, and openness**—describe aspects of generation, not four mandatory execution stages.

## 3. Actual DNO state

The executable architecture specifies

$$
\Omega_t=\left(\{(i,x_i,b_i,\ell_i,\mu_i,\tau_i)\}_{i\in I_t},L_t,o_t\right).
$$

- $i$: carrier identity;
- $x_i$: effective content;
- $b_i$: boundary features;
- $\ell_i$: mutable local-law features;
- $\mu_i$: nonnegative support;
- $\tau_i$: local elapsed time;
- $L_t$: current incidence/links;
- $o_t$: observation state.

The implementation also uses `next_id` for identity allocation. A carrier's `law` is **mutable state**, not a claim that neural network weights are rewritten during inference.

## 4. Shared learnable generation

The architecture summarizes its shared generator as

$$
(\delta a,\delta b,c)=G_\theta(a,b;o,e,\Delta t).
$$

The generator is invoked for single-carrier local evolution and executed two-carrier composition. Outputs are interpreted by the concrete operation. The overall recurrence is shorthand for

$$
\Omega_{t+\Delta t}=\mathrm{Canon}(\mathrm{Apply}_{G_\theta}(\Omega_t,e_t,a_t,\Delta t)).
$$

This does **not** introduce a new global controller. `Canon` describes current-state housekeeping, especially exact-zero-support release.

## 5. Theory-to-code correspondence

| Delta concept | DNO implementation or representation | Evidence level |
|---|---|---|
| Effective current state carries historical consequences | Carrier `content`, `boundary`, `law`, `support`, `links`, observer state | Implemented mechanism |
| Relatively autonomous local evolution | `evolve()` | Implemented mechanism |
| Executed interaction changes participants | `compose()` | Implemented mechanism |
| Interaction can form a reusable carrier | Conditional new carrier from `compose()` | Implemented; creation not guaranteed |
| Conditional addressing without execution | `propose()` | Implemented mechanism |
| Introduction of numerical carriers | `introduce()` | Implemented mechanism |
| Release of zero-support structures | `compact()` | Implemented; not arbitrary optimal forgetting |
| Active observation versus passive reading | `attend()` versus `observe()` | Implemented distinction |
| Evolution without mandatory active observation | `evolve()` and observer flow | Implemented mechanism |
| Conditional successor space | Mathematical image of allowed operations | Abstract characterization; no separate enumerator |
| Open-ended creation of new mathematical rule types | Not established by current tests | Research question |
| Complete structure–possibility–observation coevolution | Not a separately verified universal operator | Theoretical extension |
| Arbitrary inference-time neural-weight rewriting | Not claimed | Outside verified mechanism |
| Semantic learning across language, vision, audio, video | Numeric interfaces exist; semantic multimodal training not demonstrated by interface tests | Future research |

## 6. Historical consequences without mandatory event logs

Delta treats history as effective consequences carried by the present. In DNO, past interactions can modify current carrier fields and links, which then condition subsequent actions. The architecture does not require a complete ever-growing event log for these local consequences.

This does **not** prove that finite state suffices for all future tasks or that compression is always lossless.

## 7. Observation and conditional successors

Observation is not mandatory at each update. `observe()` reads current state, `attend()` actively couples structure into observational state, and observation state can evolve through the documented observer flow.

Conditional successor possibilities can be characterized as

$$
\mathcal P_t=\mathrm{Im}\left[(e,a)\mapsto\mathrm{Canon}(\mathrm{Apply}_{G_\theta}(\Omega_t,e,a))\right].
$$

The expression does not mean all possible successors are computed, stored, or enumerated.

## 8. Existing repository evidence (v1.0.0)

The repository's [README](../README.md) and [experimental report](EXPERIMENTS.md) document these **specified** results:

| Verification | Reported result |
|---|---:|
| Core tests | 32/32 passed |
| Numerical interface tests | 16/16 passed |
| Optional research tests | 114/114 passed |
| **Total specified checks** | **162/162 passed** |
| Fresh checkpoint load-and-evaluate checks | **4/4 passed** |
| 64-seed structural counterfactual: new carriers | 30/64 |
| 64-seed structural counterfactual: detectable interaction effects | 64/64 |

The repository also reports successful results on defined pair-addressing and delayed one-bit synthetic tasks. These are **task-specific, repository-reported results**, not an independent reproduction for this document. The 114 optional research tests are not 114 trained skills.

See [experiments](EXPERIMENTS.md), [training and data](TRAINING_AND_DATA.md), [provenance](PROVENANCE.md), and `verification/current_run/` for procedures and records.

## 9. Mathematical research program

Future formal work may specify admissible classes of evolving structural states; equivalence relations for effective historical sufficiency; interaction and path-dependence measures; conditions under which new carriers are reusable; distinctions among state, structure, and rule changes; and finite-compute approximations to successor relations.

Empirical work can compare DNO with recurrent, state-space, graph, Transformer, and hybrid architectures under matched data, compute budgets, and evaluation protocols. Candidate domains include language and action streams, long-horizon state, numerical signals, and audiovisual representations. These are proposed directions, not verified efficiency or capability claims.

## 10. Boundary between theory, implementation, and philosophy

- **Theory:** Delta describes recursively changing effective structures and conditions.
- **Implementation:** DNO executes a specific finite, trainable set of numerical operations.
- **Evidence:** Tests establish outcomes only under their stated conditions.
- **Philosophy:** Process, incomplete self-description, and non-unique identity are interpretive discussions, not runtime mechanisms or proven mathematical theorems.

## 11. Related documents

- [README](../README.md)
- [Architecture and mathematics](ARCHITECTURE.md)
- [Training and data](TRAINING_AND_DATA.md)
- [Experimental results](EXPERIMENTS.md)
- [Research outlook](RESEARCH_OUTLOOK.md)
- [Code-to-principle traceability](CODE_AND_EVIDENCE_MAP.md)
- [Data provenance](PROVENANCE.md)
