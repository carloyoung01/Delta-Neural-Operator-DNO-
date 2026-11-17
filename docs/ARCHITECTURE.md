# Mathematical architecture and computational mechanisms

## 1. Foundation

DNO represents an effective current structure and updates it through learnable shared generation. Carriers can evolve locally; executed interactions can alter the carriers involved and produce new carriers, which can subsequently participate in the same computational process.

A general Delta state can be discussed as

$$
\Omega_t=(\mathcal U_t,\mathcal R_t,\mathcal C_t),
$$

where the three terms denote current units, relations, and effective conditions. The executable DNO state is more specific:

$$
\Omega_t=\left(\{(i,x_i,b_i,\ell_i,\mu_i,\tau_i)\}_{i\in I_t},L_t,o_t\right).
$$

Here, `i` is identity, `x` effective content, `b` boundary features, `ℓ` mutable effective local-law features, `μ` nonnegative support, `τ` local elapsed time, `L` present incidence, and `o` current observation state. These are the fields of `State` in `dno.py`, with `next_id` supplying implementation-level identity allocation.

## 2. Shared generative law

$$
(\delta a,\delta b,c)=G_\theta(a,b;o,e,\Delta t).
$$

`Gθ` summarizes the shared `generator` called by `_call()`. A single-carrier evolution uses its single-ended invocation; a two-carrier `compose()` invokes the same generator with both carriers. The outputs are subsequently interpreted and conditioned by the relevant operation. The network parameters `θ` are trained; a carrier's `law` field is an evolving state quantity, not a modification of neural network weights at inference time.

## 3. Recurrence and conditional successors

$$
\Omega_{t+\Delta t}=\mathrm{Canon}\left(\mathrm{Apply}_{G_\theta}(\Omega_t,e_t,a_t,\Delta t)\right).
$$

This is shorthand for the concrete operation that is **actually invoked**, not a new controller or mandatory pipeline. `Canon` represents current-state housekeeping, chiefly exact-zero-support release in `compact()`.

The broader mathematical view of conditional successors is

$$
\Omega_{t+1}\in\mathcal D_{\mathcal B}(\Omega_t,e_t).
$$

The map is set-valued in the mathematical description, while execution follows concrete input and action choices. Given an allowed domain of conditions, a corresponding successor image is

$$
\mathcal P_t=\mathrm{Im}\left[(e,a)\mapsto\mathrm{Canon}(\mathrm{Apply}_{G_\theta}(\Omega_t,e,a))\right].
$$

This expression does not add a separate successor enumerator or data structure. Only actual requested operations run. As the effective state changes, the image of possible successors may also change.

## 4. Operations

| Code | Function |
|---|---|
| `introduce()` | Introduce new numerical carriers into the current state |
| `evolve()` | Continue local carrier evolution and observation-state evolution without compulsory global observation |
| `propose()` | Compute conditional source/target addressing and activity, without executing an interaction |
| `compose()` | Rewrite two interacting carriers and optionally introduce a new carrier |
| `compact()` | Remove exact-zero-support carriers and links to absent identities |
| `attend()` | Actively couple current structure into updated observational state |
| `observe()` | Read out from current state, without writing a new state |

A generated carrier has the same structural fields as other carriers and may participate again. Present `links` affect active reading and planning, but are not automatically transmitted in passive local evolution. The observer can change through `observer_flow` without a required `attend()` call; an active encounter can also update it. Observation and non-observation both have a place in the dynamics.

The implementation contains separate observation and addressing neural paths. They are not secretly collapsed into the single shared carrier generator. The shared generator describes autonomous carrier evolution and executed two-carrier composition.

## 5. State consequences and unexpanded possibilities

Past interactions can alter present content, boundary, law, and relations. The resulting state rather than a full past-event log supplies subsequent conditions. An opportunity does not force an action, and possible successors do not need to be listed. An executed interaction produces a concrete successor, which then supplies the next current state.
