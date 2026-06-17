# HNN Literature Review — Cowork Task Plan

> **Purpose.** Condense the HNN-family literature into a concept-centric review that doubles as a *domain analysis*: its extraction schema becomes the **configuration axes** of a Python HNN framework and the **task/metric grid** of a generalized benchmark.
>
> **Dual mandate (hold both at all times).** (a) Produce a coherent scholarly map (Master's Thesis / Q1 paper). (b) Reverse-engineer the common + variable features across the HNN family so the map is an executable spec for a benchmark and library.
>
> **How Cowork should use this file.** Treat the papers in this task as the corpus. For each paper: read full text → fill the **Extraction Template** (§3) using the **controlled vocabularies** → rate code/reproducibility → note gaps → place in the taxonomy → append a row to the master matrix. The **Greydanus reference fill (§5)** is the gold standard for what a complete extraction looks like. Run §6 (synthesis/gaps/mapping) only after the corpus is extracted.

---

## 1. Scope & Questions

**Knowledge RQs**
- RQ1: which physical inductive biases are encoded, and by what mechanism (architecture vs loss)?
- RQ2: which architectures, integrators, and loss formulations are used?
- RQ3: how is evaluation done (systems, metrics, horizons, splits)?
- RQ4: what are the documented failure modes / open problems?

**Design DQs (what makes this fit-for-purpose)**
- DQ1: what is the minimal set of abstractions spanning the variant space?
- DQ2: what task + metric grid constitutes a fair, comprehensive benchmark?
- DQ3: which existing libraries/benchmarks already cover part of this, and where is the gap?

**Boundary.** In: HNN family (HNN, LNN, SymODEN, SympNets, port-/dissipative-Hamiltonian, constrained variants). Adjacent (cite, don't fully extract unless foundational): generic Neural ODEs, PINNs, geometric DL.

## 2. Eight-step pipeline (condensed)

| Step | Core action | Key question / standpoint |
|------|-------------|---------------------------|
| S1 Scope & Questions | Fix RQs + DQs + inclusion boundary | What is in the HNN family vs adjacent? |
| S2 Search & Sourcing | DB search + snowballing; log protocol | Reproducible? PRISMA-ready? |
| S3 Screen & Appraise | 2-pass screen + code/reproducibility axis | Knowledge source, build resource, or both? |
| S4 **Extraction schema (HUB)** | Concept matrix: rows=papers, cols=axes | Every column = a future config flag |
| S5 Synthesis & Taxonomy | Cluster by matrix position; build tree | Does each variant fall in exactly one cell? |
| S6 Gap Analysis → Requirements | Find 3 gap types; derive requirements | Why must a new benchmark + lib exist? |
| S7 Map to Artifacts | Benchmark grid + library modules | Can each model be expressed as a config? |
| S8 Living Review & Validation | Version protocol; reimplement landmarks | Does any model resist the abstraction? |

**Sourcing (S2).** Scopus / WoS / IEEE / ACM **+** arXiv / Semantic Scholar / OpenReview. Boolean seeds: `"Hamiltonian neural network"`, `"Lagrangian neural network"`, `symplectic`, `energy-conserving`, `physics-informed/guided`, `neural ODE`, `learning dynamics`. Forward/backward snowball from anchors (Greydanus 2019; Cranmer 2020; Chen 2018; Zhong 2020 SymODEN; Jin 2020 SympNets; Finzi 2020 constrained).

**Citation-graph accelerators:** Connected Papers, Research Rabbit, Litmaps, Semantic Scholar.

---

## 3. Extraction Template (copy per paper)

```yaml
paper_id:            # AuthorYear, e.g. Greydanus2019
title:
venue_year:          # e.g. NeurIPS 2019
arxiv_id:

# AXIS 1 — Physics / inductive-bias structure
dynamics_class:      # conservative | dissipative | controlled/port-Hamiltonian | hybrid
separability:        # separable | non-separable | learned-full(agnostic)
constraints:         # unconstrained | holonomic | nonholonomic
bias_encoded_in:     # architecture | loss | both

# AXIS 2 — State representation
coordinate_type:     # canonical(q,p) | generalized(q,qdot) | learned-latent | pixels/observations
representation_via:  # given | autoencoder | other-embedding

# AXIS 3 — Backbone
backbone:            # MLP | CNN | GNN | RNN | Transformer | other
network_output:      # scalar-energy | vector-field | two-scalar(Helmholtz) | other
hyperparams:         # width/depth/activation/optimizer — VERIFY FROM CODE

# AXIS 4 — Differentiation & integration
derivative_source:   # autodiff-of-energy | finite-difference | analytic
symplectic_form_applied: # yes(J=[[0,I],[-I,0]]) | no
integrator:          # symplectic/leapfrog | generic-RK(RK45/dopri5) | none
integrator_in_training_loop: # yes | no
test_time_solver:    # solver + tolerance

# AXIS 5 — Supervision
target:              # time-derivatives(vector-field matching) | trajectory/rollout | energy | mixed
loss_form:           # short formula
regularizers:        # energy | constraint | canonical-coord-aux | none

# AXIS 6 — Evaluation
systems:             # [ ... ]
metrics:             # [state-RMSE, energy-drift, VPT, rollout-error-vs-horizon, symplecticity/constraint-violation, reversibility, sample-efficiency, wall-clock]
horizons_splits:

# AXIS 7 — Reproducibility / build-resource value
code_released:       # yes/no + URL
framework:           # PyTorch | JAX | TF | other
reproducible_as_is:  # high | med | low
port_priority:       # high | med | low

# AXIS 8 — Provenance & taxonomy
stated_advantage:
known_limitations:
taxonomy_position:   # which axes flip vs the HNN root cell
```

**Fixed reading lens** (apply uniformly, fast):
1. What physical structure is assumed — baked into architecture or only into the loss?
2. Canonical, generalized, latent, or pixel coordinates?
3. Derivatives by autodiff or finite difference; symplectic vs generic integrator; integrator in the training loop?
4. Derivative-matching or rollout-matching supervision?
5. Exactly which systems, metrics, horizons — and is code released?
6. Claimed advantage vs admitted failure?

---

## 4. Per-paper task workflow (for Cowork)

1. **Read** the full PDF (use the pdf-reading approach; don't rely on abstract alone).
2. **Fill** the §3 template. Use only the controlled-vocabulary values so columns stay machine-mappable.
3. **Verify, don't hallucinate.** Architecture hyperparameters and exact metrics must come from the paper or its code — flag any cell you cannot confirm as `UNVERIFIED`.
4. **Rate build value** (Axis 7): open code in a maintained framework = high port priority.
5. **Note gaps** the paper exposes (evaluation inconsistency / fragmentation / untested combination).
6. **Place in taxonomy** (Axis 8): state which switches flip relative to the HNN root.
7. **Append** the filled block to the master matrix file and add one row to the summary table.

---

## 5. Gold-standard reference fill — Greydanus, Dzamba & Yosinski (2019)

*Source: arXiv:1906.01563 (NeurIPS 2019); code `github.com/greydanus/hamiltonian-nn`; blog `greydanus.github.io/2019/05/15/hamiltonian-nns`. This is the taxonomy ROOT — every variant is a switch-flip from this cell.*

```yaml
paper_id:            Greydanus2019
title:               Hamiltonian Neural Networks
venue_year:          NeurIPS 2019
arxiv_id:            1906.01563

dynamics_class:      conservative          # no dissipation
separability:        learned-full          # learns full H(q,p), handles non-separable
constraints:         unconstrained
bias_encoded_in:     architecture          # symplectic gradient is structural; loss is plain L2

coordinate_type:     canonical(q,p)        # + latent variant
representation_via:  given                 # + autoencoder for the pixel-pendulum variant
backbone:            MLP
network_output:      scalar-energy         # outputs scalar H
hyperparams:         UNVERIFIED            # width/activation/optimizer — confirm in nn_models.py / paper

derivative_source:   autodiff-of-energy
symplectic_form_applied: yes               # S_H = (dH/dp, -dH/dq)
integrator:          generic-RK            # RK45 via scipy.solve_ivp
integrator_in_training_loop: no            # trains on the vector field, not rollouts
test_time_solver:    scipy.integrate.solve_ivp, tol 1e-9

target:              time-derivatives      # vector-field matching
loss_form:           || dq/dt - dH/dp ||^2 + || dp/dt + dH/dq ||^2
regularizers:        canonical-coord-aux   # only in the pixel/autoencoder variant

systems:             [mass-spring(linear), single-pendulum, two-body, three-body(chaotic), pixel-pendulum(Gym Pendulum-v0)]
metrics:             [derivative/trajectory-MSE, energy-drift-vs-energy-like-quantity, orbit-stability, reversibility]
horizons_splits:     long rollouts integrated at tol 1e-9; per-task train/test

code_released:       yes — github.com/greydanus/hamiltonian-nn
framework:           PyTorch
reproducible_as_is:  high
port_priority:       high                  # first reference implementation to port

stated_advantage:    learns an (near-)conserved energy-like quantity; stable rollouts; reversible (zero-divergence field)
known_limitations:   "conserved quantity ~ energy up to a constant, not exact"; poor on 3-body (large dynamic range); needs ground-truth time-derivatives (finite-diff noise); success judged largely by conservation
taxonomy_position:   ROOT = {conservative · canonical · MLP · autodiff-symplectic · derivative-matching · no-integrator-in-loop}
```

**Baseline (control) in the same paper:** plain MLP regressing the time derivatives directly → config `{direct-MLP, no-symplectic-form, RK45, derivative-loss}`. Its inward spiral on conservative systems is pure model error — the contrast that motivates the HNN.

**How variants flip switches from this root:** LNN → `coordinate_type:generalized` + Lagrangian energy; SymODEN → `integrator_in_training_loop:yes` + `dynamics_class:controlled`; Dissipative/port-HNN → `dynamics_class:dissipative` (Helmholtz two-scalar output); SympNet → `backbone:symplectic-by-construction`.

---

## 6. After extraction — synthesis → artifacts

**S5 Taxonomy.** Primary axes: physics-structure (conservative→dissipative→controlled; separable→non-separable; unconstrained→constrained) × representation (canonical→latent→pixels) × supervision (derivative→rollout; integrator-in-loop or not). Build RQ-keyed comparison tables; any variant that doesn't fit a cell means an axis is missing.

**S6 Gaps → requirements.**
- *Evaluation inconsistency* (different systems/metrics/horizons; non-standard energy metrics) → **the core argument for the benchmark.**
- *Implementation fragmentation* (one incompatible repo per paper) → **the argument for the library.**
- *Untested design-space combinations* → research opportunities.
- Survey existing infra explicitly to differentiate: `torchdiffeq`, `torchdyn`, `dysts`, DeepXDE, NeuroMANCER, official HNN/LNN/SympNet repos.

**S7 Artifact mapping.**
- **Benchmark = graded task suite** (pendulum → spring → double-pendulum/chaotic → two-body → three-body/N-body → dissipative → controlled → pixel) **+ metric battery** (state RMSE, energy drift, valid-prediction-time, rollout-error-vs-horizon, symplecticity/constraint violation, sample efficiency, wall-clock).
- **Library modules mirror the taxonomy axes:** `system/data` · `coordinate/representation` · `energy-network (H/L)` · `differentiation (symplectic-gradient operator)` · `integrator` · `loss/training` · `evaluation`. Each surveyed model must express as a *configuration* of these modules — that is the proof the abstraction is correct.

**S8 Validation.** First milestone: reimplement HNN + baseline in the framework, reproduce the spring and two-body energy-conservation contrast. Then LNN, SymODEN, a dissipative variant, a SympNet. If one resists clean instantiation → framework limitation or genuine research gap (both publishable).

---

## 7. Quality bars / weighted views (Q1-level)

1. **The real crux is the supervision-and-integrator axis, not the energy function.** Most reproducibility issues and design decisions live in whether the integrator is differentiated through during training — the benchmark should isolate exactly that.
2. **Evaluation fragmentation justifies the whole project** — lead the motivation with it.
3. **Energy conservation is necessary but not sufficient** as a metric. Engage the critical line (e.g. Gruver et al.: much of the benefit comes from the second-order ODE bias, not energy conservation per se). A benchmark that rewards only energy drift will mislead.
4. **Survey libraries as rigorously as models**, or a reviewer asks why `dysts`/NeuroMANCER doesn't already suffice.

**Process heuristics that compound:** stay concept-centric (never author-centric); pilot the schema on ~10 papers before scaling; make every schema column a literal config flag so the two artifacts co-evolve; prioritize code-bearing papers; LLM-assisted pre-fill is fine **but verify every cell** — hallucinated metrics silently corrupt the benchmark design.

> **Highest-leverage move:** design this extraction schema and the library's module interfaces in the same sitting. When the taxonomy axes and the code abstractions are the same object, the review stops being a write-up done before the engineering and becomes the specification that produces it.
