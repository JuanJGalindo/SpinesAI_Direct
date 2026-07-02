# HamiltonianNet (hnet) — Framework Design Plan

## Context

Three Jupyter notebooks in `C:\Users\juanj\Desktop\WorkSpace\PhysicsSurrogates\SpinesAI_Direct\` contain mature, working HNN implementations across three paradigms (data-driven, Self-Supervised, non-canonical). The goal is to refactor this into a reusable, extensible Python library analogous to DeepXDE, publishable as `pip install hamiltoniannet`.

HNN variants are implemented in a physically motivated pedagogical order: from simple conservative systems up to infinite-dimensional operator learning. Experiments use **Kaggle Local Benchmarks** (`kaggle-benchmarks` SDK) for reproducible, community-comparable evaluation — locally validated, then pushed to Kaggle leaderboards when ready.

---

## Architecture Philosophy: Robustness and Extensibility First

### 1. Protocol-Based Interfaces (not deep class hierarchies)

Every public interface is defined as a `typing.Protocol` in `hnet/_protocols.py`. Classes satisfy protocols by structural subtyping — no mandatory base class. A user can implement a custom system or model in their own codebase without importing any hnet ABC.

```python
# hnet/_protocols.py
class PhysicsSystemProtocol(Protocol):
    config: SystemConfig
    def hamiltonian(self, z: np.ndarray, **params) -> float: ...
    def equations_of_motion(self, t: float, z: np.ndarray, **params) -> np.ndarray: ...

class HamiltonianModelProtocol(Protocol):
    def parameters(self) -> Iterator[nn.Parameter]: ...
    def vector_field(self, z: Tensor, **kwargs) -> Tensor: ...
    def energy(self, z: Tensor) -> Tensor: ...

class IntegratorProtocol(Protocol):
    def integrate(self, model, z0, t_span, n_steps, **kwargs) -> tuple[np.ndarray, np.ndarray]: ...
```

### 2. Strict Dependency Direction (enforced by CI with `import-linter`)

```
_protocols  →  (nothing)
utils       →  _protocols
systems     →  utils, _protocols
data        →  systems, utils
models      →  utils, _protocols
integrators →  models, utils
losses      →  models, utils
training    →  losses, data, models, utils
evaluation  →  systems, integrators, models, utils
visualization → evaluation
benchmarks  →  evaluation, systems        ← Kaggle SDK integration lives here
```

### 3. Frozen Dataclasses for All Configuration

All config objects are `@dataclass(frozen=True)`: no mutation during training, hashable for caching, free `__repr__` for logging.

### 4. Registry Pattern for Extensibility Without Core Changes

```python
@hnet.register_system("pendulum")   # built-in
class NonlinearPendulum(...): ...

@hnet.register_system("my_spine")   # user extension — zero changes to library
class SpinalMechanicsSystem(...): ...

system = hnet.system("my_spine")    # works immediately
```

Same registry for models, integrators, loss terms, and activations.

### 5. Versioned Contracts + Legacy Namespace

When a breaking change is necessary, the old class moves to `hnet.legacy.<version>` with a deprecation warning. Users never silently break.

### 6. Pure Functions for Metrics and Geometry

All evaluation metrics and geometry utilities are pure functions — no side effects, no class state, trivially parallelizable.

### 7. Explicit Validation at System Boundaries

All user inputs (IC shapes, parameter ranges, tensor dtypes) are validated with informative error messages. Internal computations trust themselves.

---

## Kaggle Local Benchmark Integration

Kaggle's 2026 local benchmark platform (`kaggle-benchmarks` SDK) is used throughout this project for reproducible, community-shareable evaluation.

**Workflow per experiment:**
```
write task.py locally  →  python task.py (local validate)  →  kaggle b t push
→  kaggle b t run -m <model>  →  kaggle b t download  →  analyze results
```

**HNN-specific usage pattern:**
```python
import kaggle_benchmarks as kbench
from hnet.evaluation import energy_error, relative_l2_error

@kbench.task(name="hnn-pendulum-energy-conservation")
def energy_conservation_task(llm):
    # Use llm as the oracle for system identification or model selection
    # Or use structured output to query a trained HNN via API
    result = llm.prompt("Predict the Hamiltonian trajectory...", schema=TrajectoryResult)
    kbench.assertions.assert_less(result.max_energy_error, 1e-3)

@kbench.task(name="hnn-benchmark-comparison")
def model_comparison_task(llm) -> float:
    # Scored task: returns relative L2 error as benchmark score
    return result.relative_l2
```

**Primary use cases in this project:**
1. **Local validation** — run `python task.py` to verify metrics before any Kaggle submission
2. **Multi-model comparison** — evaluate Baseline vs HNN vs SHNN on same task without touching Kaggle quota
3. **Community benchmarking** — push validated tasks to Kaggle for community model comparison
4. **Seminar demos** — live local benchmark runs as part of weekly presentations

**Required setup** (documented in `docs/kaggle_setup.md`):
```bash
pip install kaggle-benchmarks kaggle-cli
kaggle benchmarks auth
kaggle benchmarks init
```

Benchmark task files live in `benchmarks/` alongside the library:
```
benchmarks/
├── 01_conservative_energy_conservation.py
├── 02_dissipative_decay_rate.py
├── 03_chaotic_lyapunov.py
├── 04_vision_pixel_reconstruction.py
├── 05_noncanonical_casimir.py
└── 06_operator_generalization.py
```

---

## Package Structure

```
hamiltoniannet/
├── pyproject.toml                   # package: hamiltoniannet, import: hnet
│                                    # extras: [kaggle], [hgn], [sno], [docs]
├── src/
│   └── hnet/
│       ├── __init__.py
│       ├── _protocols.py            # typing.Protocol definitions
│       │
│       ├── systems/
│       │   ├── base.py              # PhysicsSystem ABC, SystemConfig
│       │   ├── pendulum.py          # NonlinearPendulum, SimplePendulum
│       │   ├── double_pendulum.py   # DoublePendulum (chaotic)
│       │   ├── henon_heiles.py      # HenonHeiles (2-DOF chaotic)
│       │   ├── damped_oscillator.py # DampedHarmonicOscillator (dissipative)
│       │   ├── spin.py              # SpinPrecession (Lie-Poisson, S²)
│       │   └── registry.py
│       │
│       ├── models/
│       │   ├── base.py              # HamiltonianNet ABC
│       │   ├── backbones/
│       │   │   ├── mlp.py           # MLP, SinMLP
│       │   │   └── ficnn.py         # FullyInputConvexNN
│       │   │
│       │   │   ── Canonical/Conservative ──
│       │   ├── hnn.py               # HNN (Greydanus 2019)
│       │   ├── shnn.py              # SHNN (David & Mehats 2023)
│       │   ├── selfsup_hnn.py       # SelfSupervisedHNN (Mattheakis 2022)
│       │   │
│       │   │   ── Dissipative ──
│       │   ├── phnn.py              # pHNN (Desai 2021)
│       │   ├── sphnn.py             # sPHNN (Roth 2025)
│       │   │
│       │   │   ── Dissipative/Chaotic ──
│       │   ├── adaptable_hnn.py     # AdaptableHNN (Han 2021)
│       │   │
│       │   │   ── Computer-Vision ──
│       │   ├── hgn.py               # HGN (Toth 2020)
│       │   │
│       │   │   ── Non-Canonical ──
│       │   ├── poisson_hnn.py       # PoissonHNN: B(z)∇H
│       │   │
│       │   │   ── Operator Learning ──
│       │   ├── sno.py               # SNO (Makara 2026)
│       │   │
│       │   └── baseline.py          # BaselineMLP (benchmark foil)
│       │
│       ├── geometry/
│       │   ├── canonical.py         # Darboux maps: S² ↔ (q,p)
│       │   ├── poisson.py           # so3_tensor, se3_tensor
│       │   └── projection.py        # sphere_projection, stiefel_projection
│       │
│       ├── integrators/
│       │   ├── base.py
│       │   ├── symplectic.py        # SymplecticEuler, StormerVerlet, LeapFrog
│       │   ├── runge_kutta.py       # ForwardEuler, RK4, Midpoint
│       │   ├── scipy_wrapper.py     # ScipyIntegrator
│       │   └── projected.py         # ProjectedIntegrator
│       │
│       ├── losses/
│       │   ├── base.py              # LossTerm ABC, WeightedLoss
│       │   ├── derivative.py        # DerivativeMatchingLoss
│       │   ├── symplectic.py        # SymplecticSchemeLoss (SHNN)
│       │   ├── collocation.py       # CollocationLoss (Self-Supervised)
│       │   ├── energy.py            # EnergyRegularizationLoss
│       │   ├── trajectory.py        # TrajectoryMatchingLoss
│       │   ├── vae.py               # VAELoss (HGN)
│       │   └── parsimony.py         # L1ParsimonLoss (pHNN)
│       │
│       ├── data/
│       │   ├── base.py
│       │   ├── derivative_dataset.py # Trajectory + CubicSpline derivatives
│       │   ├── state_pair_dataset.py
│       │   ├── collocation_dataset.py
│       │   └── pixel_dataset.py
│       │
│       ├── training/
│       │   ├── config.py
│       │   ├── trainer.py
│       │   ├── callbacks.py
│       │   └── curriculum.py        # CausalCurriculum
│       │
│       ├── evaluation/
│       │   ├── metrics.py           # Pure functions
│       │   ├── evaluator.py
│       │   └── benchmark.py         # BenchmarkSuite
│       │
│       ├── visualization/
│       │   ├── phase_portrait.py
│       │   ├── energy_drift.py
│       │   └── benchmark_report.py
│       │
│       └── utils/
│           ├── autograd.py
│           ├── device.py
│           └── reproducibility.py
│
├── benchmarks/                      # Kaggle local benchmark task files
│   ├── 01_conservative_energy.py
│   ├── 02_dissipative_decay.py
│   ├── 03_chaotic_lyapunov.py
│   ├── 04_vision_pixel.py
│   ├── 05_noncanonical_casimir.py
│   └── 06_operator_generalization.py
│
├── tests/
│   ├── unit/
│   └── integration/                 # < 60 s on CPU
│
├── examples/
│   ├── 01_canonical_hnn.py          # Greydanus HNN on pendulum
│   ├── 02_shnn_symplectic_loss.py   # SHNN
│   ├── 03_selfsup_hnn.py            # Self-Supervised HNN
│   ├── 04_phnn_dissipative.py       # pHNN on damped oscillator
│   ├── 05_sphnn_stable.py           # sPHNN stability
│   ├── 06_adaptable_chaotic.py      # AdaptableHNN on chaotic systems
│   ├── 07_hgn_vision.py             # HGN from pixel observations
│   ├── 08_poisson_spin.py           # PoissonHNN on spin precession
│   ├── 09_sno_operator.py           # SNO on Hamiltonian PDE
│   ├── 10_full_benchmark.py         # All variants comparison
│   └── notebooks/getting_started.ipynb
│
└── docs/
    ├── literature_review.md
    ├── architecture.md
    ├── kaggle_setup.md
    ├── research_directions.md       # Living document — future ideas
    └── api/
```

---

## Implementation Phases

### Phase 1 — Canonical / Conservative Systems
*Variants: HNN (Greydanus 2019), SHNN (David & Mehats 2023), Self-Supervised HNN (Mattheakis 2022)*
*Physics: nonlinear pendulum, simple harmonic oscillator — integrable, energy-conserving*

**Core infrastructure (shared by all phases):**
- `_protocols.py`, `systems/base.py`, `systems/pendulum.py`, `utils/`
- `models/base.py`, `models/backbones/mlp.py`
- `losses/base.py`, `losses/derivative.py`, `losses/collocation.py`, `losses/symplectic.py`
- `data/derivative_dataset.py` (CubicSpline), `data/state_pair_dataset.py`, `data/collocation_dataset.py`
- `training/trainer.py`, `training/curriculum.py`
- `integrators/scipy_wrapper.py`, `integrators/symplectic.py`, `integrators/runge_kutta.py`
- `evaluation/metrics.py`, `evaluation/evaluator.py`

**Model implementations:**
- `models/hnn.py` — direct port of `HNN_Core_Benchmarks.ipynb` Cell 3
- `models/shnn.py` + `losses/symplectic.py` — David & Mehats scheme-point loss
- `models/selfsup_hnn.py` — trial solution `z(t) = z₀ + (1-e⁻ᵗ)·Net(t)`, from Cell 5
- `models/baseline.py` — benchmark foil

**Kaggle benchmark task:**
```python
# benchmarks/01_conservative_energy.py
@kbench.task(name="hnn-conservative-energy-conservation")
def conservative_energy_task(llm) -> float:
    # Score: normalized max energy error over 100s rollout
    # Locally validated: python benchmarks/01_conservative_energy.py
```

**Examples:** `01_canonical_hnn.py`, `02_shnn_symplectic_loss.py`, `03_selfsup_hnn.py`

---

### Phase 2 — Dissipative Systems
*Variants: pHNN (Desai 2021), sPHNN (Roth 2025)*
*Physics: damped harmonic oscillator, driven pendulum — energy decays to attractor*

- `systems/damped_oscillator.py`
- `models/phnn.py` — H + F(t) + N decomposition; `losses/parsimony.py`
- `models/backbones/ficnn.py` — FullyInputConvexNN (non-negative weights + softplus)
- `models/sphnn.py` — FICNN Hamiltonian + Cholesky dissipation `R = LL^T`; Lyapunov stability certificate
- `losses/trajectory.py` — TrajectoryMatchingLoss for adjoint-mode sPHNN training
- `evaluation/benchmark.py` — BenchmarkSuite (first use: pHNN vs sPHNN vs baseline on damped system)

**Key architectural point:** FICNN in `backbones/ficnn.py` is reusable beyond sPHNN — it is a general backbone for any convexity-constrained energy function (optimal transport, Lyapunov functions, contact Hamiltonians).

**Kaggle benchmark task:**
```python
# benchmarks/02_dissipative_decay.py
@kbench.task(name="hnn-dissipative-energy-decay")
def dissipative_energy_task(llm) -> float:
    # Score: accuracy of learned energy decay rate vs oracle
```

**Examples:** `04_phnn_dissipative.py`, `05_sphnn_stable.py`

---

### Phase 3 — Dissipative / Chaotic Systems
*Variants: AdaptableHNN (Han 2021)*
*Physics: Henon-Heiles (chaotic), double pendulum, parameter-varying systems near chaos threshold*

- `systems/henon_heiles.py`, `systems/double_pendulum.py`
- `models/adaptable_hnn.py` — H(q,p,α), symplectic gradient taken only w.r.t. z
- New metric: `evaluation/metrics.py` → `lyapunov_exponent()`, `poincare_section()`
- Visualization: `visualization/poincare_map.py`

**Kaggle benchmark task:**
```python
# benchmarks/03_chaotic_lyapunov.py
@kbench.task(name="hnn-chaotic-lyapunov-estimation")
def chaotic_lyapunov_task(llm) -> float:
    # Score: relative error on estimated Lyapunov exponent vs oracle
```

**Examples:** `06_adaptable_chaotic.py`

---

### Phase 4 — Computer-Vision Based Systems
*Variants: HGN (Toth 2020)*
*Physics: pendulum/spring observed as image sequences — no coordinate access*

- `models/hgn.py` — VAE encoder/decoder + leapfrog in training graph
- `losses/vae.py` — ELBO + energy conservation term
- `data/pixel_dataset.py` — image sequence dataset generator
- New metric: pixel reconstruction error, latent energy conservation

**Kaggle benchmark task:**
```python
# benchmarks/04_vision_pixel.py
@kbench.task(name="hgn-pixel-trajectory-prediction")
def vision_task(llm) -> float:
    # Score: pixel reconstruction MSE on held-out frames
```

**Examples:** `07_hgn_vision.py`

---

### Phase 5 — Non-Canonical Systems
*Variants: PoissonHNN (Greydanus generalization)*
*Physics: spin precession (S² manifold), rigid body (SO(3)), fluid particles*

- `geometry/canonical.py` (Darboux maps), `geometry/poisson.py` (so3_tensor), `geometry/projection.py`
- `systems/spin.py` — SpinPrecession (Lie-Poisson, S²)
- `models/poisson_hnn.py` — `B(z)∇H` cross-product; direct port of `SpinPrecession_HNNs.ipynb`
- `integrators/projected.py` — ProjectedIntegrator (RK4 + sphere projection, fixes manifold drift)
- New metrics: `casimir_error()`, `cosine_similarity()`, `spin_norm_conservation()`

**Kaggle benchmark task:**
```python
# benchmarks/05_noncanonical_casimir.py
@kbench.task(name="poisson-hnn-casimir-conservation")
def noncanonical_task(llm) -> float:
    # Score: max Casimir invariant drift over 150s rollout
```

**Examples:** `08_poisson_spin.py`

---

### Phase 6 — Operator Learning and Advanced Approaches
*Variants: SNO (Makara 2026)*
*Physics: Hamiltonian PDEs — KdV, nonlinear wave equation, Schrödinger equation*

- `models/sno.py` — SAFNO operator architecture; symplectic shear composition; mesh-independent evaluation
- New metric: `mesh_independence_test()`, `operator_generalization_error()`

**Architectural note:** SNO does not fit the `scalar_field → vector_field` paradigm — it learns the time-τ flow map directly. It satisfies `HamiltonianModelProtocol` via its `vector_field` method but does not subclass `HamiltonianNet`. This is honest, not a hack.

**Kaggle benchmark task:**
```python
# benchmarks/06_operator_generalization.py
@kbench.task(name="sno-operator-generalization")
def operator_task(llm) -> float:
    # Score: relative L2 on unseen initial conditions across mesh resolutions
```

**Examples:** `09_sno_operator.py`, `10_full_benchmark.py`

---

### Phase 7 — Packaging and Documentation
- `pyproject.toml` with extras: `[kaggle]`, `[hgn]`, `[sno]`, `[docs]`
- Sphinx `autodoc` + tutorial pages
- `docs/kaggle_setup.md` — setup guide for Kaggle local benchmarks
- `docs/literature_review.md` — extended from `HNN_Literature_Analysis.md`
- `docs/research_directions.md` — living document for future ideas
- Version 0.1.0 GitHub release + PyPI publication

---

## Variant-to-Abstraction Map

| Variant | Phase | `scalar_field` | `vector_field` override | Loss | Dataset |
|---|---|---|---|---|---|
| HNN (Greydanus) | 1 | MLP | Default symplectic | `DerivativeMatching` | `Derivative` (spline) |
| SHNN (David) | 1 | MLP | Inherited, unused in training | `SymplecticScheme` | `StatePair` |
| SelfSupervisedHNN | 1 | — (ODE solver) | trial_solution → autograd | `Collocation` | `Collocation` |
| pHNN (Desai) | 2 | MLP + F(t) + N | `[J-N]∇H + F(t)` | `Derivative + L1Parsimony` | `Derivative` |
| sPHNN (Roth) | 2 | FICNN | `[J(x)-R(x)]∇H + Gu` | `Derivative` or `Trajectory` | Either |
| AdaptableHNN (Han) | 3 | MLP(z, α) | Symplectic w.r.t. z only | `DerivativeMatching` | `Derivative` + α |
| HGN (Toth) | 4 | MLP in latent | Leapfrog in latent | `VAELoss` | `Pixel` |
| PoissonHNN | 5 | MLP | `B(z)∇H` cross-product | `DerivativeMatching` | `Derivative` |
| SNO (Makara) | 6 | — (flow map) | Direct surrogate | L2 on function pairs | Function pairs |
| BaselineMLP | all | — | Direct MLP output | `DerivativeMatching` | `Derivative` |

---

## 6-Month Weekly Roadmap

**Budget: 15 h/week** split as:
- Literature Review (L): ~3 h
- Framework Implementation (I): ~6 h
- Experiments & Kaggle Benchmarks (E): ~4 h
- Seminar Presentation (S): ~2 h

Each week produces: code merged to main + one Kaggle benchmark task (locally validated) + one 20-30 min seminar presentation (lit review summary + live experiment).

---

### Month 1 — Canonical/Conservative: Core Infrastructure (Weeks 1–4)

**Week 1 — The Energy Problem + Project Bootstrap**
- L: Greydanus 2019 — symplectic gradient, modified Hamiltonian, backward error analysis
- I: Repo scaffold (pyproject.toml, src layout, CI skeleton), `_protocols.py`, `systems/base.py`, `systems/pendulum.py`, `utils/`, `models/baseline.py`
- E: Reproduce baseline MLP energy drift from `HNN_Demo.ipynb`; install `kaggle-benchmarks`; create `benchmarks/` directory and scaffold `01_conservative_energy.py`
- S: "The Energy Problem in Neural Dynamics — Why MLPs Cannot Conserve Energy"

**Week 2 — HNN + CubicSpline Derivative Reconstruction**
- L: David & Mehats 2023 Prop. 1 — forward-Euler derivative labels impose non-symplectic bias
- I: `models/base.py`, `models/backbones/mlp.py` (MLP + SinMLP), `models/hnn.py`, `data/derivative_dataset.py` (CubicSpline)
- E: HNN training on pendulum; CubicSpline vs finite-difference derivative quality comparison; first local `python benchmarks/01_conservative_energy.py` run
- S: "Hamiltonian Neural Networks: Architecture and the Derivative Label Problem"

**Week 3 — Training Loop + Integrators**
- L: Chen et al. 2022 review — taxonomy of HNN variants and training paradigms
- I: `losses/derivative.py`, `training/trainer.py`, `training/config.py`, `integrators/scipy_wrapper.py`, `integrators/runge_kutta.py`, `integrators/symplectic.py`
- E: Full HNN training run; energy conservation over 100 s rollout; integrator order comparison (Euler vs RK4 vs SymplecticEuler); Kaggle task local validation
- S: "From Math to Code: The hnet Architecture and DeepXDE Design Philosophy"

**Week 4 — Evaluation Suite + Example 01 + Kaggle Push**
- L: Mattheakis 2022 motivation — data-free HNN paradigm introduction
- I: `evaluation/metrics.py`, `evaluation/evaluator.py`, `visualization/phase_portrait.py`, `visualization/energy_drift.py`, `examples/01_canonical_hnn.py`, unit tests Phase 1
- E: Parameter sweep (hidden_dim, n_layers); finalize and **push** `benchmarks/01_conservative_energy.py` to Kaggle; download and analyze community results
- S: "Evaluating Physical Fidelity: Metrics Beyond MSE"

---

### Month 2 — Canonical: SHNN + Self-Supervised (Weeks 5–8)

**Week 5 — SHNN: Symplectic Scheme in the Loss**
- L: David & Mehats 2023 full paper — proof that symplectic Euler loss trains an exactly modified-Hamiltonian-conserving model
- I: `losses/symplectic.py` (SymplecticSchemeLoss: symplectic Euler + implicit midpoint), `models/shnn.py`, `data/state_pair_dataset.py`
- E: SHNN vs HNN on pendulum — data requirement (state pairs vs derivative labels), long-time energy drift; update Kaggle task with SHNN results
- S: "Embedding Symplectic Schemes in the Training Loss: SHNN"

**Week 6 — Self-Supervised HNN Architecture**
- L: Mattheakis 2022 full paper — trial solution, sinusoidal activations for periodicity, energy regularization
- I: `models/selfsup_hnn.py` (SelfSupervisedHNN, Sin activation, trial_solution), `data/collocation_dataset.py`
- E: Self-supervised HNN on pendulum; λ sweep for energy regularization (0, 0.01, 0.1, 1.0); trajectory accuracy vs time horizon T
- S: "Physics as Loss: Learning Trajectories Without Training Data"

**Week 7 — Causal Curriculum Learning**
- L: Krishnapriyan et al. 2021 — failure modes of PINNs; causal training as remedy for spurious constant minima
- I: `losses/collocation.py`, `training/curriculum.py` (CausalCurriculum, WARM_FRAC=0.6), `training/callbacks.py`
- E: Curriculum vs no-curriculum convergence; identify spurious minima; Kaggle benchmark task for self-supervised paradigm
- S: "Curriculum Learning for Physics-Informed Neural Networks: Why and How"

**Week 8 — Three-Paradigm Conservative Benchmark**
- L: Review benchmark methodology and reproducibility best practices for ML physics papers
- I: `evaluation/benchmark.py` (BenchmarkSuite: run, print_table, to_dataframe), `visualization/benchmark_report.py`, `examples/03_selfsup_hnn.py`, `examples/02_shnn_symplectic_loss.py`
- E: Three-paradigm benchmark (Baseline, HNN, SHNN, SelfSup) on pendulum near separatrix; reproduce and extend `HNN_Core_Benchmarks.ipynb`; push Kaggle comparison task
- S: "Four Approaches to Conservative Hamiltonian Learning: A Reproducible Benchmark"

---

### Month 3 — Dissipative + Dissipative/Chaotic (Weeks 9–13)

**Week 9 — pHNN: Port-Hamiltonian Decomposition**
- L: Desai 2021 full paper — H + F(t) + N, L1 parsimony for sparsity, damped/forced systems
- I: `systems/damped_oscillator.py`, `models/phnn.py`, `losses/parsimony.py`, `examples/04_phnn_dissipative.py`
- E: pHNN on viscously-damped pendulum; energy decay rate learning; pHNN vs HNN on same system; scaffold `benchmarks/02_dissipative_decay.py`
- S: "Port-Hamiltonian Neural Networks: Learning Dissipation and Forcing"

**Week 10 — Input-Convex Neural Networks**
- L: Amos et al. 2017 ICNN — convexity in energy-based models; connection to Lyapunov stability theory
- I: `models/backbones/ficnn.py` (FICNN: non-negative weights + softplus, equilibrium normalization), unit tests for convexity verification
- E: FICNN vs MLP expressivity comparison; gradient landscape visualization; convexity property test via random point evaluation
- S: "Convexity as a Tool: From Input-Convex NNs to Stable Hamiltonians"

**Week 11 — sPHNN: Guaranteed Stability**
- L: Roth et al. 2025 full paper — FICNN Hamiltonian, Cholesky `R = LL^T`, Lyapunov certificate, comparison with pHNN
- I: `models/sphnn.py`, `losses/trajectory.py` (TrajectoryMatchingLoss), `examples/05_sphnn_stable.py`
- E: sPHNN vs pHNN on damped oscillator — stability certificate, perturbation response; push `benchmarks/02_dissipative_decay.py` to Kaggle
- S: "Guaranteed Stability: sPHNN and the Lyapunov Certificate"

**Week 12 — AdaptableHNN: Chaotic Parameter-Varying Systems**
- L: Han et al. 2021 — parameter conditioning, meta-learning perspective, chaos onset via parameter sweep
- I: `systems/henon_heiles.py`, `models/adaptable_hnn.py` (H(q,p,α), symplectic gradient w.r.t. z only), `examples/06_adaptable_chaotic.py`
- E: AdaptableHNN on Henon-Heiles sweeping energy parameter; interpolation vs extrapolation in parameter space; chaos onset detection
- S: "Adaptable HNNs: One Model Across the Regular-Chaotic Transition"

**Week 13 — Chaotic Metrics + Poincaré Maps**
- L: Chaotic dynamics review — Lyapunov exponents, Poincaré sections, KAM theory
- I: `systems/double_pendulum.py`, `evaluation/metrics.py` → `lyapunov_exponent()`, `poincare_section()`; `visualization/poincare_map.py`; `benchmarks/03_chaotic_lyapunov.py` (locally validated)
- E: Full dissipative/chaotic benchmark (pHNN, sPHNN, AdaptableHNN, HNN baseline) on double pendulum; Poincaré section comparison
- S: "Measuring Chaos: Lyapunov Exponents and Poincaré Maps for HNN Validation"

---

### Month 4 — Computer-Vision + Non-Canonical (Weeks 14–18)

**Week 14 — HGN: Learning from Pixels**
- L: Toth et al. 2020 HGN full paper — VAE + leapfrog in training graph, latent Hamiltonian, pixel-space dynamics
- I: `models/hgn.py` (VAE encoder/decoder + leapfrog in latent), `losses/vae.py` (ELBO + energy), `data/pixel_dataset.py`, scaffold `benchmarks/04_vision_pixel.py`
- E: HGN training on rendered pendulum pixel sequences; latent space energy conservation; pixel reconstruction quality
- S: "Hamiltonian Generative Networks: Learning Physics from Visual Observations"

**Week 15 — HGN Evaluation + Kaggle Vision Benchmark**
- L: VAE theory review; structured latent spaces; disentanglement in physics-informed generative models
- I: Pixel reconstruction metrics, latent phase portrait visualization, `examples/07_hgn_vision.py`; push `benchmarks/04_vision_pixel.py` to Kaggle
- E: HGN vs direct trajectory regression on pixel data; latent space traversal experiment; download Kaggle community results
- S: "What Does the Latent Space of a Hamiltonian Generative Network Look Like?"

**Week 16 — Lie-Poisson Geometry**
- L: Lie-Poisson mechanics — so(3) algebra, Casimir invariants, coadjoint orbits; relevant theory from `SpinPrecession_HNNs.ipynb`
- I: `geometry/canonical.py` (Darboux maps), `geometry/poisson.py` (so3_tensor), `geometry/projection.py` (sphere_projection), `systems/spin.py`
- E: Canonical HNN on spin precession — Casimir error and energy error over 150 s; baseline vs HNN on manifold-constrained system
- S: "Beyond Cartesian Coordinates: Lie-Poisson Systems and Geometric HNNs"

**Week 17 — PoissonHNN + Manifold-Aware Integration**
- L: Comparison of coordinate approaches for non-canonical systems; manifold drift problem in numerical integration
- I: `models/poisson_hnn.py`, `integrators/projected.py` (ProjectedIntegrator), `examples/08_poisson_spin.py`
- E: Canonical HNN vs PoissonHNN on spin; RK45 drift vs ProjectedIntegrator; cosine similarity and spin norm; scaffold `benchmarks/05_noncanonical_casimir.py`
- S: "PoissonHNN: Respecting Manifold Structure in Non-Canonical Systems"

**Week 18 — Non-Canonical Benchmark + Kaggle Push**
- L: Further non-canonical systems — rigid body (SO(3)), fluid vortices, Vlasov equation
- I: New metrics: `casimir_error()`, `cosine_similarity()`, `spin_norm_conservation()`; push `benchmarks/05_noncanonical_casimir.py` to Kaggle
- E: Full non-canonical benchmark (canonical HNN, PoissonHNN, baseline) on spin system; download Kaggle results; cross-phase benchmark comparison
- S: "The Geometry of Learning: Casimir Invariants as HNN Validation Metrics"

---

### Month 5 — Operator Learning + Complete Benchmark (Weeks 19–22)

**Week 19 — SNO: Symplectic Neural Operators**
- L: Makara et al. 2026 full paper — SAFNO architecture, symplectic shear composition, Hamiltonian PDEs, connection to FNO/DeepONet
- I: `models/sno.py` (SNO: SAFNO operator, symplectic shear decomposition), scaffold `benchmarks/06_operator_generalization.py`
- E: SNO on nonlinear wave equation or KdV equation; mesh-independence test; compare with finite-dimensional HNN on same system
- S: "From Finite to Infinite Dimensions: Symplectic Neural Operators"

**Week 20 — SNO Evaluation + Operator Benchmark**
- L: Neural operator theory review (DeepONet, FNO); function-space learning; connection to backward error analysis
- I: `evaluation/metrics.py` → `mesh_independence_test()`, `operator_generalization_error()`; `examples/09_sno_operator.py`; push `benchmarks/06_operator_generalization.py` to Kaggle
- E: SNO generalization across initial conditions and mesh resolutions; download and analyze community Kaggle results
- S: "Neural Operators for Hamiltonian PDEs: Generalization Across Scales"

**Week 21 — Complete Literature Benchmark**
- L: Survey papers published since Week 1 — keep `docs/literature_review.md` current
- I: `examples/10_full_benchmark.py` (all implemented variants on shared pendulum test system); `docs/literature_review.md` finalized (extended from `HNN_Literature_Analysis.md`, citable)
- E: Full benchmark: all variants → Rel-L2, max ΔH/|H₀|, training time, parameter count, data requirement; publication-quality Pareto figure
- S: "The HNN Landscape 2019–2026: A Complete Reproducible Benchmark"

**Week 22 — Intellectual Playfulness: Open Research Directions**
- L: Graph neural networks for multi-body physics; stochastic Hamiltonian systems; Riemannian HNNs; quantum Hamiltonians; contact geometry for friction
- I: `docs/research_directions.md` — formalize open problems; prototype Graph HNN for multi-body pendulum (message-passing Hamiltonian)
- E: Multi-body pendulum as graph; preliminary result; identify most tractable research direction for next 6 months
- S: "Open Problems and Wild Ideas: Where Hamiltonian Learning Can Go"

---

### Month 6 — Packaging, Documentation, Future Vision (Weeks 23–26)

**Week 23 — Performance + Scalability**
- L: PyTorch performance — `torch.compile`, `vmap`, memory-efficient adjoint training
- I: Profiling of training/evaluation loops; batched evaluation with `vmap`; optional `torch.compile` support in Trainer
- E: Scaling experiments: state_dim 2 → 100 → 1000; training time vs system size; memory benchmarks
- S: "Scaling Physics-Informed Learning: Performance Considerations"

**Week 24 — Plugin System + Community Use Case**
- L: SpinesHNN_DirectSurrogate.pdf — biomechanics surrogate as HNN application
- I: Plugin system final validation; `docs/architecture.md` (extension guide); biomechanics custom system example via plugin API — zero changes to core library
- E: Validate full plugin workflow; `kaggle benchmarks` auth + init documentation test on clean environment
- S: "HNNs in Biomechanics: Extending hnet Without Touching Core Code"

**Week 25 — PyPI + Sphinx Documentation**
- L: Scientific Python packaging standards; semantic versioning; documentation-driven development
- I: Final `pyproject.toml` (extras: `[kaggle]`, `[hgn]`, `[sno]`, `[docs]`), ruff/mypy/pre-commit, `CHANGELOG.md`, Sphinx `autodoc`, `getting_started.ipynb`
- E: Install from PyPI in clean environment; run all 10 examples; cross-platform test (Windows + Linux)
- S: "Software Architecture for Reproducible Science: Packaging hnet"

**Week 26 — Retrospective + Next 6-Month Roadmap**
- L: Comprehensive literature update; identify what the community has built since Week 1
- I: Final integration tests; address accumulated technical debt; version 0.1.0 GitHub release tag
- E: Final comprehensive benchmark report from all 6 Kaggle tasks; document improvement from Weeks 1→26
- S: "Six Months of HNNs: What We Built, What We Learned, and What's Next"

---

## Room for Future Improvements and Intellectual Playfulness

These live in `docs/research_directions.md` — a living document updated weekly. They are kept out of production code deliberately.

**Medium-term research extensions:**
- **Graph HNN** — message-passing Hamiltonian for N-body systems and molecular dynamics
- **Riemannian HNN** — H_θ on a Riemannian manifold with geodesic integration
- **Stochastic HNN** — Langevin dynamics `dz = J∇H dt + σ dW`; contact geometry for dissipation
- **Quantum HNN** — Schrödinger equation as Hamiltonian flow on Hilbert space
- **Multi-scale HNN** — hierarchical Hamiltonians for systems with separated time scales
- **Hybrid HNN** — `H = H_known + H_residual` (known analytical part + residual correction NN)

**Long-term architectural extensions:**
- JAX backend (`jax.grad` + `jax.jit`, `equinox` model definition)
- Probabilistic HNN — Bayesian scalar field H_θ with uncertainty on energy conservation
- Active-learning HNN — query oracle only where energy error is largest
- Unify HNN and SNO under a `HamiltonianOperator` ABC

**Open mathematical questions** (potential papers):
- Tightest long-time energy drift bound for a given modified Hamiltonian H_θ?
- Can SHNN training be extended to contact Hamiltonians?
- Universal approximation theorem for PoissonHNNs on compact Lie groups?

---

## Verification Plan

**After Phase 1 (Week 4):** `pytest tests/unit/` passes; HNN energy error < 1e-3 over 100 s; baseline drift > 1e-1 within 50 s; Kaggle task `01_conservative_energy.py` runs locally without errors

**After Phase 2 (Week 11):** sPHNN stability certificate verified numerically; pHNN energy decay rate within 5% of oracle

**After Phase 3 (Week 13):** AdaptableHNN interpolates between regular and chaotic regimes; Lyapunov exponent estimate within 20% of oracle

**After Phase 4 (Week 15):** HGN pixel reconstruction MSE < 0.01 on held-out frames; latent energy conservation bounded

**After Phase 5 (Week 18):** PoissonHNN Casimir error < 0.01 over 150 s; spin norm conservation within 0.01 with ProjectedIntegrator

**After Phase 6 (Week 20):** SNO generalizes across 3 mesh resolutions; `benchmarks/06_operator_generalization.py` pushed to Kaggle

**Ongoing CI:** `import-linter` enforces dependency direction; all metric functions pass property-based tests; integration test suite runs in < 60 s on CPU

---

## Source Files to Migrate From

| Source | Content |
|---|---|
| `HNN_Core_Benchmarks.ipynb` | `HamiltonianBase`, `HNN`, `BaselineMLP`, `TrajectoryNet`, `Sin`, `trial_solution`, integrators, CausalCurriculum, CubicSpline |
| `SpinPrecession_HNNs.ipynb` | `PoissonHNN`, `integrate_poisson_rk4_proj`, Darboux maps, full metric suite |
| `HNN_Demo.ipynb` | Training loop patterns, phase portrait visualization |
| `HNN_Literature_Analysis.md` | Mathematical formulations for all 8 variants → `docs/literature_review.md` |

**Pre-read before implementing advanced variants:**
- `bib/BasicHNNs/SHNNs_DavidMehats2023.pdf` — `SymplecticSchemeLoss` correctness
- `bib/sPHNNs_RothEtAl2026.pdf` — FICNN normalization and Cholesky constraints
- `bib/SNO_MakaraEtAl2026.pdf` — SAFNO operator architecture