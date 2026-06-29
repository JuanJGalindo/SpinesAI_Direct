# CLAUDE.md — SpinesAI_Direct

Context for AI assistants and new contributors working in this repository.

## What this repo is

A **research precursor**, not a Python package. It contains a literature review and a set of mature Jupyter notebooks that establish the mathematics, experiments, and abstractions for Hamiltonian Neural Networks (HNNs) and variants. The productized library is the sibling repo **[hamiltoniannet](../hamiltoniannet)** (`import hnet`). When code here and there disagree, this repo is the *conceptual* reference and `hamiltoniannet` is the *engineered* implementation.

The north-star roadmap is [`HNNs_WorkPlan.md`](HNNs_WorkPlan.md) (6-month, phase-based). Read it before proposing structural work.

## Layout

- **Notebooks** (pedagogical, not importable library code):
  - `Core/HNN_Core_Benchmarks.ipynb` — **source-of-truth** reference implementation; the classes ported into `hnet` (`HNN`, `BaselineMLP`, `Sin`, integrators, `CubicSpline` data prep).
  - `SpinesIA_Direct.ipynb` — main narrative + literature landscape table.
  - `HNN_Demo.ipynb` — warm-up (regression → MLP → HNN).
  - `SpinPrecession_HNNs.ipynb` — non-canonical PoissonHNN on the S² spin system.
- `Core/weights_baseline.pt`, `weights_hnn.pt`, `weights_selfsup.pt` — pretrained weights for **demo mode** (load instead of retraining).
- `bib/` — 16 papers (2019–2026); `bib/Analysis/` holds the extraction notes and the literature-review protocol.

## Stack

PyTorch · NumPy · SciPy (`solve_ivp`, `CubicSpline`) · Matplotlib, run under JupyterLab. Python 3.10+. No package install — see `requirements.txt` / `requirements-lock.txt`.

## Conventions (preserve these)

- **Physics notation is deliberate:** `q` (position), `p` (momentum), `z = [q, p]`, `H` (Hamiltonian), `t`, and `m, l, g` for pendulum constants. Spin systems use `S ∈ ℝ³` on S² with Casimir `|S|² = 1`.
- **Symplectic gradient:** vector field is `[∂H/∂p, −∂H/∂q]`, computed via `torch.autograd.grad(..., create_graph=True)`.
- **Derivative labels** use **CubicSpline** (O(h⁴)), never raw finite differences — this avoids the non-symplectic bias proven in David & Mehats 2023 (Prop. 1).
- **Metrics:** energy drift `|ΔH|/|H₀|`, relative L2 trajectory error, Casimir error, spin-norm error, cosine-similarity trace.
- **Plot palette:** black dashed = oracle/ground truth, crimson = HNN, orange = baseline MLP, teal = self-supervised, blue = PoissonHNN.
- **Demo vs train:** prefer demo mode (load `Core/*.pt`) for quick checks; full training is in the configuration cell of each notebook.

## Working norms

- Do **not** convert notebooks into a package here — that work belongs in `hamiltoniannet`.
- Keep `bib/Analysis/` extraction format consistent (one structured entry per paper).
- This repo's reproducibility depends on the requirements files; update them if you add an import.

## Related

[hamiltoniannet](../hamiltoniannet) (library) · [`HNNs_WorkPlan.md`](HNNs_WorkPlan.md) · [`docs/index.md`](docs/index.md) · root [`WORKSPACE_OVERVIEW.md`](../WORKSPACE_OVERVIEW.md) (research↔library parallel).
