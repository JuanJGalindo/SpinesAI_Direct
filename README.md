# SpinesAI_Direct

**Research precursor** for Hamiltonian Neural Networks (HNNs) and their variants — a literature review plus working, pedagogical notebooks that map the landscape of *structure-preserving* neural dynamics. The abstractions distilled here are productized in the sibling library **[hamiltoniannet](../hamiltoniannet)** (import name `hnet`).

> Status: early-stage research repository. Notebooks are the deliverable; there is no installable package here.

## What's inside

| Path | What it is |
|---|---|
| `SpinesIA_Direct.ipynb` | Main narrative: from MLP energy drift → HNN inductive bias → self-supervised HNN, on the nonlinear pendulum. Includes the literature landscape table. |
| `HNN_Demo.ipynb` | Warm-up: regularized regression → MLP → first HNN on the pendulum. |
| `SpinPrecession_HNNs.ipynb` | Non-canonical (Lie-Poisson) case: canonical HNN vs **PoissonHNN** on spin precession (S² manifold), with Casimir/energy/cosine metrics. |
| `Core/HNN_Core_Benchmarks.ipynb` | **Source-of-truth** consolidated reference implementation (the classes the library ports). |
| `Core/weights_*.pt` | Pretrained weights (`baseline`, `hnn`, `selfsup`) for instant demo mode without retraining. |
| `bib/` | 16-paper literature corpus (2019–2026) + `bib/Analysis/` extraction notes and review protocol. |
| `HNNs_WorkPlan.md` | The 6-month north-star roadmap that defines the library architecture and phases. |

## The models studied

HNN (Greydanus 2019) · SHNN (David & Mehats 2023) · Self-Supervised HNN (Mattheakis 2022) · pHNN (Desai 2021) · sPHNN (Roth 2025) · AdaptableHNN (Han 2021) · HGN (Toth 2020) · PoissonHNN · SNO (Makara 2026) — with a plain MLP baseline as the energy-drift control.

## Running the notebooks

Requires Python 3.10+. The notebooks use **PyTorch, NumPy, SciPy, Matplotlib** and run under **JupyterLab**.

```bash
pip install -r requirements.txt        # flexible bounds
# or, to reproduce the exact tested environment:
pip install -r requirements-lock.txt   # pinned (CPU PyTorch build)
jupyter lab
```

Most notebooks support a **demo mode** that loads `Core/*.pt` instead of training from scratch — check the configuration cell near the top of each notebook.

## Conventions

Physics notation is intentional: `q, p` (canonical phase space), `z = [q, p]`, `H` (Hamiltonian), `S ∈ ℝ³` (spin on S², with Casimir `|S|² = 1`). Derivative labels are reconstructed with **CubicSpline** (O(h⁴)) to avoid the non-symplectic bias of finite differences (David & Mehats 2023). See [CLAUDE.md](CLAUDE.md) for the full convention set.

## Related

- **[hamiltoniannet](../hamiltoniannet)** — the library this research feeds into.
- [`HNNs_WorkPlan.md`](HNNs_WorkPlan.md) — research + engineering roadmap.
- [`docs/index.md`](docs/index.md) — one-page map of this repo.
