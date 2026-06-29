# SpinesAI_Direct — Documentation Index

A one-page map of the repository. See the [README](../README.md) for the overview and [CLAUDE.md](../CLAUDE.md) for conventions.

## Notebooks (the deliverable)

| Notebook | Role | Key content |
|---|---|---|
| [`SpinesIA_Direct.ipynb`](../SpinesIA_Direct.ipynb) | Main narrative | Energy-drift problem → HNN inductive bias → self-supervised HNN; literature landscape table |
| [`HNN_Demo.ipynb`](../HNN_Demo.ipynb) | Warm-up | Regularized regression → MLP → first HNN on the pendulum |
| [`SpinPrecession_HNNs.ipynb`](../SpinPrecession_HNNs.ipynb) | Non-canonical case | Canonical HNN vs PoissonHNN on the S² spin system; Casimir/energy/cosine metrics |
| [`Core/HNN_Core_Benchmarks.ipynb`](../Core/HNN_Core_Benchmarks.ipynb) | Source of truth | Consolidated reference classes that `hnet` ports |

Pretrained weights for demo mode: `Core/weights_baseline.pt`, `weights_hnn.pt`, `weights_selfsup.pt`.

## Literature corpus (`bib/`)

- `bib/Analysis/HNN_Literature_Analysis.md` — structured extraction of the core papers.
- `bib/Analysis/HNN_literature_review_plan.md` — the review protocol.
- `bib/BasicHNNs/` — foundational variants (Greydanus, Desai, Mattheakis, David & Mehats, Han).
- `bib/*.pdf` — extended landscape (HGN, chaotic, dissipative, SNO operator learning, 2025–2026 preprints).

### Variant → reference map

| Variant | Paper | Regime |
|---|---|---|
| HNN | Greydanus 2019 | canonical / conservative |
| SHNN | David & Mehats 2023 | symplectic-loss training |
| Self-Supervised HNN | Mattheakis 2022 | data-free / collocation |
| pHNN | Desai 2021 | dissipative (port-Hamiltonian) |
| sPHNN | Roth 2025 | dissipative + Lyapunov-stable |
| AdaptableHNN | Han 2021 | chaotic / parameter-varying |
| HGN | Toth 2020 | computer vision (pixels) |
| PoissonHNN | (generalization) | non-canonical / Lie-Poisson |
| SNO | Makara 2026 | operator learning (PDEs) |

## Planning & related

- [`HNNs_WorkPlan.md`](../HNNs_WorkPlan.md) — the 6-month north-star roadmap.
- [hamiltoniannet](../../hamiltoniannet) — the library that productizes this research.
- [`WORKSPACE_OVERVIEW.md`](../../WORKSPACE_OVERVIEW.md) — research ↔ library parallel and migration notes.
