# Contributing to SpinesAI_Direct

This is a research repository. Contributions are notebooks, literature analysis, and experiments — not packaged library code (that lives in **[hamiltoniannet](../hamiltoniannet)**).

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
jupyter lab
```

Use `requirements-lock.txt` instead if you need to reproduce the exact tested environment.

## Notebooks

- Keep the **demo / train** switch near the top of each notebook so reviewers can run it without a GPU or long training. Demo mode loads `Core/weights_*.pt`.
- Follow the existing **physics notation** (`q, p, z, H, S`) and the **plot palette** (oracle = black dashed, HNN = crimson, baseline = orange, self-supervised = teal, PoissonHNN = blue). See [CLAUDE.md](CLAUDE.md).
- Reconstruct derivative labels with **CubicSpline**, not finite differences (David & Mehats 2023, Prop. 1).
- Before committing a notebook, **Restart & Run All** to ensure it executes top-to-bottom, then clear bulky outputs if they inflate the diff.
- If you add a new import, update both `requirements.txt` (bound) and `requirements-lock.txt` (pin).

## Literature review (`bib/`)

- One paper per entry; mirror the structured format already used in `bib/Analysis/HNN_Literature_Analysis.md` (problem → method → math → relevance to a library variant).
- Follow the review protocol in `bib/Analysis/HNN_literature_review_plan.md`.
- Name files `Variant_AuthorYear.pdf` (e.g., `pHNNs_DesaiEtAl2021.pdf`); add a `.txt` extract for papers you analyze in depth.

## Scope

- New mathematics, experiments, or variant studies → here.
- Reusable, tested, importable implementations → open a PR in `hamiltoniannet` instead, and reference the notebook this repo it came from.

## Roadmap

All structural work is guided by [`HNNs_WorkPlan.md`](HNNs_WorkPlan.md). Check the current phase before starting large efforts.
