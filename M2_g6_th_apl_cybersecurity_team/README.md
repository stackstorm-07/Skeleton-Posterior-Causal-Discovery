# CSE 516 (PGM) — Milestone 2: SPOT vs ABIC for Microservice Latency on PetShop

**Group:** Anant Narayan, Ankit Kumar, Milan Shah
**Mode / domain:** Applied AI — latent-aware causal structure learning on the PetShop microservice benchmark.

## What this project does
We learn a causal graph with latent confounders (an ADMG) from PetShop latency data, using two methods on **one shared optimizer**:
- **ABIC** (Bhattacharya et al. 2021): differentiable ADMG learning, the baseline.
- **SPOT** (Ma et al., KDD'24): the same optimizer with a gradient mask driven by a learned *skeleton posterior*.

Learned graphs are scored against the **inverted call graph** (latent-projected to an ADMG, then MAG, then PAG) using skeleton/arrowhead/tail F1, TPR and FDR, plus runtime. A root-cause inference layer (mechanism-shift scoring on the learned ADMG) is implemented and unit-tested; its evaluation on the real issue runs is planned for M3.

## Repository layout
```
M2_<group_name>/
├── README.md                  <- this file
├── Report/
│   └── M2_<group_name>_Report.pdf     M2 technical report (7 pages)
├── Video/
│   └── M2_<group_name>_video          link or file for the presentation
├── Code/
│   ├── README.md              <- how to install and run everything
│   └── src/                   <- the full petshop_spot package (pgm/, skeleton/, scripts/, tests/, models/, ...)
├── Data/
│   └── README.md              <- how to obtain PetShop (not redistributed here)
└── Results/                   <- raw run logs/JSONL and the summary tables used in the report
```

## Where to look
| You want | Go to |
|---|---|
| The write-up, figures and tables | `Report/` |
| To run or reproduce anything | `Code/README.md` |
| Benchmark numbers (Tables IV–VII of the report) | `Results/` (`report_tables.md`, `ps_2d_main_gauss_rank.jsonl`, `syn_2c_*`) |
| The ADMG justification and per-stage design notes | `Code/src/ADMG_JUSTIFICATION.md`, `Code/src/STAGE*_NOTES.md` |

## Status at M2 (summary)
- Done: PetShop pipeline and ground truth; ABIC and SPOT on one verified optimizer (analytic gradients, finite-difference checked); skeleton posterior (held-out AUROC 0.989); paper metrics with a corrected indexing; synthetic benchmark and ablations.
- Partial: PetShop benchmark covers 3 of 4 scenarios with 2–3 optimizer seeds, one restart; PetShop ablations and equal-budget runs are not run.
- Planned (M3): root-cause evaluation on the real issue runs, FCI/RFCI baseline, remaining scenario and seeds, posterior better suited to PetShop-like data.

## Team contributions
- **Anant Narayan** — SPOT implementation and the report.
- **Ankit Kumar** — benchmarks (PetShop and synthetic runs, ablations, result tables).
- **Milan Shah** — inference (root-cause ranking, ADMG justification) and the video presentation.

## Notes
- Code was written with generative-AI assistance; each member's own form submission states their individual use.
- Licences: SPOT code is Apache-2.0 (`third_party/Spot_original`); PetShop data is CC-BY-4.0.
