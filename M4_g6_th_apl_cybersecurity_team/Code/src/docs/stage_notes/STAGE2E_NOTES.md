# Stage 2e notes: report tables

`python scripts/make_report.py` reads the raw stage-2d JSONL (`results/ps_2d_main_*.jsonl`) and writes
`results/report_tables.md` (paste-ready), `results/report_tables.tex` (booktabs, wrapped in `\resizebox`, needs `graphicx`)
and one CSV per table. Nothing is typed in by hand: re-run it after the queue finishes and every table, the coverage block and
the "key readings" update.

| table | content |
|---|---|
| 1 | skeleton / arrowhead / tail F1, corrected and legacy; scenario-level means, difference to ABIC, wins/losses/ties over scenarios |
| 2 | per-scenario F1 (mean ± sd over optimizer seeds), skeleton TPR/FDR |
| 3 | runtime: offline posterior training (one-time), online inference, optimization (per run, run-weighted and scenario-mean), L-BFGS evaluations, free pairs |
| 4 | estimated vs true directed / bidirected edge counts |
| 5 | skeleton-posterior AUROC/AUPRC on PetShop vs held-out simulated graphs, share of true edges pinned by the 6e-3 floor |

## Status of the numbers (PARTIAL)
Completed: low_traffic and high_traffic seeds 0-2, temporal_traffic1 seeds 0-1; Gaussian-rank, one restart. Not run: temporal_traffic2,
seeds 3-4, PetShop ablations, equal-budget runs, raw_centered. The generated file says so at the top.

## Things worth knowing before pasting
* Optimization time is reported two ways: run-weighted over all 8 runs (ABIC 115 s / SPOT-cal 46 s / SPOT-1e-3 58 s) and as the mean of
  scenario means (107 / 45 / 57 s). They differ because high_traffic (slow) has 3 runs and temporal_traffic1 (fast) has 2.
* The speed-up tracks the number of free pairs after the floor (105 vs 198 vs all), which points at pruning rather than the stochastic mask,
  but no PetShop ablation separates the two, so the report text says "tracks", not "is caused by".
* The skeleton-F1 gain (+0.02 to +0.05) is positive in 3/3 scenarios but of the same order as the optimizer-seed spread; there is one dataset
  per scenario, so no significance statement is made. Arrowhead/tail differences are inside the spread.
* Legacy tail/arrowhead F1 are much higher than corrected ones (e.g. ABIC tail 0.27 vs 0.06): the legacy convention only compares which node
  carries a head/tail. Always label which one is quoted.
