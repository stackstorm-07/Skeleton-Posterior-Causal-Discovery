# Stage 3 notes: inference on the learned graph, root-cause ranking, ADMG justification

**Status: pipeline built and tested on FAKE PetShop-format data only. No real PetShop root-cause result exists yet** (the dataset is not in the
container; the issue folders were never used here). Everything below that says "result" must come from running `scripts/run_rca.py` where the data is.

## What exists
| file | purpose |
|---|---|
| `pgm/rca.py` | normal-period transform, RICF model of a graph, mechanism-shift scores (`anomaly`, `dag`, `admg`), ancestor filter, ranking, graph (de)serialisation |
| `pgm/petshop_issues.py` | reads `<scenario>/{train,test}/issueK/{metrics.csv,target.json}`, abnormal-window extraction, node-name matching |
| `scripts/run_rca.py` | learns (or loads cached) ABIC/SPOT graphs + truth MAG, scores every latency issue, one JSONL row per (case, graph source, method) |
| `scripts/summarize_rca.py` | top-1 / top-3 recall, MRR, per scenario and pooled, `--split train|test|all` |
| `ADMG_JUSTIFICATION.md` | the written argument for an ADMG, with what our results do and do not support |
| `tests/test_rca.py`, `tests/fake_petshop.py::make_fake_issues` | 5 fast tests: GLS identity of the score, recovery of a shifted mechanism on a true ADMG, helpers/transforms, loader + matching, end-to-end runner |

## Method (details in the `pgm/rca.py` docstring)
Normal data = `noissue/metrics.csv`; abnormal data = rows of the issue file at or after the injection timestamp. Both are mapped by a transform fitted on the
normal data only (`log` default; `zscore`, `gauss_rank` available). A linear-Gaussian model of the graph is fitted on the normal data with RICF. With `dmu` the
abnormal mean shift and `d_eps = (I-W1)^T dmu`, node i scores `(Omega^-1 d_eps)_i^2 / (Omega^-1)_ii`.
* `anomaly`: no graph, score = `dmu_i^2` (associational baseline).
* `dag`: directed edges only (Omega diagonal). `admg`: directed + bidirected.
* optional `ancestors only`: candidates restricted to directed ancestors of the target in the graph (all nodes if there are none).
* the target node is never a candidate (the benchmark's root cause is always another node).
Graph sources: `abic`, `spot_cal` (and `spot` = released floor), learned on Gaussian-rank data with the stage-2d settings and seed; `truth` = ground-truth MAG
(reference upper bound, not available in practice). Labels are never used to learn graphs or to set parameters.

## How to run (where the data is)
```
export PETSHOP_ROOT=<petshop-root-cause-analysis>/dataset
python scripts/run_rca.py --scenarios low_traffic high_traffic temporal_traffic1 temporal_traffic2 --sources abic spot_cal truth --out results/rca_stage3.jsonl
python scripts/summarize_rca.py results/rca_stage3.jsonl --out results/rca_summary            # add --split test for the held-out split
python scripts/run_rca.py ... --transform zscore --out results/rca_stage3_zscore.jsonl        # sensitivity
```
First run learns 2 graphs per scenario (from stage 2d: ABIC about 50-190 s and SPOT about 20-80 s per scenario on one core) and caches them in
`results/rca_graphs/`; scoring itself takes seconds. Graph learning with `--graph-seed 0` is meant to reproduce the stage-2d seed-0 graphs (deterministic given the seed), but
the 2d JSONL stores only edge counts, so this has not been verified edge by edge.

## Assumptions to check on the real data (the runner prints what it finds)
1. **Layout and fields** follow the PetShop paper (App. B, Sec. 4.3): `graph.csv`, `noissue/metrics.csv`, `train|test/issueK/{metrics.csv,target.json}`, target and root-cause given as node names, injection time as unix seconds. I verified the layout against the paper, not against the files.
2. **Node names** in `target.json` may differ from the `graph.csv` names (the paper's example shows short names such as `PetSearch`). The loader tries exact, case-insensitive, then a *unique* prefix/substring match and prints `fuzzy`/`ambiguous`/`none`. Check `root_match` counts in the summary; an unmatched root cause counts as a miss.
3. **Abnormal window** = rows at or after the injection time (falls back to the whole file if fewer than 3 rows, with a note in the row). The SLO violation can lag the injection, so the window may include a few normal rows.
4. **Case counts.** The paper lists 14 / 14 / 8 latency issues for low / high / temporal traffic; this dataset has two temporal folders, so expect counts to differ for those. The summary prints the number scored per scenario and split.
5. Only **latency** issues are scored (the graphs were learned on latency). Availability issues are out of scope.
6. In the temporal scenarios the load varies over the day, so "normal vs abnormal" can differ by load alone.

## Context: published numbers (graph GIVEN, different protocol details; not a like-for-like comparison)
PetShop paper, latency issues, top-3 recall (Table 3) / top-1 recall (Table 8): 
| scenario | traversal | CIRCA | counterfactual | RCD | correlation |
|---|---|---|---|---|---|
| low | 0.57 / 0.57 | 0.86 / 0.36 | 0.71 / 0.36 | 0.21 / 0.07 | 0.57 / 0.43 |
| high | 0.79 / 0.57 | 1.00 / 0.50 | 0.86 / 0.57 | 0.07 / 0.00 | 0.79 / 0.64 |
| temporal | 1.00 / 1.00 | 1.00 / 0.75 | 0.50 / 0.38 | 0.75 / 0.25 | 0.75 / 0.62 |
(each cell: top-3 / top-1). Those methods use the true service map or none; ours uses graphs learned from the normal data, so the interesting comparison here is
learned-graph vs truth-graph vs no-graph within our own table. The paper notes a simple correlation baseline is hard to beat with little data.

## Tests
`python tests/run_tests.py -k test_rca` (5 pass here; the end-to-end test needs the Tetrad jar via `CAUSAL_CMD_JAR` or `third_party/Spot_original/learn/bin/`, else it skips).
The fake data is deliberately easy (every method scores 1.0), so those tests check logic and plumbing, not accuracy.

## Bugs found while testing (fixed)
* Timestamp conversion used the integer view of a pandas DatetimeIndex, whose unit depends on the pandas version; it silently selected zero abnormal rows and fell back to the whole file. Now resolution-independent (`total_seconds`), and the fallback is recorded in every row (`window`).
* `df.transform` in the summarizer collided with the DataFrame method.
