# Stage 2a notes: model, RICF, shared optimizer (ABIC), gradient checks

## What exists now
| file | content |
|---|---|
| `pgm/admg_model.py` | parameters (W1 directed, W2 bidirected, Wii noise var), vector layout, bounds, pseudo-variables, Gaussian log-likelihood, BIC, thresholding to an ADMG |
| `pgm/ricf.py` | RICF fit of a *fixed* ADMG (used for BIC; verified equal to direct ML) |
| `pgm/dcd.py` | the one shared optimizer: analytic losses, augmented Lagrangian, L-BFGS-B, BIC-selected restarts, wall-clock budget, **gradient-mask hook** and **hard-skeleton hook** (the only SPOT hooks) |
| `pgm/abic.py` | `run_abic` = `dcd` with no mask |
| `skeleton/simulator.py` | random ancestral ADMGs + linear-Gaussian data (paper recipe). Labelling with the true PAG comes in 2b |
| `scripts/gradcheck.py`, `scripts/time_abic.py` | finite-difference check; ABIC runtime vs size |
| `tests/` | `python tests/run_tests.py [--fast]` (plain Python, no pytest needed; pytest also works) |

## Problems found in the WIP code, and what was done
1. **Penalty differed from the released code.** The WIP used the matrix-exponential form (paper Eq. 4) only. The released ABIC/SPOT code uses a NOTEARS-poly cycle term plus a truncated-series ancestrality term. Both are now implemented with exact analytic gradients; `penalty="released"` is the default (what the paper's ABIC actually ran), `"expm"` is available. Both pass the finite-difference check (rel. error about 1e-9) and give similar results.
2. **Smooth-L0 scale differed by 2x.** Released code: `(e^x-1)/(e^x+1) = tanh(x/2)`, `x = log(n)|theta|`. WIP used `tanh(x)`. Now `l0_scale=0.5` reproduces the released code.
3. **Slow data-fit evaluation.** The O(n d^2) einsum dominated (70% of each evaluation at d=30). Replaced by an exactly equivalent Gram-matrix form (`QuadLoss`, O(d^3) per evaluation): 2.1 ms -> 0.63 ms per value+gradient at d=30, n=1000. Equality with the direct form is part of the gradient test (gap 4e-15).
4. **No timing breakdown.** `DCDResult` now has `t_optimize` (what `time_budget` limits) and `t_select` (thresholding + BIC) for the runtime table.
5. `gradcheck.py` rewritten to cover both penalties, the Gram path, and pinned coordinates; stage-1 tests now skip (not fail) when the PetShop dataset is absent; the Tetrad jar is not in the WIP zip (it is in your `Spot.zip`; put it in `third_party/Spot_original/learn/bin/`).
RICF, the likelihood, BIC, pseudo-variables, the bounds/vector layout and the budget logic were checked and are correct.

## Verified
* Gradients: analytic vs central finite differences, max relative error ~1e-9 (both penalties, full primal loss, pinned coordinates exactly zero).
* RICF: converges, likelihood monotone, equals a direct BFGS maximum-likelihood fit to <1e-3 nats, recovers the generating parameters at n=20000, and the pseudo-variable regression identity holds.
* `h = 0` exactly for ancestral supports and positive for cycles / bidirected edges between an ancestor pair.
* Budget, determinism by seed, mask hook (identity mask == no mask), hard skeleton constraint.

## Measured behaviour of ABIC (important for 2c/2d)
Synthetic ER ancestral ADMGs, d=10, n=1000, 4 graphs, 1 restart, scored like the paper (PAG skeleton / arrowhead / tail F1):

| data | skeleton | arrowhead | tail | #bidirected edges found (truth ~1-2) |
|---|---|---|---|---|
| raw, as in the paper's simulator | 0.92 | 0.80 | 0.43 | 1-5 |
| z-scored | 0.75 | 0.52 | 0.23 | 8-19 |
| z-scored, **oracle skeleton** as hard constraint | 0.95 | 0.67 | 0.39 | 5-7 |
| raw, oracle skeleton | 0.99 | 0.94 | 0.67 | 1-3 |

* Raw-data numbers are in the range the paper reports for ABIC. On z-scored data ABIC degrades, mostly by replacing directed edges with bidirected ones. Cause: the pseudo-variable least-squares *value* counts a bidirected edge in both endpoints' equations (2-variable example: 0.36 vs 0.68 for the same correlation), so descent prefers bidirected explanations unless the ancestral penalty forbids them. The truth has a BIC 100-500 lower than what ABIC returns on z-scored data, so this is a bias of the objective plus local optima, not a gradient bug. It is not something I changed, because it is the published method.
* Raw data in the paper's simulator is also *var-sortable* (downstream variances are larger), which a scale-dependent loss can exploit; PetShop will not have that, so expect PetShop absolute F1 nearer the z-scored row.
* Restarts matter a lot (d=5, n=1000, raw: tail F1 0.67 with 3 restarts, BIC gap to truth 0-27).
* An oracle skeleton raises every F1 and cuts optimizer work 3-6x. That is the upside SPOT's posterior is trying to capture (2c).

## Decisions for you
1. **Preprocessing for PetShop.** Plan default is Gaussian-rank + z-score. Given the above, ABIC (and SPOT) are scale-sensitive; I will still default to z-score (real latency data cannot be treated as var-sortable and heavy tails break the Gaussian fit), and run raw/centered as a sensitivity check in 2d. Tell me if you want a different default.
2. **Restarts vs time.** I will set restarts and budgets from the timing table (`results/abic_timing.json`) in 2d, equal for ABIC and SPOT.

## ABIC runtime (2 cores, n=500, 1 restart, converged to h<=1e-8; `results/abic_timing.json`)
| d | data | optimize time per restart (s) | L-BFGS evals |
|---|---|---|---|
| 10 | raw | 4, 5 | 20k |
| 10 | z-scored | 6, 17 | 24-63k |
| 20 | raw | 56, 25 | 35-90k |
| 20 | z-scored | 71, 67 | ~115k |
| 30 | raw | 79 | 68k |
| 30 | z-scored | 163, 171 | ~140k |
(d=35 was not finished; expect roughly 250 s per z-scored restart.) Cost per evaluation is ~0.6 ms at d=30, so the time is dominated by the *number* of evaluations (about 100-140 L-BFGS solves per restart), not by one evaluation.

**Consequence for 2d:** a z-scored PetShop run (d=29-35) costs about 3-4 min per restart per method. Five restarts x 4 scenarios x 5 seeds x 2 methods would be about 10 hours on one core, so the benchmark will use fewer restarts (probably 2) and/or an equal wall-clock budget per run, stated openly in the report. SPOT should also finish faster (oracle skeleton: 3-6x fewer evaluations), which feeds the runtime comparison.
