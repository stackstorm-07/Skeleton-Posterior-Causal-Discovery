> PetShop = one dataset per scenario: the sd over seeds is optimizer variance, not data variance. Corrected metrics unless marked legacy.

## Preprocessing `gauss_rank`, no wall-clock budget

### Across scenarios (scenario seed-means)

| variant   |   n_scenarios |   skel F1 |   head F1 |   tail F1 | skel F1 Δ vs ABIC    | head F1 Δ vs ABIC    | tail F1 Δ vs ABIC    |
|:----------|--------------:|----------:|----------:|----------:|:---------------------|:---------------------|:---------------------|
| abic      |             3 |     0.514 |     0.301 |     0.056 | nan                  | nan                  | nan                  |
| spot_cal  |             3 |     0.55  |     0.317 |     0.105 | +0.036 (W 3/L 0/T 0) | +0.016 (W 2/L 1/T 0) | +0.049 (W 2/L 0/T 1) |
| spot      |             3 |     0.535 |     0.297 |     0.063 | +0.021 (W 3/L 0/T 0) | -0.004 (W 2/L 1/T 0) | +0.007 (W 1/L 1/T 1) |

| variant   |   n_scenarios |   skel F1 (legacy) |   head F1 (legacy) |   tail F1 (legacy) | skel F1 (legacy) Δ vs ABIC   | head F1 (legacy) Δ vs ABIC   | tail F1 (legacy) Δ vs ABIC   |
|:----------|--------------:|-------------------:|-------------------:|-------------------:|:-----------------------------|:-----------------------------|:-----------------------------|
| abic      |             3 |              0.514 |              0.479 |              0.271 | nan                          | nan                          | nan                          |
| spot_cal  |             3 |              0.55  |              0.488 |              0.385 | +0.036 (W 3/L 0/T 0)         | +0.009 (W 2/L 1/T 0)         | +0.115 (W 2/L 1/T 0)         |
| spot      |             3 |              0.535 |              0.471 |              0.35  | +0.021 (W 3/L 0/T 0)         | -0.007 (W 1/L 2/T 0)         | +0.079 (W 2/L 1/T 0)         |

### high_traffic (n=589, d=35, 9 latent nodes; 3 seeds)

Corrected

| variant   |   n_graphs | skel F1       | head F1       | tail F1       | skel F1 Δ vs ABIC         | head F1 Δ vs ABIC         | tail F1 Δ vs ABIC         |
|:----------|-----------:|:--------------|:--------------|:--------------|:--------------------------|:--------------------------|:--------------------------|
| abic      |          3 | 0.513 ± 0.022 | 0.299 ± 0.015 | 0.167 ± 0.072 | nan                       | nan                       | nan                       |
| spot_cal  |          3 | 0.537 ± 0.017 | 0.301 ± 0.073 | 0.181 ± 0.203 | +0.025 (W 3/L 0, p=0.250) | +0.002 (W 1/L 2, p=1.000) | +0.014 (W 1/L 2, p=1.000) |
| spot      |          3 | 0.536 ± 0.023 | 0.258 ± 0.017 | 0.123 ± 0.004 | +0.023 (W 2/L 1, p=0.500) | -0.040 (W 0/L 3, p=0.250) | -0.044 (W 1/L 2, p=0.500) |

Legacy

| variant   |   n_graphs | skel F1 (legacy)   | head F1 (legacy)   | tail F1 (legacy)   | skel F1 (legacy) Δ vs ABIC   | head F1 (legacy) Δ vs ABIC   | tail F1 (legacy) Δ vs ABIC   |
|:----------|-----------:|:-------------------|:-------------------|:-------------------|:-----------------------------|:-----------------------------|:-----------------------------|
| abic      |          3 | 0.513 ± 0.022      | 0.479 ± 0.086      | 0.419 ± 0.074      | nan                          | nan                          | nan                          |
| spot_cal  |          3 | 0.537 ± 0.017      | 0.480 ± 0.105      | 0.385 ± 0.230      | +0.025 (W 3/L 0, p=0.250)    | +0.000 (W 1/L 2, p=1.000)    | -0.034 (W 1/L 2, p=0.750)    |
| spot      |          3 | 0.536 ± 0.023      | 0.452 ± 0.021      | 0.389 ± 0.096      | +0.023 (W 2/L 1, p=0.500)    | -0.027 (W 1/L 1, p=1.000)    | -0.030 (W 1/L 1, p=1.000)    |

Estimated vs true graph size

| variant   |   est_directed |   est_bidirected |   true_directed |   true_bidirected |   true_skeleton_edges |
|:----------|---------------:|-----------------:|----------------:|------------------:|----------------------:|
| abic      |           12.3 |             39.3 |              38 |                 0 |                    38 |
| spot_cal  |           14.3 |             32   |              38 |                 0 |                    38 |
| spot      |           13.3 |             37   |              38 |                 0 |                    38 |

Runtime (paired means)

| variant   |   n_graphs |   t_optimize_s |   t_posterior_s |   t_total_s |   n_fev | converged   | timed_out   |   est_directed |   est_bidirected | pairs_free   | time_ratio_vs_abic   | fev_ratio_vs_abic   |
|:----------|-----------:|---------------:|----------------:|------------:|--------:|:------------|:------------|---------------:|-----------------:|:-------------|:---------------------|:--------------------|
| abic      |          3 |          192.2 |            0    |       195.3 |  141263 | 3/3         | 0/3         |           12.3 |             39.3 | all          | nan                  | nan                 |
| spot_cal  |          3 |           78.4 |            0.41 |        81.8 |   61851 | 3/3         | 0/3         |           14.3 |             32   | 108          | 0.41x                | 0.43x               |
| spot      |          3 |           92.5 |            0.41 |        93.7 |   69587 | 3/3         | 0/3         |           13.3 |             37   | 221          | 0.48x                | 0.49x               |

### low_traffic (n=589, d=29, 13 latent nodes; 3 seeds)

Corrected

| variant   |   n_graphs | skel F1       | head F1       | tail F1       | skel F1 Δ vs ABIC         | head F1 Δ vs ABIC         | tail F1 Δ vs ABIC         |
|:----------|-----------:|:--------------|:--------------|:--------------|:--------------------------|:--------------------------|:--------------------------|
| abic      |          3 | 0.484 ± 0.023 | 0.288 ± 0.022 | 0.000 ± 0.000 | nan                       | nan                       | nan                       |
| spot_cal  |          3 | 0.521 ± 0.034 | 0.243 ± 0.018 | 0.000 ± 0.000 | +0.037 (W 2/L 1, p=0.500) | -0.044 (W 0/L 3, p=0.250) | +0.000 (W 0/L 0, p=1.000) |
| spot      |          3 | 0.514 ± 0.021 | 0.298 ± 0.016 | 0.000 ± 0.000 | +0.030 (W 3/L 0, p=0.250) | +0.011 (W 2/L 1, p=0.500) | +0.000 (W 0/L 0, p=1.000) |

Legacy

| variant   |   n_graphs | skel F1 (legacy)   | head F1 (legacy)   | tail F1 (legacy)   | skel F1 (legacy) Δ vs ABIC   | head F1 (legacy) Δ vs ABIC   | tail F1 (legacy) Δ vs ABIC   |
|:----------|-----------:|:-------------------|:-------------------|:-------------------|:-----------------------------|:-----------------------------|:-----------------------------|
| abic      |          3 | 0.484 ± 0.023      | 0.455 ± 0.083      | 0.127 ± 0.111      | nan                          | nan                          | nan                          |
| spot_cal  |          3 | 0.521 ± 0.034      | 0.425 ± 0.044      | 0.355 ± 0.150      | +0.037 (W 2/L 1, p=0.500)    | -0.030 (W 1/L 2, p=0.750)    | +0.227 (W 2/L 0, p=0.500)    |
| spot      |          3 | 0.514 ± 0.021      | 0.400 ± 0.092      | 0.244 ± 0.077      | +0.030 (W 3/L 0, p=0.250)    | -0.055 (W 1/L 2, p=0.500)    | +0.117 (W 2/L 0, p=0.500)    |

Estimated vs true graph size

| variant   |   est_directed |   est_bidirected |   true_directed |   true_bidirected |   true_skeleton_edges |
|:----------|---------------:|-----------------:|----------------:|------------------:|----------------------:|
| abic      |            7.3 |             25   |              30 |                 1 |                    31 |
| spot_cal  |            9.3 |             19.7 |              30 |                 1 |                    31 |
| spot      |            9.3 |             20.7 |              30 |                 1 |                    31 |

Runtime (paired means)

| variant   |   n_graphs |   t_optimize_s |   t_posterior_s |   t_total_s |   n_fev | converged   | timed_out   |   est_directed |   est_bidirected | pairs_free   | time_ratio_vs_abic   | fev_ratio_vs_abic   |
|:----------|-----------:|---------------:|----------------:|------------:|--------:|:------------|:------------|---------------:|-----------------:|:-------------|:---------------------|:--------------------|
| abic      |          3 |           82.5 |            0    |        84.7 |   90369 | 3/3         | 0/3         |            7.3 |             25   | all          | nan                  | nan                 |
| spot_cal  |          3 |           18.7 |            0.25 |        20.5 |   20787 | 3/3         | 0/3         |            9.3 |             19.7 | 58           | 0.24x                | 0.23x               |
| spot      |          3 |           31.6 |            0.25 |        33.5 |   35324 | 3/3         | 0/3         |            9.3 |             20.7 | 128          | 0.42x                | 0.41x               |

### temporal_traffic1 (n=1652, d=31, 10 latent nodes; 2 seeds)

Corrected

| variant   |   n_graphs | skel F1       | head F1       | tail F1       | skel F1 Δ vs ABIC         | head F1 Δ vs ABIC         | tail F1 Δ vs ABIC         |
|:----------|-----------:|:--------------|:--------------|:--------------|:--------------------------|:--------------------------|:--------------------------|
| abic      |          2 | 0.544 ± 0.005 | 0.317 ± 0.077 | 0.000 ± 0.000 | nan                       | nan                       | nan                       |
| spot_cal  |          2 | 0.590 ± 0.026 | 0.407 ± 0.034 | 0.133 ± 0.000 | +0.046 (W 2/L 0, p=0.500) | +0.089 (W 2/L 0, p=0.500) | +0.133 (W 2/L 0, p=0.500) |
| spot      |          2 | 0.554 ± 0.024 | 0.334 ± 0.014 | 0.067 ± 0.094 | +0.010 (W 1/L 1, p=1.000) | +0.016 (W 1/L 1, p=1.000) | +0.067 (W 1/L 0, p=1.000) |

Legacy

| variant   |   n_graphs | skel F1 (legacy)   | head F1 (legacy)   | tail F1 (legacy)   | skel F1 (legacy) Δ vs ABIC   | head F1 (legacy) Δ vs ABIC   | tail F1 (legacy) Δ vs ABIC   |
|:----------|-----------:|:-------------------|:-------------------|:-------------------|:-----------------------------|:-----------------------------|:-----------------------------|
| abic      |          2 | 0.544 ± 0.005      | 0.501 ± 0.030      | 0.267 ± 0.094      | nan                          | nan                          | nan                          |
| spot_cal  |          2 | 0.590 ± 0.026      | 0.558 ± 0.018      | 0.417 ± 0.118      | +0.046 (W 2/L 0, p=0.500)    | +0.058 (W 2/L 0, p=0.500)    | +0.150 (W 1/L 0, p=1.000)    |
| spot      |          2 | 0.554 ± 0.024      | 0.561 ± 0.055      | 0.417 ± 0.118      | +0.010 (W 1/L 1, p=1.000)    | +0.060 (W 2/L 0, p=0.500)    | +0.150 (W 2/L 0, p=0.500)    |

Estimated vs true graph size

| variant   |   est_directed |   est_bidirected |   true_directed |   true_bidirected |   true_skeleton_edges |
|:----------|---------------:|-----------------:|----------------:|------------------:|----------------------:|
| abic      |            6.5 |               34 |              32 |                 1 |                    33 |
| spot_cal  |            8.5 |               28 |              32 |                 1 |                    33 |
| spot      |            7.5 |               28 |              32 |                 1 |                    33 |

Runtime (paired means)

| variant   |   n_graphs |   t_optimize_s |   t_posterior_s |   t_total_s |   n_fev | converged   | timed_out   |   est_directed |   est_bidirected | pairs_free   | time_ratio_vs_abic   | fev_ratio_vs_abic   |
|:----------|-----------:|---------------:|----------------:|------------:|--------:|:------------|:------------|---------------:|-----------------:|:-------------|:---------------------|:--------------------|
| abic      |          2 |           46.7 |            0    |        54.6 |   44214 | 2/2         | 0/2         |            6.5 |               34 | all          | nan                  | nan                 |
| spot_cal  |          2 |           39.1 |            0.28 |        43.9 |   37370 | 2/2         | 0/2         |            8.5 |               28 | 150          | 0.79x                | 0.83x               |
| spot      |          2 |           45.7 |            0.28 |        50.4 |   41770 | 2/2         | 0/2         |            7.5 |               28 | 244          | 0.91x                | 0.94x               |

### Runtime breakdown (all scenarios)

| variant   | offline_training_s     |   online_posterior_s |   optimization_s |   total_per_run_s |   fev | converged   | timed_out   |
|:----------|:-----------------------|---------------------:|-----------------:|------------------:|------:|:------------|:------------|
| abic      | n/a                    |                 0    |            114.7 |             118.6 | 97916 | 8/8         | 0/8         |
| spot_cal  | 203 (one-time, shared) |                 0.32 |             46.2 |              49.3 | 40331 | 8/8         | 0/8         |
| spot      | 203 (one-time, shared) |                 0.32 |             58   |              60.3 | 49784 | 8/8         | 0/8         |
