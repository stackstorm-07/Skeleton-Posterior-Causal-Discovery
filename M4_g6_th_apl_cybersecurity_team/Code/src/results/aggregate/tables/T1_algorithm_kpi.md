| setting                  | method                 | unit                 |   units | skeleton F1   | arrowhead F1   | tail F1       | SHD (PAG)     | time (s)        |
|:-------------------------|:-----------------------|:---------------------|--------:|:--------------|:---------------|:--------------|:--------------|:----------------|
| synthetic, n=1000, d=25  | ABIC                   | graph                |      10 | 0.676 ± 0.094 | 0.392 ± 0.120  | 0.074 ± 0.094 | 38.50 ± 8.48  | 180.55 ± 76.94  |
| synthetic, n=1000, d=25  | SPOT-cal (floor 6e-3)  | graph                |      10 | 0.756 ± 0.073 | 0.441 ± 0.118  | 0.079 ± 0.097 | 33.90 ± 8.09  | 126.98 ± 53.79  |
| synthetic, n=1000, d=25  | FCI (alpha=0.01)       | graph                |      10 | 0.869 ± 0.071 | 0.688 ± 0.122  | 0.447 ± 0.215 | 17.90 ± 5.82  | 3.31 ± 0.27     |
| synthetic, n=1000, d=25  | FCI (alpha=0.05)       | graph                |      10 | 0.877 ± 0.074 | 0.670 ± 0.124  | 0.369 ± 0.185 | 20.40 ± 7.31  | 3.32 ± 0.30     |
| synthetic, n=1000, d=25  | RFCI (alpha=0.01)      | graph                |      10 | 0.871 ± 0.071 | 0.677 ± 0.159  | 0.478 ± 0.304 | 16.90 ± 5.26  | 3.25 ± 0.22     |
| synthetic, n=1000, d=25  | RFCI (alpha=0.05)      | graph                |      10 | 0.885 ± 0.070 | 0.680 ± 0.116  | 0.440 ± 0.269 | 18.80 ± 6.60  | 3.31 ± 0.36     |
| PetShop, optimizer seeds | ABIC                   | (scenario, seed)     |      20 | 0.513 ± 0.032 | 0.293 ± 0.057  | 0.040 ± 0.063 | 48.85 ± 8.16  | 376.30 ± 269.26 |
| PetShop, optimizer seeds | SPOT-cal (floor 6e-3)  | (scenario, seed)     |      20 | 0.542 ± 0.042 | 0.317 ± 0.074  | 0.071 ± 0.087 | 44.95 ± 7.49  | 47.41 ± 18.21   |
| PetShop, optimizer seeds | SPOT (floor 1e-3)      | (scenario, seed)     |      20 | 0.539 ± 0.044 | 0.314 ± 0.051  | 0.073 ± 0.095 | 46.05 ± 7.27  | 113.30 ± 79.14  |
| PetShop, optimizer seeds | random posterior       | (scenario, seed)     |      12 | 0.196 ± 0.083 | 0.054 ± 0.065  | 0.000 ± 0.000 | 42.92 ± 4.94  | 65.34 ± 72.82   |
| PetShop, optimizer seeds | oracle posterior       | (scenario, seed)     |      12 | 0.813 ± 0.037 | 0.567 ± 0.054  | 0.204 ± 0.135 | 21.17 ± 2.25  | 33.57 ± 27.20   |
| PetShop, optimizer seeds | floor only             | (scenario, seed)     |      12 | 0.515 ± 0.065 | 0.299 ± 0.076  | 0.078 ± 0.108 | 46.50 ± 6.52  | 109.98 ± 132.17 |
| PetShop, optimizer seeds | mask only              | (scenario, seed)     |      12 | 0.527 ± 0.048 | 0.303 ± 0.063  | 0.074 ± 0.109 | 47.83 ± 7.60  | 140.58 ± 85.83  |
| PetShop, optimizer seeds | ABIC directed-only     | (scenario, seed)     |      12 | 0.458 ± 0.048 | 0.049 ± 0.062  | 0.030 ± 0.057 | 43.17 ± 6.09  | 82.03 ± 49.40   |
| PetShop, optimizer seeds | SPOT-cal directed-only | (scenario, seed)     |      12 | 0.459 ± 0.078 | 0.030 ± 0.038  | 0.008 ± 0.027 | 41.67 ± 4.79  | 42.12 ± 22.22   |
| PetShop, block bootstrap | ABIC                   | (scenario, resample) |      60 | 0.491 ± 0.042 | 0.273 ± 0.067  | 0.045 ± 0.081 | 51.27 ± 10.51 | 254.48 ± 168.80 |
| PetShop, block bootstrap | SPOT-cal (floor 6e-3)  | (scenario, resample) |      60 | 0.513 ± 0.046 | 0.286 ± 0.069  | 0.078 ± 0.076 | 48.47 ± 9.43  | 117.24 ± 127.49 |
| PetShop, FCI / RFCI      | FCI (alpha=0.01)       | scenario             |       4 | 0.497 ± 0.032 | 0.237 ± 0.032  | 0.105 ± 0.122 | 50.50 ± 10.38 | 3.24 ± 0.70     |
| PetShop, FCI / RFCI      | FCI (alpha=0.05)       | scenario             |       4 | 0.500 ± 0.057 | 0.254 ± 0.060  | 0.128 ± 0.095 | 58.25 ± 12.82 | 3.46 ± 0.83     |
| PetShop, FCI / RFCI      | RFCI (alpha=0.01)      | scenario             |       4 | 0.518 ± 0.031 | 0.283 ± 0.066  | 0.144 ± 0.099 | 47.50 ± 10.50 | 2.92 ± 0.47     |
| PetShop, FCI / RFCI      | RFCI (alpha=0.05)      | scenario             |       4 | 0.508 ± 0.041 | 0.253 ± 0.052  | 0.127 ± 0.043 | 57.50 ± 12.01 | 3.25 ± 0.62     |
