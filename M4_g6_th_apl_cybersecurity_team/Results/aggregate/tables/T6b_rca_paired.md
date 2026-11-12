| split   | comparison                                     |   cases |   mean_diff_MRR |   ci_lo |   ci_hi |   p_bootstrap | decision                       |
|:--------|:-----------------------------------------------|--------:|----------------:|--------:|--------:|--------------:|:-------------------------------|
| all     | truth graph: ADMG vs directed-only             |      36 |           0.000 |   0.000 |   0.000 |         1.000 | not distinguishable from noise |
| test    | truth graph: ADMG vs directed-only             |      26 |           0.000 |   0.000 |   0.000 |         1.000 | not distinguishable from noise |
| all     | truth graph ADMG vs no graph (anomaly size)    |      36 |           0.068 |   0.001 |   0.149 |         0.047 | supported: first better        |
| test    | truth graph ADMG vs no graph (anomaly size)    |      26 |           0.094 |   0.002 |   0.205 |         0.037 | supported: first better        |
| all     | truth graph ADMG vs correlation baseline       |      36 |           0.107 |  -0.030 |   0.235 |         0.108 | not distinguishable from noise |
| test    | truth graph ADMG vs correlation baseline       |      26 |           0.026 |  -0.126 |   0.170 |         0.730 | not distinguishable from noise |
| all     | abic graph: ADMG vs directed-only              |      36 |          -0.014 |  -0.107 |   0.090 |         0.730 | not distinguishable from noise |
| test    | abic graph: ADMG vs directed-only              |      26 |          -0.052 |  -0.169 |   0.090 |         0.395 | not distinguishable from noise |
| all     | abic graph ADMG vs no graph (anomaly size)     |      36 |          -0.017 |  -0.109 |   0.086 |         0.700 | not distinguishable from noise |
| test    | abic graph ADMG vs no graph (anomaly size)     |      26 |          -0.042 |  -0.163 |   0.098 |         0.466 | not distinguishable from noise |
| all     | abic graph ADMG vs correlation baseline        |      36 |           0.023 |  -0.112 |   0.154 |         0.726 | not distinguishable from noise |
| test    | abic graph ADMG vs correlation baseline        |      26 |          -0.109 |  -0.241 |   0.015 |         0.080 | not distinguishable from noise |
| all     | spot_cal graph: ADMG vs directed-only          |      36 |           0.049 |  -0.037 |   0.145 |         0.285 | not distinguishable from noise |
| test    | spot_cal graph: ADMG vs directed-only          |      26 |           0.029 |  -0.078 |   0.159 |         0.688 | not distinguishable from noise |
| all     | spot_cal graph ADMG vs no graph (anomaly size) |      36 |          -0.044 |  -0.153 |   0.072 |         0.425 | not distinguishable from noise |
| test    | spot_cal graph ADMG vs no graph (anomaly size) |      26 |          -0.080 |  -0.231 |   0.074 |         0.257 | not distinguishable from noise |
| all     | spot_cal graph ADMG vs correlation baseline    |      36 |          -0.005 |  -0.150 |   0.138 |         0.936 | not distinguishable from noise |
| test    | spot_cal graph ADMG vs correlation baseline    |      26 |          -0.148 |  -0.297 |  -0.009 |         0.036 | supported: first worse         |
| all     | ADMG ranking: true graph vs ABIC graph         |      36 |           0.084 |   0.030 |   0.136 |         0.001 | supported: first better        |
| test    | ADMG ranking: true graph vs ABIC graph         |      26 |           0.136 |   0.085 |   0.189 |         0.000 | supported: first better        |
| all     | ADMG ranking: true graph vs SPOT-cal graph     |      36 |           0.112 |   0.039 |   0.187 |         0.000 | supported: first better        |
| test    | ADMG ranking: true graph vs SPOT-cal graph     |      26 |           0.174 |   0.094 |   0.269 |         0.000 | supported: first better        |
| all     | ADMG ranking: SPOT-cal graph vs ABIC graph     |      36 |          -0.028 |  -0.069 |   0.008 |         0.155 | not distinguishable from noise |
| test    | ADMG ranking: SPOT-cal graph vs ABIC graph     |      26 |          -0.038 |  -0.097 |   0.011 |         0.134 | not distinguishable from noise |
