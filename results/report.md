# Self-Pruning Neural Network — Results Report


## Results Table

| Lambda | Test Accuracy (%) | Sparsity (%) |
|:------:|:-----------------:|:------------:|
| 0.1 | 90.96 | 0.00 |
| 1.0 | 91.06 | 0.00 |
| 5.0 | 90.94 | 2.03 |

## Layer-wise Sparsity (Best Model)

Best model: lambda = 1.0, Test Acc = 91.06%

| Layer | Sparsity (%) |
|:------|:------------:|
| PrunableLinear_1 (4096→256) | 0.0 |
| PrunableLinear_2 (256→128) | 0.0 |
| PrunableLinear_3 (128→10) | 0.0 |

