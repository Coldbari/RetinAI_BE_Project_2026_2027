# Architecture comparison

| arch               |   accuracy |   macro_f1 |    qwk |    auc |   params_M |   size_MB |   cpu_ms |   cpu_fps |   gpu_ms |   gpu_batch16_ms |   gpu_fps |
|:-------------------|-----------:|-----------:|-------:|-------:|-----------:|----------:|---------:|----------:|---------:|-----------------:|----------:|
| efficientnet_b0    |     0.7577 |     0.7569 | 0.5251 | 0.8669 |       4.01 |      15.5 |     56.2 |      17.8 |     7.25 |             21.9 |     137.9 |
| efficientnet_v2_s  |     0.7057 |     0.7006 | 0.4313 | 0.87   |      20.18 |      77.6 |    130.1 |       7.7 |    16.83 |             48.9 |      59.4 |
| mobilenet_v3_large |     0.7443 |     0.7431 | 0.5001 | 0.8574 |       4.2  |      16.1 |     34.4 |      29.1 |     6.13 |             13.1 |     163.1 |
| resnet50           |     0.8855 |     0.8822 | 0.7656 | 0.9635 |      23.51 |      89.9 |    157.2 |       6.4 |     7.28 |             46.1 |     137.4 |
| densenet121        |     0.7643 |     0.7639 | 0.5367 | 0.8866 |       6.96 |      26.9 |    136   |       7.4 |    14.39 |             53.4 |      69.5 |

Recommended: **resnet50** — best auc (0.9635) with competitive latency. Selection should be confirmed with the W9 statistical tests (McNemar / DeLong) before finalising.