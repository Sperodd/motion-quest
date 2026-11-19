# Model comparison (test set, 231 windows)

| Model | Type | Accuracy | Macro-F1 | Balanced Acc. | Recording Acc. | M01→M03 | M03→M01 | Params | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **TCN (proposed)** ✔ | trained from scratch | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 101,765 | 1 |
| LSTM (baseline) | trained from scratch | 0.9524 | 0.9558 | 0.9569 | 1.0000 | 0.0000 | 0.0000 | 64,325 | 4 |
| Dummy (most frequent) | label prior only | 0.2165 | 0.0712 | 0.2000 | — | — | — | — | — |
| Dummy (stratified) | label prior only | 0.1965 | 0.1861 | 0.1883 | — | — | — | — | — |
| Dummy (uniform) | label prior only | 0.1996 | 0.1944 | 0.1945 | — | — | — | — | — |

Final model: **TCN (proposed)** — test Macro-F1 1.0000로 가장 높음.
Dummy rows are mean values over seeds; see baseline_comparison.md for standard deviations.