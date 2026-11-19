# Dummy baseline vs TCN (test set)

Dummy models are fit on train labels only. Random strategies: mean ± std over seeds 42-51.

## UI-PRMD Vicon test (4 movements)

| Model | Windows | Accuracy | Macro-F1 | Balanced Acc. | TCN F1 gap |
|---|---:|---:|---:|---:|---:|
| TCN (positions + angles) | 204 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| Dummy (most frequent) | 204 | 0.2451 | 0.0984 | 0.2500 | +0.9016 |
| Dummy (stratified) | 204 | 0.2083 ± 0.0198 | 0.2197 ± 0.0201 | 0.2085 ± 0.0199 | +0.7803 |
| Dummy (uniform) | 204 | 0.2069 ± 0.0267 | 0.2303 ± 0.0290 | 0.2070 ± 0.0266 | +0.7697 |

## Full test (Vicon + webcam idle)

| Model | Windows | Accuracy | Macro-F1 | Balanced Acc. | TCN F1 gap |
|---|---:|---:|---:|---:|---:|
| TCN (positions + angles) | 231 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| Dummy (most frequent) | 231 | 0.2165 | 0.0712 | 0.2000 | +0.9288 |
| Dummy (stratified) | 231 | 0.1965 ± 0.0208 | 0.1861 ± 0.0221 | 0.1883 ± 0.0225 | +0.8139 |
| Dummy (uniform) | 231 | 0.1996 ± 0.0253 | 0.1944 ± 0.0252 | 0.1945 ± 0.0259 | +0.8056 |
