# Edge-Aware Intrusion Detection for Healthcare IoT

An evidence-bounded implementation of the core pipeline described in the 2025 IEEE CCET paper *Edge-Aware Intrusion Detection in Distributed Healthcare IoT via PCA and Feature Engineering*.

## Reproduced architecture

```text
traffic features
  -> train-only frequency encoding
  -> correlation filtering (> 0.75)
  -> StandardScaler
  -> PCA (2 components)
  -> [x1, x2, x1^2, x2^2, sin(x1), sin(x2)]
  -> Linear(6,16) + BatchNorm + LeakyReLU + Dropout(0.3)
  -> Linear(16,8) + LeakyReLU
  -> binary logit
```

The original IoT-ICU CSV files and original training code were not available in the local project archive. Therefore, the paper-reported 99.89% accuracy is documented as a historical paper result, not claimed as reproduced here. This repository supplies a leakage-resistant implementation, synthetic non-clinical traffic generator, tests, and a route for authorized real data.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
iot-ids-train --synthetic-rows 4000 --epochs 100 --output results/metrics.json
```

To use an authorized dataset, provide a CSV containing a binary `label` column:

```bash
iot-ids-train --input data/combined.csv --epochs 200
```

The pipeline drops `class` and `ip.proto` when present. Categorical mappings, correlation filtering, scaling, and PCA are fitted only on the training split to prevent evaluation leakage.

## Verification scope

- Tests verify the exact six-dimensional nonlinear transform reported in the paper.
- Synthetic results demonstrate that the implementation runs end to end; they are not clinical or publication results.
- No patient data, network captures, IEEE PDF, credentials, or restricted dataset files are included.

## Paper reference

Z. Zhang, Y. Qin, and B. V. D. Kumar, “Edge-Aware Intrusion Detection in Distributed Healthcare IoT via PCA and Feature Engineering,” 2025 IEEE 8th International Conference on Computer and Communication Engineering Technology (CCET), DOI: `10.1109/CCET66260.2025.11199427`.
