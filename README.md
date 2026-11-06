# Legacy dual-background alpha-glucosidase inhibitory peptide models

[![DOI](https://zenodo.org/badge/1448005823.svg)](https://doi.org/10.5281/zenodo.24002488)

Archived release: [10.5281/zenodo.24002488](https://doi.org/10.5281/zenodo.24002488)  
Concept DOI (all versions): [10.5281/zenodo.24002518](https://doi.org/10.5281/zenodo.24002518)

Model version matching the manuscript.

## Metrics

| Background | MCC | AUC | Features |
|---|---:|---:|---:|
| Human | 0.6144 | 0.8836 | 150 |
| Soybean | 0.4100 | 0.7911 | 150 |

## Data

- `data/human_background_training_set.csv`
- `data/soybean_background_training_set.csv`

Each file has exactly two columns:

- `sequence`: peptide sequence
- `label`: 1, positive; 0, negative

## Models

- `models/human_background_ensemble.pkl`
- `models/human_background_feature_mask.npy`
- `models/human_background_lgbm_params.pkl`
- `models/soybean_background_ensemble.pkl`
- `models/soybean_background_feature_mask.npy`
- `models/soybean_background_lgbm_params.pkl`

## Train

```bash
python src/train.py --data data/human_background_training_set.csv --output runs/human
python src/train.py --data data/soybean_background_training_set.csv --output runs/soybean
```

## Predict

```bash
python src/predict.py --model models/human_background_ensemble.pkl --mask models/human_background_feature_mask.npy --input peptides.csv --output scores.csv
```

Input file requires a `sequence` column. Output contains `sequence` and `probability`.
