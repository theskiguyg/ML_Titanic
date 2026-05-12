# Getting into classic Machine Learning

Created by Zagrebin Egor as part of the ML Engineering course based on the book "Hands-On Machine Learning" (Aurélien Géron, 1st edition, 2017). 

Goal: Building an end-to-end production-ready ML solution with clean code, configuration, and reproducibility. 

## Project Overview 

An end-to-end machine learning project based on Titanic dataset. The project covers the full cycle: from initial data analysis (EDA) to a production-ready pipeline managed through configuration files.

## Project structure

```text
ML_Titanic/
├── configs/
│   └── baseline.yaml              # Confiduration file (params, folds, etc.)
├── data/
│   ├── raw/                       # Data
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── gender_submission.csv  # Baseline 
│   └── submissions/               # Submitted predictions 
│       └── submission.csv
├── models/
│   └── model.pkl                  # Saved Pipeline
├── notebooks/
│   └── 01_eda.ipynb               # EDA
├── src/                           # Code
│   ├── __init__.py
│   ├── features.py                # Feature engineering
│   ├── preprocess.py              # Preprocessing data
│   ├── train.py                   # Fit + CV + GridSearch
│   └── predict.py                 # Inference + submit
├── .gitignore
├── README.md
└── requirements.txt
```

## Results & metrics 

| Metrics | Meaning |
|---------|----------|
| **Best Model** | Random Forest Classifier |
| **Cross-validation (5-fold)** | 0.815 ± 0.024 |
| **Full train set validation** | 0.857 |
| **Test accuracy (vs gender_submission)** | 0.9187 |

## Analysis of results

The model shows a high correlation with the baseline "female → survived, male → did not"**. This is expected, as gender is the main predictor on the Titanic. However, in ~8% of cases, the model makes decisions that deviate from this simple rule:

- Male children → survived
- Third-class women → did not survive
- Passengers with large families → did not survive

**Conclusion:** The model identified non-trivial patterns without degrading the baseline.