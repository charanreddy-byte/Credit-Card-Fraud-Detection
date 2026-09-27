# Credit-Card Fraud Detection Using Machine Learning & Deep Learning

An end-to-end credit-card fraud detection project comparing **LightGBM** and a **PyTorch feed-forward neural network (FraudNet)** on the public Kaggle Credit Card Fraud Detection dataset.

The project covers exploratory data analysis, robust scaling, stratified validation, SMOTE-based imbalance handling, Optuna hyperparameter tuning, model evaluation, and SHAP explainability.

> **Project note:** The notebook and academic report intentionally document methodological limitations, including fitting the scaler before the train/test split and using the hold-out test set for LightGBM early stopping. The reported hold-out metrics should therefore be interpreted with that limitation in mind.

## Results at a glance

The hold-out set contains **56,962 transactions, including 98 fraud cases**.

| Metric | LightGBM | FraudNet (PyTorch) |
|---|---:|---:|
| PR-AUC | **0.8758** | 0.7053 |
| ROC-AUC | 0.9780 | **0.9824** |
| Fraud F1 | **0.8038** | 0.1011 |
| Fraud Recall | 0.8571 | **0.9184** |
| Fraud Precision | **0.7568** | 0.0535 |
| True Positives | 84 | 90 |
| False Positives | **27** | 1,592 |
| False Negatives | 14 | 8 |
| Transactions flagged | 111 (0.19%) | 1,682 (2.95%) |

At the fixed **0.5 decision threshold**, LightGBM provides a substantially better precision/recall balance on this dataset, while FraudNet catches six additional fraud cases at the cost of many more false alarms.

## Project objectives

- Understand the extreme class imbalance in credit-card transactions.
- Explore fraud patterns in amount, time, and PCA-derived features.
- Apply imbalance-aware modelling without applying SMOTE to validation/test data.
- Tune a gradient-boosted tree model using Optuna.
- Build a class-weighted PyTorch neural network as an independent benchmark.
- Evaluate models using PR-AUC, ROC-AUC, precision, recall, F1 and confusion matrices.
- Use SHAP to explain global and individual LightGBM predictions.
- Document methodological limitations and potential next steps.

## Dataset

The project uses the **Kaggle Credit Card Fraud Detection dataset**, originally collected by Worldline and the Machine Learning Group of ULB.

Dataset characteristics:

- 284,807 transactions
- 31 columns
- 492 fraudulent transactions
- Fraud rate: approximately 0.1727%
- Features `V1`–`V28`: anonymised PCA components
- `Time`: seconds elapsed from the first transaction
- `Amount`: transaction value
- `Class`: target (`0` = legitimate, `1` = fraud)

The dataset is **not included in this repository** because the source CSV is approximately 144 MB, which exceeds GitHub's normal single-file limit.

See [`DATASET.md`](DATASET.md) for download instructions.

## Methodology

### 1. Exploratory analysis

The notebook examines:

- Class imbalance
- Transaction amount distributions
- Transaction-time distributions
- Feature correlations
- Fraud-rate behaviour over time

### 2. Preprocessing

`V1`–`V28` are already PCA-derived and standardised in the source dataset. `Amount` and `Time` are transformed with `RobustScaler`.

### 3. Validation

- 80/20 stratified train/test split
- 5-fold stratified cross-validation for LightGBM tuning
- Random seed: `42`

### 4. Imbalance handling

For LightGBM:

- SMOTE is applied within CV training folds.
- `scale_pos_weight` is tuned with Optuna.

For FraudNet:

- No SMOTE is used.
- Binary cross-entropy with a positive-class weight is used.

### 5. LightGBM + Optuna

Optuna performs 30 trials, maximising mean cross-validated PR-AUC.

The selected configuration reported in the academic report includes:

- Learning rate: 0.1416
- Number of leaves: 252
- Maximum depth: 10
- Minimum child samples: 78
- Subsample: 0.680
- Column subsampling: 0.620
- `scale_pos_weight`: 360.5
- Early stopping with a maximum of 2,000 estimators

### 6. FraudNet

The neural network uses:

`30 inputs → 128 → 64 → 32 → 1 output`

Each hidden layer uses:

- Batch Normalisation
- ReLU
- 30% dropout

Training uses AdamW, class-weighted BCE loss, learning-rate reduction on plateau, and early stopping.

### 7. Explainability

SHAP is used to understand LightGBM predictions.

The report identifies **V14** as the dominant SHAP feature, followed by features including V4, V8, V1, V12 and V3. Because the `V` variables are anonymised PCA components, SHAP explains model behaviour rather than directly interpretable business drivers.

## Repository structure

```text
credit-card-fraud-detection/
│
├── README.md
├── DATASET.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── fraud_detection_end_to_end.ipynb
│
├── reports/
│   └── Fraud_Detection_Academic_Report.pdf
│
├── data/
│   └── README.md
│
├── figures/
│   └── README.md
│
├── models/
│   └── README.md
│
├── src/
│   └── README.md
│
└── docs/
    └── README.md
```

The extra README files document currently empty folders so GitHub can preserve the intended project structure.

## How to run

### Option 1 — Google Colab

1. Download/open `notebooks/fraud_detection_end_to_end.ipynb`.
2. Upload it to Google Colab.
3. Provide a Kaggle API token when the notebook requests one.
4. Run the cells from top to bottom.

The notebook uses a CUDA GPU when one is available for FraudNet.

### Option 2 — Local Jupyter environment

Create a virtual environment, install the dependencies, download the dataset, and place:

```text
creditcard.csv
```

in the notebook working directory.

Then install:

```bash
pip install -r requirements.txt
```

and open the notebook with Jupyter.

## Important methodological limitations

The academic report identifies several limitations:

1. **Test-set early stopping:** the final LightGBM model uses the hold-out test set for early stopping, making the reported hold-out metrics slightly optimistic.
2. **Scaler leakage:** `RobustScaler` is fitted before the train/test split; strictly, it should be fitted on training data only.
3. **Few fraud cases:** only 98 fraud cases are present in the hold-out set, so individual errors have a noticeable effect on recall.
4. **Random split:** real fraud detection is temporal, so a chronological split would provide a more realistic estimate.
5. **Fixed threshold:** both models are evaluated at 0.5; threshold optimisation was not performed.
6. **Single dataset:** the source data covers two days of European transactions and may not generalise to other periods or regions.
7. **Anonymised features:** business interpretation of `V1`–`V28` is limited.

## Recommended future work

- Fit preprocessing only on the training data.
- Create a separate validation set for LightGBM early stopping.
- Evaluate with chronological train/validation/test splits.
- Select decision thresholds using an explicit fraud-vs-review cost matrix.
- Re-test FraudNet at a tuned threshold.
- Investigate model stacking/ensembling.
- Add drift monitoring such as Population Stability Index.
- Validate on a dataset containing real categorical and identity features, such as IEEE-CIS Fraud Detection.

## Author

**Charan Reddy**  
MBA — Data Science & Data Analytics  
Symbiosis Centre for Information Technology (SCIT), Pune

Academic report date: **20 September 2026**
