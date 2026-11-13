# Notebook Guide

Run notebooks in this order:

| Notebook | Purpose | Dataset |
|---|---|---|
| `Isolation_forest.ipynb` | Tree-based outlier detection + decision landscape | Credit Card Fraud |
| `Gmm_density.ipynb` | Probabilistic density estimation + BIC/AIC | Network Intrusion |

## Notebook conventions

Each notebook follows the same sequence:

1. Configuration.
2. Data loading and schema validation.
3. Preprocessing.
4. Train/validation/test split.
5. Model tuning or complexity selection.
6. Final fit.
7. Imbalance-aware evaluation.
8. High-resolution visualization.
9. Export to `reports/figures/`.

