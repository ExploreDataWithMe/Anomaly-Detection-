# Dataset setup

This repository intentionally does **not** commit the Kaggle datasets. Download them from Kaggle and place the extracted files under `data/`.

## Recommended datasets

| Stage | Dataset | Kaggle |
|---|---|---|
| Isolation Forest  | Credit Card Fraud Detection | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| GMM | Network Intrusion Detection | https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection |

After extraction, update the `DATA_PATH` / `CSV_PATH` variable in the corresponding notebook. The notebooks include defensive schema discovery so small filename differences do not break the workflow.


