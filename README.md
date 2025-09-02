## 1. Train WDGRL model with **train_save_wdgrl.py**
Modify with **config.yaml**

The trained model will be save in ./trained model

The ./models/wdgrl.py is the stable model, works well with current config file

## 2. Selective inference for clustering after domain adaptation (over-conditioning)
Run **si_clustering_da.py**

## Others
If you want to test the performance of clustering after WDGRL-based domain adaptation, run **wdgrl_clustering.py**. 

View results in **./logs**


