import numpy as np
from models.wdgrl import WDGRL
import pandas as pd
import statisticaltest_cluster_da as cda
from scipy.linalg import block_diag
d = 19
K = 7

device = "cpu"

final_model = WDGRL(
    input_dim=d,
    encoder_hidden_dims=[500,100],
    critic_hidden_dims=[100],
    alpha1=0.0001,
    alpha2=0.0001
)


final_model.load_model("logs\\20251018-143239-obesity")

X_source = pd.read_csv("obesitylevel/gender0_test.csv")
X_target = pd.read_csv("obesitylevel/gender1_test.csv")
# to numpy


X_source = X_source.to_numpy(dtype=np.float64)
X_target = X_target.to_numpy(dtype=np.float64)

iterations = 30
log_dir = "logs/selective_inference_log/log_realdata/obesity"

for i in range(iterations):
    X_source = X_source[np.random.choice(X_source.shape[0], 150, replace=False)]
    X_target = X_target[np.random.choice(X_target.shape[0], 100, replace=False)]

    ns = X_source.shape[0]
    nt = X_target.shape[0]

    cov_source = np.cov(X_source, rowvar=False, bias=False)
    cov_target = np.cov(X_target, rowvar=False, bias=False)

    cov_block = block_diag(cov_source, cov_target)
    Ins = np.eye(ns)
    Int = np.eye(nt)
    cov_vecXs = np.kron(cov_source, Ins)
    cov_vecXt = np.kron(cov_target, Int)
    Sigma = block_diag(cov_vecXs, cov_vecXt)

    oc_pvalue = cda.oc_test(final_model=final_model, Xs=X_source, Xt=X_target, Sigma=Sigma, K=K, device=device)
    with open(f"{log_dir}/obesity_pvalue_oc.txt", "a") as f:
        f.write(f"{oc_pvalue}\n")
    
    permutation_pvalue = cda.permu_test(final_model=final_model, Xs=X_source, Xt=X_target, Sigma=Sigma, K=K, device=device)
    with open(f"{log_dir}/obesity_pvalue_permutation.txt", "a") as f:
        f.write(f"{permutation_pvalue}\n")

    para_pvalue = cda.parametric_test(final_model=final_model, Xs=X_source, Xt=X_target, Sigma=Sigma, K=K, device=device)
    with open(f"{log_dir}/obesity_pvalue_para.txt", "a") as f:
        f.write(f"{para_pvalue}\n")

    print("Selective p-value after DA and clustering:", oc_pvalue)
    print("Parametric p-value after DA and clustering:", para_pvalue)
    print("Permutation p-value after DA and clustering:", permutation_pvalue)
