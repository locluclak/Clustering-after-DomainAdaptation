import numpy as np
from scipy import stats
import torch
import matplotlib.pyplot as plt
import yaml
import time

import utils.construct_interval as construct_interval
from utils.kmeans import kmeans
import utils.util as util
import gendata
from models.wdgrl import WDGRL

def test_statistic(X, n_clusters, Sigma, labels_all_obs, members_all_obs):
    c1, c2 = np.random.choice(n_clusters, 2, replace=False)

    idx_cluster_c1 = np.argwhere(labels_all_obs[-1] == c1).flatten()
    idx_cluster_c2 = np.argwhere(labels_all_obs[-1] == c2).flatten()

    # cluster_c1_obs = members_all_obs[-1][c1]
    # cluster_c2_obs = members_all_obs[-1][c2]

    eta = np.zeros((X.shape[0], 1))
    eta[idx_cluster_c1] = 1 / len(idx_cluster_c1)
    eta[idx_cluster_c2] -= 1 / len(idx_cluster_c2)

    etaTX = np.dot(eta.T, X)
    etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta)
    b = np.dot(np.dot(Sigma, eta), np.linalg.inv(etaT_Sigma_eta))
    a = np.dot(np.identity(X.shape[0]) - np.dot(b, eta.T), X)
    z = etaTX.item()

    return {
        "a": a,
        "b": b,
        "zobs": z,
        "etaT_Sigma_eta": etaT_Sigma_eta.item(),
    }

if __name__ == "__main__":
    ns, nt, d = 200, 100, 1
    K = 3
    mu_s = np.full((ns, d), 2)
    mu_t = np.full((nt, d), 0)
    modelt = time.time()
    # ---- Load WDGRL model ----
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    exp_cfg = config["experiment"]
    model_cfg = config["model"]

    seed = exp_cfg["seed"]
    final_model = WDGRL(
        input_dim=d,
        encoder_hidden_dims=model_cfg["encoder_hidden_dims"],
        critic_hidden_dims=model_cfg["critic_hidden_dims"],
        alpha1=model_cfg["alpha1"],
        alpha2=model_cfg["alpha2"],
        seed=exp_cfg["model_random_state"],
    )

    final_model.load_model("trained_model/20250902-110038")
    print("Time of load model", time.time() - modelt)
    st = time.time()

    # ---- Generate synthetic data ----
    Xs = gendata.sample_normal_data(mu=mu_s, sigma=1)
    Xt = gendata.sample_normal_data(mu=mu_t, sigma=1)
    ns = Xs.shape[0]
    nt = Xt.shape[0]
    d = Xs.shape[1]
    n = ns + nt

    Xs_torch = torch.from_numpy(Xs).float().cuda()
    Xt_torch = torch.from_numpy(Xt).float().cuda()

    with torch.no_grad():
        xs_hat = final_model.extract_feature(Xs_torch).cpu().numpy()
        xt_hat = final_model.extract_feature(Xt_torch).cpu().numpy()

    X_origin = np.vstack((Xs, Xt))
    X_transformed = np.vstack((xs_hat, xt_hat))

    initial_centroids_obs, labels_all_obs, members_all_obs = kmeans(X_transformed, K)

    Sigma = np.identity(n)
    a, b, etaTX, etaT_Sigma_eta = test_statistic(X_origin, K, Sigma, labels_all_obs, members_all_obs).values()
    print("Time of data preparation:", time.time() - st)

    dast = time.time()
    interval_da, a_, b_ = construct_interval.ReLUcondition(final_model.encoder, a, b, X_origin)
    print("Time of construct interval_da:", time.time() - dast)
    kmeant = time.time()
    interval_kmean = construct_interval.KMeancondition(n, K, a_, b_, initial_centroids_obs, labels_all_obs, members_all_obs)
    print("Time of construct interval_kmean:", time.time() - kmeant)

    pvaluet = time.time()
    final_interval = util.interval_intersection(interval_da, interval_kmean)
    selective_p_value = util.compute_p_value(final_interval, etaTX, etaT_Sigma_eta)
    print("Time of computing pvalue:", time.time() - pvaluet)
    print("\nSelective p-value:", selective_p_value)