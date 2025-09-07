import numpy as np
from scipy import stats
import torch
import matplotlib.pyplot as plt
import yaml
import time
from tqdm import tqdm
import random
from functools import partial

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
        "c1": c1,
        "c2": c2,
        "cluster_c1_obs": idx_cluster_c1,
        "cluster_c2_obs": idx_cluster_c2,
    }

def overconditioning(model, X, a, b, n_clusters, initial_centroids_obs, labels_all_obs, members_all_obs,z=0,X_=None):
    # st = time.time()
    # print(np.sum(a+b*z - X))
    
    interval_da, a_, b_ = construct_interval.ReLUcondition(model.encoder, a, b, X)
    # print(np.sum(a_+b_*z - X_))
    # st1 = time.time()
    interval_kmean = construct_interval.KMeancondition(X.shape[0], n_clusters, a_, b_, initial_centroids_obs, labels_all_obs, members_all_obs)
    # st15 = time.time()
    # p, q, o = construct_interval.KMeancondition2(X.shape[0], n_clusters, a_, b_, initial_centroids_obs, labels_all_obs, members_all_obs)
    # st17 = time.time()
    # interval_kmean2 = construct_interval.solveinterval(p,q,o)
    # st2 = time.time()

    # print("Interval by DA:", interval_da)
    # print(f"Interval by kmean: {interval_kmean}")
    # print(f"Interval by kmean2: {interval_kmean2}")
    # print(f"Time constructing interval by DA: {st1 - st:.4f} seconds")
    # print(f"Time constructing interval by KMean1: {st15 -st1:.4f} seconds")
    # print(f"Time constructing interval by mid KMean2: {st17 -st15:.4f} seconds")
    # print(f"Time constructing interval by KMean2: {st2 -st15:.4f} seconds")
    final_interval = util.interval_intersection(interval_da, interval_kmean)
    return final_interval

def parametric(model, X, a, b, n_clusters, c1, c2, c1_obs, c2_obs, zmin = -20, zmax = 20, log=None):
    n, d = X.shape
    z =  zmin
    zmax = zmax
    countitv=0
    Z = []
    stepsize= 0.00001
    # approximate total iterations
    total_steps = int((zmax - zmin) / stepsize)

    with tqdm(total=total_steps) as pbar:
        while z < zmax:
            z += stepsize
            # print("z =",z)
            Xdeltaz = (a + b*z).reshape(n, d)
            Xdeltaz_torch = torch.from_numpy(Xdeltaz).double()#.cuda()
            with torch.no_grad():
                Xdeltaz_transformed = final_model.extract_feature(Xdeltaz_torch).cpu().numpy()

            initial_centroids_obs, labels_all_obs, members_all_obs = kmeans(Xdeltaz_transformed, n_clusters)
            
            # oc = util.interval_intersection(intervalFS,intervalDA)
            oc = overconditioning(model, Xdeltaz, a, b, n_clusters, initial_centroids_obs, labels_all_obs, members_all_obs, z=z,X_=Xdeltaz_transformed)
            idx_cluster_c1 = np.argwhere(labels_all_obs[-1] == c1).flatten()
            idx_cluster_c2 = np.argwhere(labels_all_obs[-1] == c2).flatten()
            # if sorted(M) == sorted(M_z):
            if np.array_equal(c1_obs, idx_cluster_c1) and np.array_equal(c2_obs, idx_cluster_c2):
                Z = util.interval_union(Z, oc)
                countitv+=1
            # print("oc:", oc)
            # print("z :", z)
            z = oc[-1][1] # ruv
            # en = time.time()
            # with open(f"./experiments/time_{n}_{p}.txt", "a") as f:
            #     f.write(f"{en-st}\n")
            pbar.update(int((z - zmin) / stepsize) - pbar.n)

    if log is not None:
        with open(log, "a") as f:
            f.write(f"Number of intervals: {countitv}\n\n")
            f.write(f"Final interval: {Z}\n")
    # print("Final interval:", Z)
    return Z

def run(final_model, mu_s, mu_t, K, device):

    dataseed = random.randint(0, 2**32 - 1)  # 32-bit seed
    # print("Data seed:", dataseed)
    # ---- Generate synthetic data ----
    try:
        Xs = gendata.sample_normal_data(mu=mu_s, sigma=1, random_state=dataseed)
        Xt = gendata.sample_normal_data(mu=mu_t, sigma=1, random_state=dataseed)
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        d = Xs.shape[1]
        n = ns + nt

        Xs_torch = torch.from_numpy(Xs).double().to(device)
        Xt_torch = torch.from_numpy(Xt).double().to(device)

        with torch.no_grad():
            xs_hat = final_model.extract_feature(Xs_torch).cpu().numpy()
            xt_hat = final_model.extract_feature(Xt_torch).cpu().numpy()

        X_origin = np.vstack((Xs, Xt))
        X_transformed = np.vstack((xs_hat, xt_hat))

        initial_centroids_obs, labels_all_obs, members_all_obs = kmeans(X_transformed, K)

        Sigma = np.identity(n)
        a, b, etaTX, etaT_Sigma_eta, c1, c2, c1_obs, c2_obs = test_statistic(X_origin, K, Sigma, labels_all_obs, members_all_obs).values()

        # final_interval = overconditioning(final_model, X_origin, a, b, K, initial_centroids_obs, labels_all_obs, members_all_obs)
        st = time.time()
        final_interval = parametric(final_model, X_origin, a, b, K, c1, c2, c1_obs, c2_obs, zmin=-20, zmax=20)
        en = time.time()
        # print(f"Time for parametric: {en-st:.4f} seconds")
        selective_p_value = util.compute_p_value(final_interval, etaTX, etaT_Sigma_eta)
        return selective_p_value
    except Exception as e:
        print("Error during run:", e)
        print("Data seed:", dataseed)
        return None
if __name__ == "__main__":
    ns, nt, d = 50, 20, 1
    K = 3
    mu_s = np.full((ns, d), 2)
    mu_t = np.full((nt, d), 0)
    # ---- Load WDGRL model ----
    device = "cpu"

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
        device=device,
    )

    final_model.load_model("trained_model/20250907-112204")
    
    
    import os
    import multiprocessing
    num_cores = multiprocessing.cpu_count() 

    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1"

    list_p_values = []
    compute_pvalue_with_args = partial(run, final_model, mu_s, mu_t, K, device)
    iteration = 120
    with multiprocessing.Pool(processes=num_cores) as pool:
        list_p_values = pool.map(compute_pvalue_with_args, range(iteration))
    print("\nSelective p-value:", list_p_values)

    underalpha = sum(1 for p in list_p_values if p <= 0.05)
    print('\nFalse positive rate:', underalpha/len(list_p_values), 'out of', len(list_p_values))

    # Kiểm định thống kê
    kstest = stats.kstest(list_p_values, stats.uniform(loc=0.0, scale=1.0).cdf)

    # Hiển thị histogram
    plt.hist(list_p_values)
    plt.savefig('logs/selective_inference_log/p_values_histogram.png')
    with open('logs/selective_inference_log/p_values.txt', 'a') as f:
        for p_value in list_p_values:
            f.write(f"{p_value}\n")

        f.write(f"\nFalse positive rate: {underalpha/len(list_p_values)} out of {len(list_p_values)}\n")
        f.write(f"\nKS test statistic: {kstest.statistic}, p-value: {kstest.pvalue}\n")
