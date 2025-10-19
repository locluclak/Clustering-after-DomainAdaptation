import numpy as np
from scipy import stats
import torch
import matplotlib.pyplot as plt
import yaml
import time
from tqdm import tqdm
import random
from gpu_accelerate import operations, conditioning

import utils.construct_interval as construct_interval
from utils.kmeans import kmeans
import utils.util as util
import gendata
from models.wdgrl import WDGRL


ns, nt, d = 250, 50, 20
K = 3
mu_s = np.full((ns, d), 2)
mu_t = np.full((nt, d), 0)
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

final_model.load_model("trained_model/20250927-154452-20dims")

# def conditional_power(M, )

def test_statistic(X_vec, Xt, ns, nt, d, n_clusters, Sigma, labels_all_obs,return_sign=False):

    c1, c2 = np.random.choice(n_clusters, 2, replace=False)
    idx_cluster_c1 = np.argwhere(labels_all_obs[-1][ns:] == c1).flatten()
    idx_cluster_c2 = np.argwhere(labels_all_obs[-1][ns:] == c2).flatten()
    if idx_cluster_c1.size == 0 or idx_cluster_c2.size == 0:
        return None

    I_d = np.identity(d)
    
    eta_c1_idx = np.zeros((nt, 1))
    eta_c1_idx[idx_cluster_c1] = 1 / len(idx_cluster_c1)
    eta_c1 = np.kron(I_d, eta_c1_idx)
    
    
    eta_c2_idx = np.zeros((nt, 1))
    eta_c2_idx[idx_cluster_c2] = 1 / len(idx_cluster_c2)
    eta_c2 = np.kron(I_d, eta_c2_idx)
   
    eta_tmp = eta_c1 - eta_c2
    sign_tmp = np.dot(eta_tmp.T, vec(Xt))
    sign = np.sign(sign_tmp).astype(int)
    # print("eta_tmp", eta_tmp)
    if return_sign:
        return sign
    eta_sign = np.dot(eta_tmp, sign)

    eta = np.vstack((np.zeros((ns*d, 1)), eta_sign))
    etaTXvec = np.dot(eta.T, X_vec)

    etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta)
    b = np.dot(np.dot(Sigma, eta), np.linalg.inv(etaT_Sigma_eta))
    a = np.dot(np.identity(X_vec.shape[0]) - np.dot(b, eta.T), X_vec)
    z = etaTXvec.item()
    
    return {
        "a": a,
        "b": b,
        "eta_tmp": eta_tmp,
        "zobs": z,
        "etaT_Sigma_eta": etaT_Sigma_eta.item(),
        "c1": c1,
        "c2": c2,
        "cluster_c1_obs": idx_cluster_c1,
        "cluster_c2_obs": idx_cluster_c2,
        "sign": sign
    }

def overconditioning(model, X,eta, a, b, np_wdgrl, n_clusters, initial_centroids_obs, labels_all_obs, members_all_obs,z=0,X_=None):
    if device == "cpu":
        interval_da, a_, b_ = construct_interval.ReLUcondition(model.encoder, a, b, X)
    else:        
        interval_da, a_, b_ = conditioning.get_dnn_interval(X,a,b,np_wdgrl)
        interval_da = [interval_da]

    # interval_kmean = construct_interval.KMeancondition(X.shape[0], n_clusters, a_, b_, initial_centroids_obs, labels_all_obs, members_all_obs,z)
    interval_kmean = construct_interval.KMeancondition2(X.shape[0], n_clusters, a_, b_, initial_centroids_obs, labels_all_obs, members_all_obs,z)
    # interval_kmean = construct_interval.KMeanconditionCUPY(X.shape[0], n_clusters, a_, b_, initial_centroids_obs, labels_all_obs, members_all_obs,z)
    interval_test_statistic = construct_interval.statistic_condition(eta, vec(a[ns:]), vec(b[ns:]), vec(X[ns:]))

    final_interval = util.interval_intersection(interval_test_statistic,
                      util.interval_intersection(interval_da, interval_kmean))
    return final_interval
def parametric(model, X, a, b, eta, np_wdgrl, n_clusters, c1, c2, c1_obs, c2_obs, signobs, zmin = -20, zmax = 20, log=None, seed=None):
    global device
    n, d = X.shape
    z =  zmin
    zmax = zmax
    countitv=0
    Z = []
    stepsize= 0.00001

    total_steps = int((zmax - zmin) / stepsize)
    with tqdm(total=total_steps, desc=f"Seed {seed}") as pbar:
        while z < zmax:
            z += stepsize
            # print("z =",z)
            Xdeltaz = a + b*z
            Xdeltaz_torch = torch.from_numpy(Xdeltaz).double().to(device)
            with torch.no_grad():
                # Xdeltaz_transformed = final_model.extract_feature(Xdeltaz_torch).cpu().numpy()
                Xdeltaz_transformed = model.extract_feature(Xdeltaz_torch).cpu().numpy()
            initial_centroids_z, labels_all_z, members_all_obs = kmeans(Xdeltaz_transformed, n_clusters)
            
            # print("sum xt",np.sum(Xdeltaz[ns:]))
            # sign_z = test_statistic(vec(Xdeltaz), Xdeltaz[ns:], ns, nt, d, n_clusters, Sigma=None, labels_all_obs=labels_all_z,return_sign=True)
            sign_z = np.sign(eta.T.dot(vec(Xdeltaz[ns:])))
            oc = overconditioning(model, Xdeltaz, eta, a, b, np_wdgrl, n_clusters, initial_centroids_z, labels_all_z, members_all_obs, z=z,X_=Xdeltaz_transformed)
            idx_cluster_c1 = np.argwhere(labels_all_z[-1][ns:] == c1).flatten()
            idx_cluster_c2 = np.argwhere(labels_all_z[-1][ns:] == c2).flatten()

            # print("sign obs",signobs.reshape(1,-1))
            # print("sign z",sign_z.reshape(1,-1))
            # print(np.array_equal(signobs, sign_z))
            # print("\nz:",z)
            if np.array_equal(c1_obs, idx_cluster_c1) and np.array_equal(c2_obs, idx_cluster_c2) and np.array_equal(signobs, sign_z):
                Z = util.interval_union(Z, oc)
                # print(oc)
                countitv+=1
            # if 1:
            #     print("sign obs",signobs.reshape(1,-1))
            #     print("sign z  ",sign_z.reshape(1,-1))
            # print("all oc:", oc)
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

def test_statistic_permutationtest(Xt, idx_cluster_c1, idx_cluster_c2):
    I_d = np.identity(d)
    
    eta_c1_idx = np.zeros((nt, 1))
    eta_c1_idx[idx_cluster_c1] = 1 / len(idx_cluster_c1)
    eta_c1 = np.kron(I_d, eta_c1_idx)
    
    
    eta_c2_idx = np.zeros((nt, 1))
    eta_c2_idx[idx_cluster_c2] = 1 / len(idx_cluster_c2)
    eta_c2 = np.kron(I_d, eta_c2_idx)
   
    eta_tmp = eta_c1 - eta_c2
    etaTx = np.abs(np.dot(eta_tmp.T, vec(Xt)))
    # print(np.sum(etaTx))

    return np.sum(etaTx)


def permutation_test(Xt, idx_cluster_c1, idx_cluster_c2, 
                     test_statistic_func, n_permutations=1000, random_state=None):
    """
    Permutation test for checking if two clusters differ significantly.
        
    Returns
    -------
    observed_stat : float
        Test statistic on the real data.
    p_value : float
        p-value from permutation test.
    permuted_stats : np.ndarray
        Distribution of permuted test statistics.
    """
    rng = np.random.default_rng(random_state)

    # observed test statistic
    observed_stat = test_statistic_func(Xt, idx_cluster_c1, idx_cluster_c2)

    # combine indices and labels
    all_indices = np.concatenate([idx_cluster_c1, idx_cluster_c2])
    n1 = len(idx_cluster_c1)

    permuted_stats = np.zeros(n_permutations)

    for i in range(n_permutations):
        # shuffle all indices
        rng.shuffle(all_indices)
        perm_idx1 = all_indices[:n1]
        perm_idx2 = all_indices[n1:]
        
        permuted_stats[i] = test_statistic_func(Xt, perm_idx1, perm_idx2)

    # two-sided p-value
    p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))

    return {
        "observed_stat": observed_stat,
        "p_value": p_value,
        "permuted_stats": permuted_stats
    }

def vec(A):
    vec = A.reshape(-1)
    return vec.reshape(-1,1)


def run(mu_s, mu_t, K, device,_=None):
    global final_model
    dataseed = _ #random.randint(0, 2**32 - 1)
    # print("Data seed:", dataseed)
    # ---- Generate synthetic data ----
    # try:
    Xs = gendata.sample_normal_data(mu=mu_s, sigma=1, random_state=dataseed)
    Xt = gendata.sample_normal_data(mu=mu_t, sigma=1, random_state=dataseed)
    ns = Xs.shape[0]
    nt = Xt.shape[0]
    d = Xs.shape[1]
    n = ns + nt

    Xs_torch = torch.from_numpy(Xs).double().to(device)
    Xt_torch = torch.from_numpy(Xt).double().to(device)
    # print(Xt_torch.device)  
    with torch.no_grad():
        xs_hat = final_model.extract_feature(Xs_torch).cpu().numpy()
        xt_hat = final_model.extract_feature(Xt_torch).cpu().numpy()

    
    Xs_vec = vec(Xs)
    Xt_vec = vec(Xt)
    X_vec = np.vstack((Xs_vec, Xt_vec))
    X_origin = np.vstack((Xs, Xt))
    X_transformed = np.vstack((xs_hat, xt_hat))

    initial_centroids_obs, labels_all_obs, members_all_obs = kmeans(X_transformed, K)
    # print(labels_all_obs)
    Sigma = np.identity(n*d)
    try:
        a, b, eta_tmp, etaTX, etaT_Sigma_eta, c1, c2, c1_obs, c2_obs, sign = test_statistic(X_vec, Xt, ns, nt, d, K, Sigma, labels_all_obs).values()
    except Exception as e:
        print("test statistic is none", e) 
        return None
    
    permutation_test_pvalue = permutation_test(Xt, c1_obs, c2_obs, test_statistic_permutationtest,)["p_value"]
    with open(f'logs/selective_inference_log/FPRpermutation_p_valueslist{ns}.txt', 'a') as f:
        f.write(f"{permutation_test_pvalue}\n")
    return permutation_test_pvalue
    a_2d = a.reshape(n, d)
    b_2d = b.reshape(n, d)

    np_wdgrl = None# operations.convert_network_to_numpy(final_model.encoder)
    # final_model.encoder = final_model.encoder.to(device)




    # final_interval = overconditioning(final_model, X_origin,eta_tmp, a_2d, b_2d,np_wdgrl, K, initial_centroids_obs, labels_all_obs, members_all_obs,z=etaTX, X_=X_transformed)
    final_interval = parametric(final_model, 
                                X_origin, 
                                a_2d, 
                                b_2d,
                                eta_tmp,
                                np_wdgrl, 
                                K, c1, c2, c1_obs, c2_obs, 
                                signobs = sign, 
                                zmin=-20, zmax=20,seed=dataseed)
    # final_interval = [(-np.inf, np.inf)]
    
    # print(etaTX)
    # print("Final interval",final_interval)
    selective_p_value = util.compute_p_value(final_interval, etaTX, etaT_Sigma_eta)
    print(f"test-stat: {etaTX}, p-value:", selective_p_value)

    with open(f'logs/selective_inference_log/FPRpara_p_valueslist{ns}.txt', 'a') as f:
        f.write(f"{selective_p_value}\n")
    return selective_p_value
    # except Exception as e:
    #     print("Error during run:", e)
    #     # print("Data seed:", dataseed)
    #     return None

import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run iterations from start to end index")
    parser.add_argument("start", type=int, nargs="?", default=1, help="Start iteration index (default: 0)")
    parser.add_argument("end", type=int, nargs="?", default=1, help="End iteration index (inclusive, default: 1)")

    list_p_values = []
    # iteration = 24

    args = parser.parse_args()


    for i in range(args.start, args.end + 1):
        print(f"\n--- Iteration {i}/{args.end} ---")
        p_value = run(mu_s, mu_t, K, device, i)
        if p_value is not None:
            list_p_values.append(p_value)

    # print("Running time:", time.time() - st, "(s)")
    # underalpha = sum(1 for p in list_p_values if p <= 0.05)
    # print('\nFalse positive rate:', underalpha/len(list_p_values), 'out of', len(list_p_values))

    # # Kiểm định thống kê
    # kstest = stats.kstest(list_p_values, 'uniform')
    # print(kstest)
    # # Hiển thị histogram
    # plt.hist(list_p_values)
    # plt.show()
    # plt.savefig('logs/selective_inference_log/p_values_histogram.png')


    # with open('logs/selective_inference_log/p_values.txt', 'a') as f:

    #     f.write(f"\nFalse positive rate: {underalpha/len(list_p_values)} out of {len(list_p_values)}\n")
    #     f.write(f"\nKS test statistic: {kstest.statistic}, p-value: {kstest.pvalue}\n")
