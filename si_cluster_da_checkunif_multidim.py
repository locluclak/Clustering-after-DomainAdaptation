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


ns, nt, d = 100, 50, 20
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



def test_statistic(X, X_vec, Xt, ns, nt, d, n_clusters, Sigma, labels_all_obs, members_all_obs):

    c1, c2 = np.random.choice(n_clusters, 2, replace=False)
    idx_cluster_c1 = np.argwhere(labels_all_obs[-1][ns:] == c1).flatten()
    idx_cluster_c2 = np.argwhere(labels_all_obs[-1][ns:] == c2).flatten()
    if idx_cluster_c1.size == 0 or idx_cluster_c2.size == 0:
        return None
   
            
    # cluster_c1_obs = members_all_obs[-1][c1]
    # cluster_c2_obs = members_all_obs[-1][c2]

    
    I_d = np.identity(d)
    
    eta_c1_idx = np.zeros((nt, 1))
    eta_c1_idx[idx_cluster_c1] = 1 / len(idx_cluster_c1)
    eta_c1 = np.kron(I_d, eta_c1_idx)
    
    
    eta_c2_idx = np.zeros((nt, 1))
    eta_c2_idx[idx_cluster_c2] = 1 / len(idx_cluster_c2)
    eta_c2 = np.kron(I_d, eta_c2_idx)
   
    eta_tmp = eta_c1 - eta_c2
    # print(eta_tmp)
    
    sign_tmp = np.dot(eta_c1_idx.T, Xt) - np.dot(eta_c2_idx.T, Xt)
    # print(sign_tmp)
    sign = np.transpose(np.sign(sign_tmp))
    # print(sign)
    eta_sign = np.dot(eta_tmp, sign)

    eta = np.vstack((np.zeros((ns*d, 1)), eta_sign))
    etaTXvec = np.dot(eta.T, X_vec)
    # print(etaTXvec)

    etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta)
    b = np.dot(np.dot(Sigma, eta), np.linalg.inv(etaT_Sigma_eta))
    a = np.dot(np.identity(X_vec.shape[0]) - np.dot(b, eta.T), X_vec)
    z = etaTXvec.item()
    
    return {
        "a": a,
        "b": b,
        "zobs": z,
        "etaT_Sigma_eta": etaT_Sigma_eta.item(),
        "c1": c1,
        "c2": c2,
        "cluster_c1_obs": idx_cluster_c1,
        "cluster_c2_obs": idx_cluster_c2,
        "sign": sign
    }
    
def overconditioning(model, X, a, b, np_wdgrl, n_clusters, initial_centroids_obs, labels_all_obs, members_all_obs,z=0,X_=None):
    st = time.time()
    # print("a+b*z - X",np.sum(a+b*z - X))
    if device == "cpu":
        interval_da, a_, b_ = construct_interval.ReLUcondition(model.encoder, a, b, X)
    else:        
        interval_da, a_, b_ = conditioning.get_dnn_interval(X,a,b,np_wdgrl)
        interval_da = [interval_da]
    # print("a_+b_*z - X_",np.sum(a_+b_*z - X_))



    # st1 = time.time()
    # interval_kmean = construct_interval.KMeancondition(X.shape[0], n_clusters, a_, b_, initial_centroids_obs, labels_all_obs, members_all_obs,z)
    interval_kmean = construct_interval.KMeancondition2(X.shape[0], n_clusters, a_, b_, initial_centroids_obs, labels_all_obs, members_all_obs,z)
    # st15 = time.time()
    # interval_kmean = construct_interval.KMeanconditionCUPY(X.shape[0], n_clusters, a_, b_, initial_centroids_obs, labels_all_obs, members_all_obs,z)
    # st16 = time.time()
    # interval_kmean = construct_interval.KMeancondition3(X.shape[0], n_clusters, a_, b_, initial_centroids_obs, labels_all_obs, members_all_obs,z)
    # p, q, o = construct_interval.KMeancondition2(X.shape[0], n_clusters, a_, b_, initial_centroids_obs, labels_all_obs, members_all_obs)
    # st17 = time.time()
    # print("kmean2",st15-st1)
    # print("kmean3",st16-st15)
    # interval_kmean2 = construct_interval.solveinterval(p,q,o)
    # st2 = time.time()

    # print("Interval by DA:", interval_da)
    # print(f"Time constructing interval by DA: {st1 - st:.4f} seconds")
    # print(f"Time constructing interval by Kmean: {st15 - st1:.4f} seconds")
    # print(f"Time constructing interval by Kmean2: {st17 - st15:.4f} seconds")
    final_interval = util.interval_intersection(interval_da, interval_kmean)
    return final_interval
def parametric(model, X, a, b,np_wdgrl, n_clusters, c1, c2, c1_obs, c2_obs, zmin = -20, zmax = 20, log=None, seed=None):
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
            initial_centroids_obs, labels_all_obs, members_all_obs = kmeans(Xdeltaz_transformed, n_clusters)
            
            # oc = util.interval_intersection(intervalFS,intervalDA)
            oc = overconditioning(model, Xdeltaz, a, b, np_wdgrl, n_clusters, initial_centroids_obs, labels_all_obs, members_all_obs, z=z,X_=Xdeltaz_transformed)
            idx_cluster_c1 = np.argwhere(labels_all_obs[-1][ns:] == c1).flatten()
            idx_cluster_c2 = np.argwhere(labels_all_obs[-1][ns:] == c2).flatten()

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
def vec(A):
    vec = A.reshape(-1)
    return vec.reshape(-1,1)
def run(mu_s, mu_t, K, device,_=None):
    global final_model
    dataseed = None #random.randint(0, 2**32 - 1)  # 32-bit seed
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
    X_vec = np.vstack((Xs_vec, Xt_vec)).copy()
    X_origin = np.vstack((Xs, Xt))
    X_transformed = np.vstack((xs_hat, xt_hat))

    initial_centroids_obs, labels_all_obs, members_all_obs = kmeans(X_transformed, K)
    # print(labels_all_obs)
    Sigma = np.identity(n*d)
    try:
        a, b, etaTX, etaT_Sigma_eta, c1, c2, c1_obs, c2_obs, sign = test_statistic(X_origin, X_vec, Xt, ns, nt, d, K, Sigma, labels_all_obs, members_all_obs).values()
    except:
        print("test statistic is none") 
        return None
    a_2d = a.reshape(n, d)
    b_2d = b.reshape(n, d)
    # with torch.no_grad():
    #     a_hat = final_model.extract_feature(torch.from_numpy(a_2d).double().to(device)).cpu().numpy()
    #     b_hat = final_model.extract_feature(torch.from_numpy(b_2d).double().to(device)).cpu().numpy()
    # print("run", np.sum(a_hat + b_hat*etaTX - X_transformed))
    np_wdgrl = None# operations.convert_network_to_numpy(final_model.encoder)
    # final_model.encoder = final_model.encoder.to(device)
    # final_interval = overconditioning(final_model, X_origin, a_2d, b_2d,np_wdgrl, K, initial_centroids_obs, labels_all_obs, members_all_obs,z=etaTX, X_=X_transformed)
    st = time.time()
    final_interval = parametric(final_model, X_origin, a_2d, b_2d,np_wdgrl, K, c1, c2, c1_obs, c2_obs, zmin=-20, zmax=20,seed=None)
    en = time.time()
    print(f"Time for parametric: {en-st:.4f} seconds")
    selective_p_value = util.compute_p_value(final_interval, etaTX, etaT_Sigma_eta)
    print(f"p-value:", selective_p_value)

    with open('logs/selective_inference_log/p_values.txt', 'a') as f:
        f.write(f"{selective_p_value}\n")
    return selective_p_value
    # except Exception as e:
    #     print("Error during run:", e)
    #     # print("Data seed:", dataseed)
    #     return None
if __name__ == "__main__":

    list_p_values = []
    iteration = 1

    st = time.time()
    for i in range(iteration):
        print(f"\n--- Iteration {i+1}/{iteration} ---")
        p_value = run(mu_s, mu_t, K, device)
        if p_value is not None:
            list_p_values.append(p_value)

    # print("Running time:", time.time() - st, "(s)")
    # underalpha = sum(1 for p in list_p_values if p <= 0.05)
    # print('\nFalse positive rate:', underalpha/len(list_p_values), 'out of', len(list_p_values))

    # # Kiểm định thống kê
    # kstest = stats.kstest(list_p_values, stats.uniform(loc=0.0, scale=1.0).cdf)

    # # Hiển thị histogram
    # plt.hist(list_p_values)
    # plt.savefig('logs/selective_inference_log/p_values_histogram.png')
    # with open('logs/selective_inference_log/p_values.txt', 'a') as f:

    #     f.write(f"\nFalse positive rate: {underalpha/len(list_p_values)} out of {len(list_p_values)}\n")
    #     f.write(f"\nKS test statistic: {kstest.statistic}, p-value: {kstest.pvalue}\n")
