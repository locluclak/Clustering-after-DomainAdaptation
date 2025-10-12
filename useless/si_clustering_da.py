import numpy as np
from scipy import stats
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import yaml
import os

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# import utils.construct_interval as construct_interval
import utils.construct_interval as construct_interval
import util
import gendata
from models.wdgrl import WDGRL

def compute_z_interval(n, K, a, b, initial_centroids, labels_all, members_all):
    trunc_interval = [(-np.inf, np.inf)]
   
    for i in range(n):
        if  i == initial_centroids[labels_all[0][i]]:
            continue
        e_i_ci0 = np.zeros((1,n))
        e_i_ci0[0][i] = 1
        u1 = (a[i] - a[initial_centroids[labels_all[0][i]]]).reshape(-1,1)
        v1 = (b[i] - b[initial_centroids[labels_all[0][i]]]).reshape(-1,1)

        e_i_ci0[0][initial_centroids[labels_all[0][i]]] = -1
        A1 = np.eye(u1.shape[0])
        p1, q1, o1 = util.construct_p_q_t(u1, v1, A1)

        for k in range(K):
            if k == labels_all[0][i]:
                continue
            e_i_ck0 = np.zeros((1,n))
            e_i_ck0[0][i] = 1
            e_i_ck0[0][initial_centroids[k]] = -1
            #print(e_i_ck0)
            u2 = (a[i] - a[initial_centroids[k]]).reshape(-1,1)
            v2 = (b[i] - b[initial_centroids[k]]).reshape(-1,1)
            A2 = np.eye(u2.shape[0])
            p2, q2, o2 = util.construct_p_q_t(u2, v2, A2)

            p = (p1 - p2).item()
            q = (q1 - q2).item()
            o = (o1 - o2).item()

            res = util.solve_quadratic_inequality(p, q, o)
            # print("res",res)
            if res == "No solution":
                print(p, q, t)
                #continue
            #elif res[0][0] == -np.inf and res[0][1] == np.inf:
             #   continue
                #print(p, q, t, i, k) 
            else:
                
                trunc_interval = util.interval_intersection(trunc_interval,res)
    #             print("trunc_interval in loop", trunc_interval)
    # print("K-means truncation interval:", trunc_interval)

    for t in range(1, len(labels_all)):
        for i in range(n):
            e_i = np.zeros((1,n))
            e_i[0][i] = 1
            
            gamma_i = np.zeros((1,n))
            label_i = labels_all[t][i] 
            
            C_i_t_minus = list(members_all[t-1][label_i])  #cluster at iteration t-1 which forms centroid at iteration t
            if len(C_i_t_minus) == 0:
                    continue
    
            gamma_i[:,C_i_t_minus] = 1
        
            E_temp_1 = e_i - gamma_i/len(C_i_t_minus)
            u3 = E_temp_1.dot(a).reshape(-1,1)
            v3 = E_temp_1.dot(b).reshape(-1,1)
            E1 = np.eye(u3.shape[0])
            p3, q3, o3 = util.construct_p_q_t(u3, v3, E1)
            for k in range(K):
                e_i = np.zeros((1,n))
                e_i[0][i] = 1
                
                gamma_k = np.zeros((1,n))         
                C_k_t_minus = list(members_all[t-1][k])
                if len(C_k_t_minus) == 0:
                    continue
    
                if k == label_i:
                    continue
                gamma_k[:,C_k_t_minus] = 1
                
            
                E_temp_2 = e_i - gamma_k/len(C_k_t_minus)
                u4 = E_temp_2.dot(a).reshape(-1,1)
                v4 = E_temp_2.dot(b).reshape(-1,1)
                E2 = np.eye(u4.shape[0])

                p4, q4, o4 = util.construct_p_q_t(u4, v4, E2)

                p_comma = (p3 - p4).item()
                q_comma = (q3 - q4).item()
                o_comma = (o3 - o4).item()

                res = util.solve_quadratic_inequality(p_comma, q_comma, o_comma)
                if res == "No solution":
                    print(p_comma, q_comma, o_comma)
                else:
                    trunc_interval = util.interval_intersection(trunc_interval,res)
    return trunc_interval

if __name__ == "__main__":
    ns, nt, d = 200, 100, 1
    K = 3
    mu_s = np.full((ns,d),2)
    mu_t = np.full((nt,d),0)
    

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    exp_cfg = config["experiment"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    # ==== Generate data ====
    seed = exp_cfg["seed"]
    # ==== WDGRL model ====
    final_model = WDGRL(
        input_dim=d,
        encoder_hidden_dims=model_cfg["encoder_hidden_dims"],
        critic_hidden_dims=model_cfg["critic_hidden_dims"],
        alpha1=model_cfg["alpha1"],
        alpha2=model_cfg["alpha2"],
        seed=exp_cfg["model_random_state"],
    )

    final_model.load_model("trained_model/20250902-110038")


    max_iteration = 500
    list_p_value = []
    underalpha = 0
    Alpha = 0.05
    count = 0
    for i in range(max_iteration):
        Xs = gendata.sample_normal_data(mu=mu_s, sigma=1)
        Xt = gendata.sample_normal_data(mu=mu_t, sigma=1)
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        d = Xs.shape[1]
        n = ns + nt




        Xs_torch = torch.from_numpy(Xs).float()
        Xt_torch = torch.from_numpy(Xt).float()

        xs_hat = final_model.extract_feature(Xs_torch.cuda())
        xt_hat = final_model.extract_feature(Xt_torch.cuda())
        xs_hat = xs_hat.cpu().numpy()
        xt_hat = xt_hat.cpu().numpy()

        # kmean clustering
        X_origin = np.vstack((Xs, Xt))
        X_transformed = np.vstack((xs_hat, xt_hat))
        # print("sum origin", np.sum(X_origin))
        # print("sum transformed", np.sum(X_transformed))
        initial_centroids_obs, labels_all_obs, members_all_obs = util.kmeans(X_transformed, K)


        # print("initial_centroids_obs", initial_centroids_obs)
        # print("labels_all_obs", labels_all_obs)
        # print("members_all_obs", members_all_obs)
        c1, c2 = np.random.choice(K, 2, replace=False)

        

        idx_cluster_c1 = np.argwhere(labels_all_obs[-1] == c1)
        idx_cluster_c2 = np.argwhere(labels_all_obs[-1] == c2)
        cluster_c1_obs = list(members_all_obs[-1][c1])
        cluster_c2_obs = list(members_all_obs[-1][c2])
        # print(cluster_c1_obs, cluster_c2_obs)

        
        eta_c_u = np.zeros((n, 1))
        eta_c_u[idx_cluster_c1] = 1

        eta_c_v = np.zeros((n, 1))
        eta_c_v[idx_cluster_c2] = 1

        idx_cluster_c1[:ns] = 0
        idx_cluster_c2[:ns] = 0

        eta = eta_c_v/len(idx_cluster_c2) -  eta_c_u/len(idx_cluster_c1)
        etaTX = np.dot(eta.T, X_origin)
        # print(X_transformed.shape)
        # print(X_origin.shape)
        # print(etaTX)

        Sigma = np.identity(n) #cov
        etaT_Sigma_eta=np.dot(np.dot(eta.T, Sigma), eta)
        b = np.dot(np.dot(Sigma, eta), np.linalg.inv(etaT_Sigma_eta))
        a = np.dot(np.identity(n) - np.dot(b, eta.T), X_origin)
        z = np.dot(eta.T, X_origin)[0][0]

        interval_da, a_, b_ = construct_interval.ReLUcondition(final_model.encoder, a, b, X_origin)
        interval_kmean = compute_z_interval(n, K, a_, b_, initial_centroids_obs, labels_all_obs, members_all_obs)
        # threshold = 20

        # print("Sum a+bz", np.sum(a + b*z))
        # print("Sum a_+b_z", np.sum(a_ + b_*z))

        # print(interval_da)
        # print(interval_kmean)

        final_interval = util.interval_intersection(interval_da, interval_kmean)
        selective_p_value = util.compute_p_value(final_interval, z, etaT_Sigma_eta.item())
        # print("Final selective p-value", selective_p_value)


        if i % 5 == 0:
            print(i)

        if selective_p_value is not None:
            list_p_value.append(selective_p_value)
            count += 1
            if selective_p_value <= Alpha:
                underalpha += 1
    
    # Tính và in False positive rate
    print('\nFalse positive rate:', underalpha/count, count)

    # Kiểm định thống kê
    print(stats.kstest(list_p_value, stats.uniform(loc=0.0, scale=1.0).cdf))

    # Hiển thị histogram
    plt.hist(list_p_value, bins=20)

    plt.show()