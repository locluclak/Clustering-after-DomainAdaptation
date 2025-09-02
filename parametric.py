import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from multiprocessing import Pool
from tqdm import tqdm 
import util

def compute_z_interval(n, K, a, b, initial_centroids, labels_all, members_all):
    trunc_interval = [(-np.inf, np.inf)]
   
    for i in range(n):
        if  i == initial_centroids[labels_all[0][i]]:
            continue
        e_i_ci0 = np.zeros((1,n))
        e_i_ci0[0][i] = 1
        
        e_i_ci0[0][initial_centroids[labels_all[0][i]]] = -1
        A1 = np.dot(e_i_ci0.T, e_i_ci0)
        p1, q1, o1 = util.construct_p_q_t(a, b, A1)
        for k in range(K):
            if k == labels_all[0][i]:
                continue
            e_i_ck0 = np.zeros((1,n))
            e_i_ck0[0][i] = 1
            e_i_ck0[0][initial_centroids[k]] = -1
            #print(e_i_ck0)
            A2 = np.dot(e_i_ck0.T, e_i_ck0)
            
            p2, q2, o2 = util.construct_p_q_t(a, b, A2)
            
            p = p1 - p2
            q = q1 - q2
            o = o1 - o2
            
            res = util.solve_quadratic_inequality(p[0][0], q[0][0], o[0][0])
            if res == "No solution":
                print(p, q, t)
                #continue
            #elif res[0][0] == -np.inf and res[0][1] == np.inf:
             #   continue
                #print(p, q, t, i, k) 
            else:
                
                trunc_interval = util.interval_intersection(trunc_interval,res)
                
                
    
    for t in range(1, len(labels_all)):
        for i in range(n):
            e_i = np.zeros((1,n))
            e_i[0][i] = 1
            
            gamma_i = np.zeros((1,n))
            label_i = labels_all[t][i] 
            
            C_i_t_minus = list(members_all[t-1][label_i])  #cluster at iteration t-1 which forms centroid at iteration t
           
            gamma_i[:,C_i_t_minus] = 1
        
            E_temp_1 = e_i - gamma_i/len(C_i_t_minus)
            
            E1 = np.dot(E_temp_1.T, E_temp_1)
            p3, q3, o3 = util.construct_p_q_t(a, b, E1)
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
                
                E2 = np.dot(E_temp_2.T, E_temp_2)
                
                p4, q4, o4 = util.construct_p_q_t(a, b, E2)
                
                p_comma = p3 - p4
                q_comma = q3 - q4
                o_comma = o3 - o4
                """
                if (q_comma[0][0])**2 - 4*p_comma[0][0]*o_comma[0][0] > 0:
                    print(p_comma, q_comma, o_comma)
                    print(p_comma[0][0]*z*z + q_comma[0][0]*z + o_comma[0][0], (X[i] - np.mean(X[C_i_t_minus]))**2 - (X[i] - np.mean(X[C_k_t_minus]))**2)
                    print(i, label_i, t)
                    print(C_i_t_minus, C_k_t_minus)
                    print(abs(np.dot(E_temp_1, X)), abs(X[i] - np.mean(X[C_i_t_minus])))
                    print(abs(np.dot(E_temp_2, X)), abs(X[i] - np.mean(X[C_k_t_minus])))           
                    print(labels_all[t])
                    print(labels_all[t-1])
                    print("sosss 222")
                    sys.exit(0)
                """
                res = util.solve_quadratic_inequality(p_comma[0][0], q_comma[0][0], o_comma[0][0])
                if res == "No solution":
                    print(p_comma, q_comma, o_comma)
                else:
                    trunc_interval = util.interval_intersection(trunc_interval,res)
    return trunc_interval

def line_search(a, b, n, K, threshold, u, v):
    zk = -threshold
    list_zk = [zk]
    list_interval = []
    list_centroid = []
    list_oc_labels = []
    while zk < threshold:
        X_zk = a + b*zk
        initial_centroids_zk, labels_all_zk, members_all_zk = util.kmeans(X_zk, K)
        cluster_u_zk = list(members_all_zk[-1][u])
        cluster_v_zk = list(members_all_zk[-1][v])
        list_centroid.append((cluster_u_zk, cluster_v_zk))
        list_oc_labels.append(labels_all_zk)
        oc_interval = compute_z_interval(n, K, a, b, initial_centroids_zk, labels_all_zk, members_all_zk)
        for each_interval in oc_interval:
            if each_interval[0] <= zk <= each_interval[1]:
                next_zk = each_interval[1]
                list_interval.append([each_interval[0], each_interval[1]])
                break

        zk = next_zk + 0.0001        #[[zk;...], [zk+1,...]]
        if zk < threshold:
            list_zk.append(zk)
        else:
            list_zk.append(threshold)
    return list_centroid, list_zk


def parametric(n, K):
    n = 20
    K = 3
    X = util.generate(n)


    initial_centroids_obs, labels_all_obs, members_all_obs = util.kmeans(X, K)

    u, v = np.random.choice(K, 2, replace=False)
    idx_cluster_u = np.argwhere(labels_all_obs[-1] == u)
    idx_cluster_v = np.argwhere(labels_all_obs[-1] == v)
    cluster_u_obs = list(members_all_obs[-1][u])
    cluster_v_obs = list(members_all_obs[-1][v])

    eta_c_u = np.zeros((n, 1))
    eta_c_u[idx_cluster_u] = 1
    eta_c_v = np.zeros((n, 1))
    eta_c_v[idx_cluster_v] = 1
    eta = eta_c_v/len(idx_cluster_v) -  eta_c_u/len(idx_cluster_u)
    etaTX = np.dot(eta.T, X)


    Sigma = np.identity(n) #cov
    etaT_Sigma_eta=np.dot(np.dot(eta.T, Sigma), eta)
    b = np.dot(np.dot(Sigma, eta), np.linalg.inv(etaT_Sigma_eta))
    a = np.dot(np.identity(n) - np.dot(b, eta.T), X)
    z = np.dot(eta.T, X)[0][0]
    threshold = 20
    
    list_centroid, list_zk = line_search(a, b, n, K, threshold, u, v)
    minimal_interval = []
    minimal_zk = []
    #print(cluster_u_obs, cluster_v_obs)
    for i in range(len(list_centroid)):
        cluster_u_zk, cluster_v_zk = list_centroid[i]
    #    print(list_centroid[i], list_zk[i+1])
        if np.array_equal(np.sort(cluster_u_obs), np.sort(cluster_u_zk)) and np.array_equal(np.sort(cluster_v_obs), np.sort(cluster_v_zk)):
            minimal_interval.append([list_zk[i], list_zk[i + 1] - 0.0001])
            minimal_zk.append(list_zk[i])
    #print(z_interval)
    #print(minimal_interval)
    new_z_interval = []
    for each_interval in minimal_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(each_interval)
        else:
            sub = each_interval[0] - new_z_interval[-1][1]
            if abs(sub) <= 0.001:
                new_z_interval[-1][1] = each_interval[1]
            else:
                new_z_interval.append(each_interval)
    #print("z interval", len(z_interval), z_interval)
    #print("new z interval", len(new_z_interval), new_z_interval)
    #print("z interval", len(z_interval), z_interval)
    #print("zk", len(z_k), z_k)
    for i in range (len(minimal_zk)):
        if  minimal_interval[i][0]<= minimal_zk[i] <= minimal_interval[i][1]:
            continue
        else:
            print("err", minimal_zk[i], minimal_interval[i])
    #print(new_z_interval)

    selective_p_value = util.compute_p_value(new_z_interval, etaTX[0][0], etaT_Sigma_eta[0][0])
    #print(z, new_z_interval)
    if selective_p_value is None:
        print("None")
        return None
    else:
        return selective_p_value

if __name__ == "__main__":
    max_iteration = 500
    Alpha = 0.05
    count = 0
    n = 50
    K = 3
    list_p_value = []
    underalpha = 0
    for i in range(max_iteration):
        if i % 10 == 0:
            print(i)
        p_value = parametric(n, K)
        if p_value:
            list_p_value.append(p_value)
            count += 1
            if p_value <= Alpha:
                underalpha += 1
    
    # Tính và in False positive rate
    print('\nFalse positive rate:', underalpha/count, count)

    # Kiểm định thống kê
    print(stats.kstest(list_p_value, stats.uniform(loc=0.0, scale=1.0).cdf))

    # Hiển thị histogram
    plt.hist(list_p_value, bins=20)

    plt.show()
