import numpy as np
import utils.util as util
import solveinequalities.interval as bst_solving_eq

def ReLUcondition(model, a, b, X):
    layers = []
    for name, param in model.named_children():
        temp = dict(param._modules)
        
        for layer_name in temp.values():
            if ('Linear' in str(layer_name)):
                layers.append('Linear')
            elif ('ReLU' in str(layer_name)):
                layers.append('ReLU')
    ptr = 0 

    itv = [(-np.inf, np.inf)]
    weight = None
    bias = None
    for name, param in model.named_parameters():
        if (layers[ptr] == 'Linear'):
            if ('weight' in name):
                weight = np.asarray(param.data.cpu())
            elif ('bias' in name):
                bias = np.asarray(param.data.cpu()).reshape(-1, 1)
                bias = bias.dot(np.ones((1, X.shape[0]))).T
                ptr += 1
                X = X.dot(weight.T) + bias
                a = a.dot(weight.T) + bias
                b = b.dot(weight.T)
        # t2 = time.time()
        if (ptr < len(layers) and layers[ptr] == 'ReLU'):
            ptr += 1

            sign_X = np.sign(X)
            at = (a * -1*sign_X).flatten()
            bt = (b * -1*sign_X).flatten()
            itv = util.interval_intersection(itv, bst_solving_eq.interval_intersection(bt,-at))

            sign_X[sign_X < 0] = 0
            X = X*sign_X
            a = a*sign_X
            b = b*sign_X

            # sub_itv = [(-np.inf, np.inf)]

            # for i in range(X.shape[0]):
            #     for j in range(X.shape[1]):
            #         if X[i][j] > 0:
            #             sub_itv = util.interval_intersection(
            #                 sub_itv, 
            #                 util.solve_quadratic_inequality(a=0, b=-b[i][j], c=-a[i][j])
            #                 )
            #         else:
            #             sub_itv = util.interval_intersection(
            #                 sub_itv, 
            #                 util.solve_quadratic_inequality(a=0, b=b[i][j], c = a[i][j])
            #                 )

            #             X[i][j] = 0
            #             a[i][j] = 0
            #             b[i][j] = 0
            
            # itv = util.interval_intersection(itv, sub_itv)

    return itv, a, b



def KMeancondition(n, K, a, b, initial_centroids, labels_all, members_all):
    trunc_interval1 = [(-np.inf, np.inf)]
    a = np.asarray(a)
    b = np.asarray(b)

    # Precompute initial centroids' indices for faster access
    initial_centroids_labels = labels_all[0]

    for i in range(n):
        current_label = initial_centroids_labels[i]
        if i == initial_centroids[current_label]:
            continue

        u1 = (a[i] - a[initial_centroids[current_label]]).reshape(-1, 1)
        v1 = (b[i] - b[initial_centroids[current_label]]).reshape(-1, 1)
        p1, q1, o1 = util.construct_p_q_t(u1, v1)

        for k in range(K):
            if k == current_label:
                continue

            u2 = (a[i] - a[initial_centroids[k]]).reshape(-1, 1)
            v2 = (b[i] - b[initial_centroids[k]]).reshape(-1, 1)
            p2, q2, o2 = util.construct_p_q_t(u2, v2)

            p, q, o = (p1 - p2).item(), (q1 - q2).item(), (o1 - o2).item()
            res = util.solve_quadratic_inequality(p, q, o)
            trunc_interval1 = util.interval_intersection(trunc_interval1, res)
    # print("Initial K-means truncation interval:", trunc_interval1)
    trunc_interval2 = [(-np.inf, np.inf)]

    for t in range(1, len(labels_all)):
        for i in range(n):
            e_i = np.zeros((1,n))
            e_i[0][i] = 1
            
            gamma_i = np.zeros((1,n))
            label_i = labels_all[t][i] 
            
            C_i_t_minus = list(members_all[t-1][label_i])  
            if len(C_i_t_minus) == 0:
                    continue
    
            gamma_i[:,C_i_t_minus] = 1
        
            E_temp_1 = e_i - gamma_i/len(C_i_t_minus)
            u3 = E_temp_1.dot(a).reshape(-1,1)
            v3 = E_temp_1.dot(b).reshape(-1,1)
            p3, q3, o3 = util.construct_p_q_t(u3, v3)
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

                p4, q4, o4 = util.construct_p_q_t(u4, v4)

                p_comma, q_comma, o_comma = (p3 - p4).item(), (q3 - q4).item(), (o3 - o4).item()
                res = util.solve_quadratic_inequality(p_comma, q_comma, o_comma)
                # if t == 4 and i ==22 and k==1:
                    # print(f"{t}, {i}, {k}.p_comma, q_comma, o_comma:", f"{p_comma:.8f}", f"{q_comma:.8f}", f"{o_comma:.8f}")
                    # print("res:", res)
                trunc_interval2 = util.interval_intersection(trunc_interval2, res)
    #     print(f"Iteration {t} K-means truncation interval:", trunc_interval2)
    # print("Subsequent K-means truncation interval:", trunc_interval2)
    trunc_interval = util.interval_intersection(trunc_interval1, trunc_interval2)
    trunc_interval = [(float(a), float(b)) for (a, b) in trunc_interval]
    return trunc_interval


def KMeancondition2(n, K, a, b, initial_centroids, labels_all, members_all):
    trunc_interval = [(-np.inf, np.inf)]
    a = np.asarray(a)
    b = np.asarray(b)

    P,Q,O = [],[],[]
    initial_centroids_labels = labels_all[0]

    for i in range(n):
        current_label = initial_centroids_labels[i]
        if i == initial_centroids[current_label]:
            continue

        u1 = (a[i] - a[initial_centroids[current_label]]).reshape(-1, 1)
        v1 = (b[i] - b[initial_centroids[current_label]]).reshape(-1, 1)
        
        p1 = np.dot(v1.T, v1)
        q1 = np.dot(v1.T, u1) + np.dot(u1.T, v1)
        o1 = np.dot(u1.T, u1)

        for k in range(K):
            if k == current_label:
                continue

            u2 = (a[i] - a[initial_centroids[k]]).reshape(-1, 1)
            v2 = (b[i] - b[initial_centroids[k]]).reshape(-1, 1)
            p2 = np.dot(v2.T, v2)
            q2 = np.dot(v2.T, u2) + np.dot(u2.T, v2)
            o2 = np.dot(u2.T, u2)

            p, q, o = (p1 - p2).item(), (q1 - q2).item(), (o1 - o2).item()
            P.append(p)
            Q.append(q)
            O.append(o)

    for t in range(1, len(labels_all)):
        for i in range(n):
            e_i = np.zeros((1,n))
            e_i[0][i] = 1
            
            gamma_i = np.zeros((1,n))
            label_i = labels_all[t][i] 
            
            C_i_t_minus = list(members_all[t-1][label_i]) 
            if len(C_i_t_minus) == 0:
                    continue
    
            gamma_i[:,C_i_t_minus] = 1
        
            E_temp_1 = e_i - gamma_i/len(C_i_t_minus)
            u3 = E_temp_1.dot(a).reshape(-1,1)
            v3 = E_temp_1.dot(b).reshape(-1,1)

            p3 = np.dot(v3.T, v3)
            q3 = np.dot(v3.T, u3) + np.dot(u3.T, v3)
            o3 = np.dot(u3.T, u3)

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

                p4 = np.dot(v4.T, v4)
                q4 = np.dot(v4.T, u4) + np.dot(u4.T, v4)
                o4 = np.dot(u4.T, u4)

                p_comma, q_comma, o_comma = (p3 - p4).item(), (q3 - q4).item(), (o3 - o4).item()
                P.append(p_comma)
                Q.append(q_comma)
                O.append(o_comma)

    return np.array(P), np.array(Q), np.array(O)

def solveinterval(P, Q, O):
    trunc_interval = [(-np.inf, np.inf)]
    for i in range(len(P)):
        p, q, o = P[i], Q[i], O[i]
        res = util.solve_quadratic_inequality_numba(p, q, o)
        trunc_interval = util.interval_intersection(trunc_interval, res)
    # trunc_interval = [(float(a), float(b)) for (a, b) in trunc_interval]
    return trunc_interval