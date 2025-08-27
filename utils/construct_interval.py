import torch
import numpy as np
import util
# import time
import solveinequalities.interval as bst_solving_eq

def ReLUcondition(model, a, b, X):
    # p = int(X.shape[0] / 2)
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

