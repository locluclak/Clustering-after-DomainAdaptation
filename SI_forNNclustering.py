import numpy as np
import torch

import gendata
import models.wdgrl_ae



if __name__ == "__main__":
    ns, nt, d = 2000, 100, 10

    mu_s = np.array([3]*d)
    mu_t = np.array([0]*d)
    Xs = gendata.sample_normal_data(mu=mu_s, sigma=1)
    Xt = gendata.sample_normal_data(mu=mu_t, sigma=1)

    