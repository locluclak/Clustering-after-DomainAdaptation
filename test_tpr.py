
import numpy as np 

def random_3_points(dim=2, delta: float=1):
    if dim < 2:
        raise ValueError("Require at least 2 dimensions")
    p1 = np.zeros(dim)
    p2 = np.zeros(dim)
    p3 = np.zeros(dim)

    p2[1] = delta
    p3[0] = delta/2
    p3[0] = np.sqrt(3)/2 *delta 

    points = np.array([p1,p2,p3])
    return points

def random_3_clusters(ns=10,nt=5, dim=5, delta: float = 1,cluster_std=[1,1,1], seed=None):
    rng = np.random.default_rng(seed)
    shift = 2

    centers_t = random_3_points(dim=dim, delta=delta)
    centers_s = centers_t + shift
    print(centers_t)
    print(centers_s)
    Xs = []
    ys = []
    mus = []
    for i, center in enumerate(centers_s):
        cluster_points = rng.normal(loc=center, scale=cluster_std[i], size=(ns, dim))
        Xs.append(cluster_points)
        ys.append(np.full(ns, i))
        for _ in range(ns):
            mus.append(center.copy())
    mus = np.array(mus)
   
    Xs = np.vstack(Xs)
    ys = np.concatenate(ys)

    Xt = []
    yt = []
    mut = []
    for i, center in enumerate(centers_t):
        cluster_points = rng.normal(loc=center, scale=cluster_std[i], size=(nt, dim))
        Xt.append(cluster_points)
        yt.append(np.full(nt, i))
        for _ in range(nt):
            mut.append(center.copy())
    mut = np.array(mut)

    Xt = np.vstack(Xt)
    yt = np.concatenate(yt)

    # Shuffle source
    idx_s = rng.permutation(len(Xs))
    Xs = Xs[idx_s]
    ys = ys[idx_s]
    mus = mus[idx_s]
    # Shuffle target
    idx_t = rng.permutation(len(Xt))
    Xt = Xt[idx_t]
    yt = yt[idx_t]
    mut = mut[idx_t]

    return Xs, Xt, ys, yt, mus, mut

Xs, Xt, ys, yt, mus, mut = random_3_clusters()
print(len(Xt))
print(len(yt))
print(mut)