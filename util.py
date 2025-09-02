import numpy as np
from mpmath import mp
from collections import defaultdict

def construct_p_q_t(a, b, A1):
    p = np.dot(b.T, np.dot(A1, b))
    q = np.dot(b.T, np.dot(A1, a)) + np.dot(a.T, np.dot(A1, b))
    t = np.dot(a.T, np.dot(A1, a))
    return p, q, t

def compute_p_value(intervals, etaT_Y, etaT_Sigma_eta):
    denominator = 0
    numerator = 0

    for i in intervals:
        leftside, rightside = i
        if leftside <= etaT_Y <= rightside:
            numerator = denominator + mp.ncdf(etaT_Y / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))
        denominator += mp.ncdf(rightside / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(leftside / np.sqrt(etaT_Sigma_eta))
    if denominator == 0:
        return None
    cdf = float(numerator / denominator)
    # print(cdf)
    # compute two-sided selective p_value
    return 2 * min(cdf, 1 - cdf)

def kmeans(X, k, max_iters=100, tol=1e-4):
    n = X.shape[0]
    rng = np.random.RandomState(42)  # Create a separate random generator
    initial_indices = rng.choice(n, k, replace=False) 
    initial_centroids = X[initial_indices]
    centroids = initial_centroids.copy()
    cluster_labels_all = []
    cluster_members_all = []

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        cluster_labels_all.append(labels.copy())
        cluster_members = defaultdict(set)
        for idx, label in enumerate(labels):
            cluster_members[label].add(idx)
        cluster_members_all.append(cluster_members)
        new_centroids = np.zeros_like(centroids)
        for cluster in range(k):
            instances_in_cluster = X[labels == cluster]
            if len(instances_in_cluster) > 0:
                new_centroids[cluster] = instances_in_cluster.mean(axis=0)
            else:
                new_centroids[cluster] = centroids[cluster]
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return initial_indices, cluster_labels_all, cluster_members_all


def solve_quadratic_inequality(a, b, c,seed = 0):
    """ ax^2 + bx +c <= 0 """
    if abs(a) < 1e-8:
        a = 0
    if abs(b) < 1e-8:
        b = 0
    if abs(c) < 1e-8:
        c = 0
    if a == 0:
        # print(f"b: {b}")
        if b > 0:
            # return [(-np.inf, -c / b)]
            return [(-np.inf, np.around(-c / b, 8))]
        elif b == 0:
            # print(f"c: {c}")
            if c <= 0:
                return [(-np.inf, np.inf)]
            else:
                print('Error bx + c', seed)
                return 
        else:
            return [(np.around(-c / b, 8), np.inf)]
    delta = b*b - 4*a*c
    if delta < 0:
        if a < 0:
            return [(-np.inf, np.inf)]
        else:
            print("Error to find interval. ")
    # print("delta:", delta)
    # print(f"2a: {2*a}")
    x1 = (- b - np.sqrt(delta)) / (2*a)
    x2 = (- b + np.sqrt(delta)) / (2*a)
    # if x1 > x2:
    #     x1, x2 = x2, x1  
    x1 = np.around(x1, 8)
    x2 = np.around(x2, 8)
    if a < 0:
        return [(-np.inf, x2),(x1, np.inf)]
    return [(x1,x2)]


def interval_intersection(a, b):
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        a_start, a_end = a[i]
        b_start, b_end = b[j]
        
        # Calculate the potential intersection
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        
        # If the interval is valid, add to results
        if start < end:
            result.append((start, end))
        
        # Move the pointer which ends first
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return result
def interval_union(a, b):
    # Merge the two sorted interval lists into one sorted list
    merged = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] < b[j][0]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    # Add any remaining intervals from a or b
    merged.extend(a[i:])
    merged.extend(b[j:])
    
    # Merge overlapping intervals
    if not merged:
        return []
    
    result = [merged[0]]
    for current in merged[1:]:
        last = result[-1]
        if current[0] < last[1]:
            # Overlapping or adjacent, merge them
            new_start = last[0]
            new_end = max(last[1], current[1])
            result[-1] = (new_start, new_end)
        else:
            result.append(current)
    return result

def generate(n):
    u = np.zeros((n, 1))
    noise = np.random.normal(loc=0, scale=1, size=n)
    y = u + noise.reshape(-1, 1)
    return y
