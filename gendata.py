from sklearn.datasets import make_blobs
from typing import Tuple, Optional, Union, Dict,Literal, List
import numpy as np 
def gen_domain_adaptation_data(
    n_source: int,
    n_target: int, 

    n_features: int,
    domain_shift_type: Literal["covariate", "label", "concept", "mixed"],
    n_clusters: int = 3,
    shift_magnitude: float = 2.0,
    noise_ratio: float = 0.3,
    cluster_std_source: Union[float, list] = 1.0,
    cluster_std_target: Union[float, list] = 1.2,
    base_center=(-3,3),
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Generate source and target domain data for domain adaptation tasks.
    
    Parameters:
    -----------
    n_source : int
        Number of samples in source domain
    n_target : int
        Number of samples in target domain
    n_features : int, default=16
        Number of features (dimensions)
    n_clusters : int, default=3
        Number of clusters/classes
    domain_shift_type : str, default="covariate"
        Type of domain shift:
        - "covariate": Shift in feature distribution
        - "label": Different label distributions
        - "concept": Same features, different decision boundaries
        - "mixed": Combination of shifts
    shift_magnitude : float, default=2.0
        Magnitude of the domain shift
    noise_ratio : float, default=0.3
        Proportion of noise/outliers in target domain
    cluster_std_source : float or list, default=1.0
        Standard deviation for source domain clusters
    cluster_std_target : float or list, default=1.2
        Standard deviation for target domain clusters
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X_source : ndarray, shape (n_source, n_features)
        Source domain data
    y_source : ndarray, shape (n_source,)
        Source domain labels
    X_target : ndarray, shape (n_target, n_features)
        Target domain data
    y_target : ndarray, shape (n_target,)
        Target domain labels
    info : dict
        Information about the generated data including centers and shift details
    """
    
    if random_state is not None:
        seed = random_state
        np.random.seed(seed)
        print("Fixed andom seed =",seed)

    # Generate base cluster centers
    base_centers = np.random.uniform(base_center[0], base_center[1], (n_clusters, n_features))
    
    # Create domain-specific modifications
    if domain_shift_type == "covariate":
        # Shift cluster centers for target domain
        shift_direction = np.random.randn(n_clusters, n_features)
        shift_direction = shift_direction / np.linalg.norm(shift_direction, axis=1, keepdims=True)
        target_centers = base_centers + shift_magnitude #* shift_direction
        
        # Generate source domain
        X_source, y_source = make_blobs(
            n_samples=n_source,
            centers=base_centers,
            n_features=n_features,
            cluster_std=cluster_std_source,
            random_state=random_state
        )
        
        # Generate target domain with shifted centers
        X_target, y_target = make_blobs(
            n_samples=int(n_target * (1 - noise_ratio)),
            centers=target_centers,
            n_features=n_features,
            cluster_std=cluster_std_target,
            random_state=random_state + 1 if random_state else None
        )
        
    elif domain_shift_type == "label":
        # Same feature distribution, different label proportions
        source_centers = base_centers
        target_centers = base_centers
        
        # Different class proportions
        source_sizes = np.random.multinomial(n_source, [1/n_clusters] * n_clusters)
        target_weights = np.random.dirichlet([0.5] * n_clusters)  # More imbalanced
        target_sizes = np.random.multinomial(int(n_target * (1 - noise_ratio)), target_weights)
        
        # Generate source with balanced classes
        X_source, y_source = make_blobs(
            n_samples=source_sizes,
            centers=source_centers,
            n_features=n_features,
            cluster_std=cluster_std_source,
            random_state=random_state
        )
        
        # Generate target with imbalanced classes
        X_target, y_target = make_blobs(
            n_samples=target_sizes,
            centers=target_centers,
            n_features=n_features,
            cluster_std=cluster_std_target,
            random_state=random_state + 1 if random_state else None
        )
        
    elif domain_shift_type == "concept":
        # Same clusters, but rotated decision boundaries
        rotation_matrix = generate_rotation_matrix(n_features, shift_magnitude * 0.3)
        target_centers = base_centers @ rotation_matrix.T
        
        X_source, y_source = make_blobs(
            n_samples=n_source,
            centers=base_centers,
            n_features=n_features,
            cluster_std=cluster_std_source,
            random_state=random_state
        )
        
        X_target, y_target = make_blobs(
            n_samples=int(n_target * (1 - noise_ratio)),
            centers=target_centers,
            n_features=n_features,
            cluster_std=cluster_std_target,
            random_state=random_state + 1 if random_state else None
        )
        
    elif domain_shift_type == "mixed":
        # Combination of covariate and concept drift
        # Shift centers
        shift_direction = np.random.randn(n_clusters, n_features)
        shift_direction = shift_direction / np.linalg.norm(shift_direction, axis=1, keepdims=True)
        shifted_centers = base_centers + shift_magnitude * 0.7 * shift_direction
        
        # Apply rotation
        rotation_matrix = generate_rotation_matrix(n_features, shift_magnitude * 0.2)
        target_centers = shifted_centers @ rotation_matrix.T
        
        X_source, y_source = make_blobs(
            n_samples=n_source,
            centers=base_centers,
            n_features=n_features,
            cluster_std=cluster_std_source,
            random_state=random_state
        )
        
        X_target, y_target = make_blobs(
            n_samples=int(n_target * (1 - noise_ratio)),
            centers=target_centers,
            n_features=n_features,
            cluster_std=cluster_std_target,
            random_state=random_state + 1 if random_state else None
        )
    
    # Add noise to target domain
    if noise_ratio > 0:
        n_noise = n_target - len(X_target)
        if n_noise > 0:
            # Generate noise samples
            noise_samples = np.random.uniform(
                X_target.min() - 2, X_target.max() + 2, 
                (n_noise, n_features)
            )
            noise_labels = np.random.choice(n_clusters, n_noise)
            
            X_target = np.vstack([X_target, noise_samples])
            y_target = np.hstack([y_target, noise_labels])
    
    # Store information about the generation process
    info = {
        'source_centers': base_centers,
        'target_centers': target_centers if 'target_centers' in locals() else base_centers,
        'domain_shift_type': domain_shift_type,
        'shift_magnitude': shift_magnitude,
        'noise_ratio': noise_ratio,
        'n_clusters': n_clusters,
        'source_class_counts': np.bincount(y_source),
        'target_class_counts': np.bincount(y_target)
    }
    
    return X_source, y_source, X_target, y_target, info

def generate_rotation_matrix(n_features: int, angle_factor: float = 0.3) -> np.ndarray:
    """Generate a random rotation matrix for concept drift."""
    # Generate random rotation in high dimensions
    A = np.random.randn(n_features, n_features)
    Q, R = np.linalg.qr(A)
    
    # Scale the rotation by angle_factor
    return angle_factor * Q + (1 - angle_factor) * np.eye(n_features)

def random_points_distance_k(d:int, k:float, seed = None):
    # First point
    if seed is not None:
        np.random.seed(seed)
    p1 = np.random.randn(d)  # can also be uniform in a bounded region
    # Random direction
    direction = np.random.randn(d)
    direction /= np.linalg.norm(direction)
    # Second point at distance k
    p2 = p1 + k * direction
    return [p1, p2]













import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
def visualby_tsne(X_source, X_target, y_source=None, y_target=None, info=None, 
                                    perplexity=30, random_state=42):
    """
    Visualize domain adaptation data using t-SNE for dimensionality reduction.
    """
    ns = len(X_source)
    nt = len(X_target)

    if y_source is None:
        y_source = np.zeros(ns)
        
    if y_target is None:
        y_target = np.zeros(nt)

    if len(X_source) > 1000:
        idx = np.random.choice(len(X_source), 1000, replace=False)
        X_source = X_source[idx]
        y_source = y_source[idx]
    if len(X_target) > 1000:
        idx = np.random.choice(len(X_target), 1000, replace=False)
        X_target = X_target[idx]
        y_target = y_target[idx]

    # Combine data for t-SNE (fit on combined data for better comparison)
    X_combined = np.vstack([X_source, X_target])
    
    # Apply t-SNE to reduce to 2D for visualization
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                init='random', learning_rate=200, max_iter=1000)
    X_combined_2d = tsne.fit_transform(X_combined)
    
    # Split back to source and target
    X_source_2d = X_combined_2d[:len(X_source)]
    X_target_2d = X_combined_2d[len(X_source):]
    
    # Create subplot figure
    plt.figure(figsize=(15, 6))
    
    # Define consistent colors for each class
    n_classes = max(len(np.unique(y_source)), len(np.unique(y_target)))
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    # Plot source domain (left subplot)
    plt.subplot(1, 2, 1)
    for i, label in enumerate(np.unique(y_source)):
        mask = y_source == label
        plt.scatter(X_source_2d[mask, 0], X_source_2d[mask, 1], 
                   c=[colors[i]], label=f'Class {label}', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    plt.title(f'Source Domain\n({len(X_source)} samples)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # Add source domain statistics
    # plt.text(0.02, 0.98, f'Classes: {len(np.unique(y_source))}\nSamples per class:\n' + 
    #          '\n'.join([f'  Class {i}: {np.sum(y_source == i)}' for i in np.unique(y_source)]),
    #          transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot target domain (right subplot)
    plt.subplot(1, 2, 2)
    for i, label in enumerate(np.unique(y_target)):
        mask = y_target == label
        plt.scatter(X_target_2d[mask, 0], X_target_2d[mask, 1], 
                   c=[colors[i]], label=f'Class {label}', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    plt.title(f'Target Domain\n({len(X_target)} samples)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # Add target domain statistics
    # plt.text(0.02, 0.98, f'Classes: {len(np.unique(y_target))}\nSamples per class:\n' + 
    #          '\n'.join([f'  Class {i}: {np.sum(y_target == i)}' for i in np.unique(y_target)]),
    #          transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Add overall title with domain shift information
    if info != None:
        plt.suptitle(f'Domain Adaptation Visualization - {info["domain_shift_type"].title()} Shift\n' +
                    f'Shift Magnitude: {info["shift_magnitude"]}, Noise Ratio: {info["noise_ratio"]}, Features: {X_source.shape[1]}',
                    fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

from sklearn.cluster import KMeans
def visualby_tsne_kmean(X_source, X_target, kmean: KMeans, y_source=None, y_target=None, info=None,
                        perplexity=30, random_state=42):
    """
    Visualize domain adaptation data using t-SNE for dimensionality reduction,
    with background showing KMeans predicted clusters and points colored by ground truth.
    """
    ns = len(X_source)
    nt = len(X_target)

    if y_source is None:
        y_source = np.zeros(ns)
        
    if y_target is None:
        y_target = np.zeros(nt)

    # Limit size for plotting
    if len(X_source) > 1000:
        idx = np.random.choice(len(X_source), 1000, replace=False)
        X_source = X_source[idx]
        y_source = y_source[idx]
    if len(X_target) > 1000:
        idx = np.random.choice(len(X_target), 1000, replace=False)
        X_target = X_target[idx]
        y_target = y_target[idx]

    # Combine data for t-SNE
    X_combined = np.vstack([X_source, X_target])
    
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                init='random', learning_rate=200, max_iter=1000)
    X_combined_2d = tsne.fit_transform(X_combined)
    
    # Split back
    X_source_2d = X_combined_2d[:len(X_source)]
    X_target_2d = X_combined_2d[len(X_source):]
    
    # Predict clusters for t-SNE space
    kmean_source_pred = kmean.fit_predict(X_source_2d)
    kmean_target_pred = kmean.fit_predict(X_target_2d)

    # Create figure
    plt.figure(figsize=(15, 6))
    
    # Colors for ground truth points
    n_classes = max(len(np.unique(y_source)), len(np.unique(y_target)))
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

    # === Source subplot ===
    plt.subplot(1, 2, 1)
    # Background mesh for predicted clusters
    h = 0.02
    x_min, x_max = X_source_2d[:, 0].min() - 1, X_source_2d[:, 0].max() + 1
    y_min, y_max = X_source_2d[:, 1].min() - 1, X_source_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = kmean.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')

    # Points = ground truth
    for i, label in enumerate(np.unique(y_source)):
        mask = y_source == label
        plt.scatter(X_source_2d[mask, 0], X_source_2d[mask, 1], 
                   c=[colors[i]], label=f'Class {label}', alpha=0.7, s=60,
                   edgecolors='black', linewidth=0.5)
    plt.title(f'Source Domain\n({len(X_source)} samples)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)

    # === Target subplot ===
    plt.subplot(1, 2, 2)
    x_min, x_max = X_target_2d[:, 0].min() - 1, X_target_2d[:, 0].max() + 1
    y_min, y_max = X_target_2d[:, 1].min() - 1, X_target_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = kmean.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')

    for i, label in enumerate(np.unique(y_target)):
        mask = y_target == label
        plt.scatter(X_target_2d[mask, 0], X_target_2d[mask, 1], 
                   c=[colors[i]], label=f'Class {label}', alpha=0.7, s=60,
                   edgecolors='black', linewidth=0.5)
    plt.title(f'Target Domain\n({len(X_target)} samples)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)

    # Overall title
    if info is not None:
        plt.suptitle(f'Domain Adaptation Visualization - {info["domain_shift_type"].title()} Shift\n' +
                     f'Shift Magnitude: {info["shift_magnitude"]}, Noise Ratio: {info["noise_ratio"]}, Features: {X_source.shape[1]}',
                     fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
from sklearn.ensemble import IsolationForest
def remove_outliers(X, y, contamination=0.01, random_state=None):
    """Xóa outlier bằng IsolationForest"""
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    yhat = iso.fit_predict(X)  # 1 = inlier, -1 = outlier
    mask = yhat == 1
    return X[mask], y[mask]

def gen_domain_adaptation_data2(
    ns: int,
    nt: int,
    n_features:int,
    dist:float=1,
    shift:float = 0,
    std_source: Union[float, List[float]] = 1.0,
    std_target: Union[float, List[float]] = 1.0,
    contamination: float = 0.02,
    random_state=None,
    ):
    
    centers_s = random_points_distance_k(n_features, dist,seed=random_state)

    X_source, y_source, center_source = make_blobs(
        n_samples=ns, 
        n_features=n_features,
        centers=centers_s,
        cluster_std=std_source,
        random_state=random_state,
        return_centers=True,
    )
    
    centers_t = center_source + shift

    X_target, y_target, center_target = make_blobs(
        n_samples=nt, 
        n_features=n_features,
        centers=centers_t,
        cluster_std=std_target,
        random_state=random_state,
        return_centers=True,
    )

    # Xóa outlier 1%
    # X_source, y_source = remove_outliers(X_source, y_source, contamination, random_state)
    # X_target, y_target = remove_outliers(X_target, y_target, contamination, random_state)

    return {
        "source": (X_source, y_source, center_source),
        "target": (X_target, y_target, center_target)
    }


def random_points_distance_k2(d: int, k: int, base_dist: float, stds, seed=None):
    """
    Generate k cluster centers in d-dimensional space.
    Ensures that larger-std clusters are further apart to reduce overlap.
    """
    if seed is not None:
        np.random.seed(seed)

    centers = []
    # First center
    p = np.random.randn(d)
    centers.append(p)

    for i in range(1, k):
        direction = np.random.randn(d)
        direction /= np.linalg.norm(direction)

        # minimum required distance based on std
        min_dist = 1.2 * (stds[i] + np.mean(stds[:i]))  # 2.5 = separation factor
        dist = base_dist * np.random.uniform(0.5, 1.5) + min_dist

        new_center = centers[0] + dist * direction
        centers.append(new_center)

    return np.array(centers)


def gen_domain_adaptation_data_k(
    ns: int,
    nt: int,
    n_features: int,
    n_clusters: int,
    dist: float = 1,
    shift: float = 0,
    std_source: Union[float, List[float]] = 1.0,
    std_target: Union[float, List[float]] = 1.0,
    contamination: float = 0.02,
    random_state=None,
):
    # handle case: scalar std -> replicate for each cluster
    if isinstance(std_source, (int, float)):
        std_source = [std_source] * n_clusters
    if isinstance(std_target, (int, float)):
        std_target = [std_target] * n_clusters

    # Generate k centers for source
    centers_s = random_points_distance_k2(
        n_features, n_clusters, dist, stds=std_source, seed=random_state
    )

    X_source, y_source, center_source = make_blobs(
        n_samples=ns,
        n_features=n_features,
        centers=centers_s,
        cluster_std=std_source,
        random_state=random_state,
        return_centers=True,
    )

    # Shift target centers
    centers_t = center_source + shift

    X_target, y_target, center_target = make_blobs(
        n_samples=nt,
        n_features=n_features,
        centers=centers_t,
        cluster_std=std_target,
        random_state=random_state,
        return_centers=True,
    )

    return {
        "source": (X_source, y_source, center_source),
        "target": (X_target, y_target, center_target),
    }

from sklearn.cluster import KMeans

def clustering(
        X, 
        n_cluster: int = 2):
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels, kmeans
def plot_clusters_with_background(X, labels, ax, title, kmeans):
    # Mesh grid for background
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))



    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot background
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k')
    ax.set_title(title)
    return scatter
from sklearn.metrics import adjusted_rand_score
from tqdm import trange


if __name__ == "__main__":
    fine = 0
    notfine = 0
    for _ in trange(1):
        ns = 1000
        nt = 40
        dataset = gen_domain_adaptation_data2(
            ns = ns, 
            nt = nt, 
            n_features= 2, 
            dist=3,
            std_source=[1,1.5],
            std_target=[1,2],
            shift=0,
            random_state=None
            )
        Xs, ys, cen_s = dataset["source"]
        Xt, yt, cen_t = dataset["target"]

        
        cluster_target,kmeansT = clustering(Xt, 2)
        X_comb = np.vstack((Xs, Xt))
        cluster_comb, kmeanC = clustering(X_comb, 2)
        # print(X_comb.shape, np.hstack((ys,yt)).shape)


        ariT = adjusted_rand_score(yt, cluster_target)
        ariC = adjusted_rand_score(yt, cluster_comb[ns:])
        # print(f"Target only: {ariT}")
        # print(f"Combined: {ariC}")


        if ariT < ariC:
            fine+=1
        else:
            notfine+=1
        # Plot both clustering results
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        plot_clusters_with_background(Xt, yt, axes[0], kmeans=kmeansT,title= "Target Clustering")
        plot_clusters_with_background(X_comb, np.hstack((ys,yt)), axes[1],kmeans=kmeanC, title="Combined Clustering")

        plt.tight_layout()
        plt.show()

    print("Fine: ", fine)
    print("not fine: ", notfine)
        
    # # Plot both clustering results
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # plot_clusters_with_background(Xt, yt, axes[0], kmeans=kmeansT,title= "Target Clustering")
    # plot_clusters_with_background(X_comb, np.hstack((ys,yt)), axes[1],kmeans=kmeanC, title="Combined Clustering")

    # plt.tight_layout()
    # plt.show()




