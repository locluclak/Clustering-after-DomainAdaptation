import os
import time
import yaml
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import gendata
from models.wdgrl import WDGRL
from dataset import datasets

def clustering(X, n_cluster: int):
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels, kmeans

def digits_dataset():
    USPS = datasets.getUSPS()
    MNIST = datasets.getMNIST()

    X_source_train = MNIST["Xtrain"]
    y_source_train = MNIST["ytrain"]

    X_target_train = USPS["Xtrain"]
    y_target_train = USPS["ytrain"]

    selected_labels = [0, 6, 3, 8, 9,2]

    target_mask = np.isin(y_target_train, selected_labels)
    source_mask = np.isin(y_source_train, selected_labels)

    X_target_train = X_target_train[target_mask]
    y_target_train = np.asarray(y_target_train[target_mask])
    # print(y_target_train.shape)

    X_source_train = X_source_train[source_mask]
    y_source_train = np.asarray(y_source_train[source_mask])

    scaler = MinMaxScaler()
    X_target_train_norm = scaler.fit_transform(X_target_train)
    X_source_train_norm = scaler.fit_transform(X_source_train)
    X_target_train_norm += np.random.normal(0, 0.01, X_target_train_norm.shape)
    X_source_train_norm += np.random.normal(0, 0.01, X_source_train_norm.shape)
    return X_source_train_norm, X_target_train_norm, y_source_train, y_target_train
def main():
    # ==== Load config ====
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    exp_cfg = config["experiment"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]



    # ==== Generate data ====
    seed = exp_cfg["seed"]
    n_clusters = 6

    print("Using seed =", seed)
    print("Data config ", data_cfg)

    Xs, Xt, Ys, Yt = digits_dataset()
    ns, nt, d = Xs.shape[0], Xt.shape[0], Xs.shape[1]

    # ==== Original clustering baseline ====
    cluster_labels, _ = clustering(Xt, n_cluster=n_clusters)
    original_ari = adjusted_rand_score(Yt, cluster_labels)
    original_sil = silhouette_score(Xt, cluster_labels)
    print(f"Adjusted Rand Index (ARI) only on target domain: {original_ari:.4f}")

    # ==== Torch datasets ====
    xs = torch.from_numpy(Xs).double()
    # ys = torch.from_numpy(Ys).long()
    xt = torch.from_numpy(Xt).double()
    # yt = torch.from_numpy(Yt).long()

    source_dataset = TensorDataset(xs)
    target_dataset = TensorDataset(xt)

    # ==== WDGRL model ====
    final_model = WDGRL(
        input_dim=256,
        encoder_hidden_dims=[200,100],
        # decoder_hidden_dims=model_cfg["decoder_hidden_dims"],
        critic_hidden_dims=[100],
        alpha1=0.0002,
        alpha2=0.0002,
        seed=exp_cfg["model_random_state"],
        reallabel=Yt,
        n_clusters = n_clusters
    )

    log_loss = final_model.train(
        source_dataset,
        target_dataset,
        num_epochs=8000,
        gamma=10,
        dc_iter=train_cfg["dc_iter"],
        batch_size=train_cfg["batch_size"],
        early_stopping=False,
        check_ari=True,
    )

    # ==== Save logs ====
    total_loss = log_loss["loss"]
    # reconstructionloss = log_loss["decoder_loss"]
    log_metric = log_loss["log_ari"]
    # ==== Logging setup ====
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "results.txt")

    np.save(os.path.join(log_dir, "total_loss.npy"), np.array(total_loss))
    # np.save(os.path.join(log_dir, "reconstruction_loss.npy"), np.array(reconstructionloss))
    np.save(os.path.join(log_dir, "log_metric.npy"), np.array(log_metric, dtype=object))

    # Extract specific metrics
    epoch_log = [d["epoch"] for d in log_metric]
    ari_comb = [d["ari_comb"] for d in log_metric]
    # silhouette_comb = [d["silhouette_comb"] for d in log_metric]
    # ari_Tonly = [d["ari_Tonly"] for d in log_metric]
    # sil_Tonly = [d["sil_Tonly"] for d in log_metric]

    # ==== Plot & Save Figures ====
    epochs = range(1, len(total_loss) + 1)

    # Loss
    plt.figure(figsize=(14, 6))
    plt.plot(epochs, total_loss, linestyle='-', color='blue')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.ylim(0, 1.0)
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "loss.png"))
    plt.close()

    # # Silhouette
    # plt.figure(figsize=(14, 6))
    # plt.plot(epochs, silhouette_comb, linestyle='-', color='wheat', label="Combine S&T")
    # plt.plot(epochs, [original_sil] * len(epochs), linestyle='-', color='green', label="Original")
    # plt.plot(epochs, sil_Tonly, linestyle='-', color='plum', label="Transfered T")
    # plt.title("Silhouette over Epochs")
    # plt.xlabel("Epoch")
    # plt.ylabel("Silhouette")
    # plt.ylim(0, 1.0)
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(os.path.join(log_dir, "silhouette.png"))
    # plt.close()

    # ARI
    plt.figure(figsize=(14, 6))
    plt.plot(epoch_log, ari_comb, linestyle='-', color='y', label="Combine S&T")
    # plt.plot(epoch_log, ari_Tonly, linestyle='-', color='m', label="Transfered T")
    plt.plot(epoch_log, [original_ari] * len(epoch_log), linestyle='-', color='green', label="Original")
    plt.title("ARI over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("ARI")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(log_dir, "ari.png"))
    plt.close()

    # ==== Final evaluation ====
    xs_hat = final_model.extract_feature(xs.cuda())
    xt_hat = final_model.extract_feature(xt.cuda())
    xs_hat = xs_hat.cpu().numpy()
    xt_hat = xt_hat.cpu().numpy()

    x_comb = np.vstack((xs_hat, xt_hat))
    comb_cluster_labels, _ = clustering(x_comb, n_clusters)
    ari = adjusted_rand_score(Yt, comb_cluster_labels[ns:])
    print(f"Adjusted Rand Index (ARI) of target on transported domain: {ari:.4f}")
    clusterT, _ = clustering(xt_hat, n_clusters)
    ariT = adjusted_rand_score(Yt, clusterT)
    # Save summary to txt
    with open(log_file, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)  # lưu luôn config
        f.write("\n\n")
        f.write(f"Original ARI: {original_ari:.4f}\n")
        f.write(f"Final ARI (transported): {ari:.4f}\n")
        f.write(f"Final ARI (transported, target only): {ariT:.4f}\n")
        f.write("Training finished successfully.\n")


if __name__ == "__main__":
    main()
