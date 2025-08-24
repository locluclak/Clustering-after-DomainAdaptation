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

import gendata
from models.wdgrl_ae import WDGRL


def clustering(X, n_cluster: int):
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels, kmeans


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
    ns, nt, d = data_cfg["ns"], data_cfg["nt"], data_cfg["n_features"]
    n_clusters = exp_cfg["n_clusters"]

    print("Using seed =", seed)
    print("Data config ", data_cfg)
    # Select data generation method
    data_gen_method = data_cfg.get("data_gen_method", "gen_domain_adaptation_data2")
    if data_gen_method == "gen_domain_adaptation_data_k":
        dataset = gendata.gen_domain_adaptation_data_k(
            ns=ns,
            nt=nt,
            n_features=d,
            n_clusters=n_clusters,
            dist=data_cfg["dist"],
            std_source=data_cfg["std_source"],
            std_target=data_cfg["std_target"],
            shift=data_cfg["shift"],
            random_state=seed,
        )
    else:
        dataset = gendata.gen_domain_adaptation_data2(
            ns=ns,
            nt=nt,
            n_features=d,
            dist=data_cfg["dist"],
            std_source=data_cfg["std_source"],
            std_target=data_cfg["std_target"],
            shift=data_cfg["shift"],
            random_state=seed,
        )
    Xs, Ys, _ = dataset["source"]
    Xt, Yt, _ = dataset["target"]

    ns = Xs.shape[0]
    nt = Xt.shape[0]

    # ==== Scaling ====
    X_train_all = np.vstack([Xs, Xt])
    scaler = StandardScaler().fit(X_train_all)
    Xs = scaler.transform(Xs)
    Xt = scaler.transform(Xt)

    # ==== Original clustering baseline ====
    cluster_labels, _ = clustering(Xt, n_cluster=n_clusters)
    original_ari = adjusted_rand_score(Yt, cluster_labels)
    original_sil = silhouette_score(Xt, cluster_labels)
    print(f"Adjusted Rand Index (ARI) only on target domain: {original_ari:.4f}")

    # ==== Torch datasets ====
    xs = torch.from_numpy(Xs).float()
    ys = torch.from_numpy(Ys).long()
    xt = torch.from_numpy(Xt).float()
    yt = torch.from_numpy(Yt).long()

    source_dataset = TensorDataset(xs)
    target_dataset = TensorDataset(xt)

    # ==== WDGRL model ====
    final_model = WDGRL(
        input_dim=data_cfg["n_features"],
        encoder_hidden_dims=model_cfg["encoder_hidden_dims"],
        decoder_hidden_dims=model_cfg["decoder_hidden_dims"],
        critic_hidden_dims=model_cfg["critic_hidden_dims"],
        alpha1=model_cfg["alpha1"],
        alpha2=model_cfg["alpha2"],
        seed=exp_cfg["model_random_state"],
        reallabel=Yt,
    )

    log_loss = final_model.train(
        source_dataset,
        target_dataset,
        num_epochs=train_cfg["num_epochs"],
        gamma=train_cfg["gamma"],
        delta=train_cfg["delta"],
        lambda_=train_cfg["lambda"],
        dc_iter=train_cfg["dc_iter"],
        batch_size=train_cfg["batch_size"],
        verbose=train_cfg["verbose"],
        early_stopping=train_cfg["early_stopping"],
        patience=train_cfg["patience"],
        min_delta=train_cfg["min_delta"],
        check_ari=exp_cfg["check_ari"],
    )

    # ==== Save logs ====
    total_loss = log_loss["loss"]
    reconstructionloss = log_loss["decoder_loss"]
    log_metric = log_loss["log_ari"]
    # ==== Logging setup ====
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "results.txt")

    np.save(os.path.join(log_dir, "total_loss.npy"), np.array(total_loss))
    np.save(os.path.join(log_dir, "reconstruction_loss.npy"), np.array(reconstructionloss))
    np.save(os.path.join(log_dir, "log_metric.npy"), np.array(log_metric, dtype=object))

    # Extract specific metrics
    ari_comb = [d["ari_comb"] for d in log_metric]
    silhouette_comb = [d["silhouette_comb"] for d in log_metric]
    ari_Tonly = [d["ari_Tonly"] for d in log_metric]
    sil_Tonly = [d["sil_Tonly"] for d in log_metric]

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

    # Silhouette
    plt.figure(figsize=(14, 6))
    plt.plot(epochs, silhouette_comb, linestyle='-', color='wheat', label="Combine S&T")
    plt.plot(epochs, [original_sil] * len(epochs), linestyle='-', color='green', label="Original")
    plt.plot(epochs, sil_Tonly, linestyle='-', color='plum', label="Transfered T")
    plt.title("Silhouette over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Silhouette")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(log_dir, "silhouette.png"))
    plt.close()

    # ARI
    plt.figure(figsize=(14, 6))
    plt.plot(epochs, ari_comb, linestyle='-', color='y', label="Combine S&T")
    plt.plot(epochs, ari_Tonly, linestyle='-', color='m', label="Transfered T")
    plt.plot(epochs, [original_ari] * len(epochs), linestyle='-', color='green', label="Original")
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
    comb_cluster_labels, _ = clustering(x_comb, 2)
    ari = adjusted_rand_score(Yt, comb_cluster_labels[ns:])
    print(f"Adjusted Rand Index (ARI) of target on transported domain: {ari:.4f}")
    clusterT, _ = clustering(xt_hat, 2)
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
