import numpy as np
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import gendata
from models.wdgrl_ae import WDGRL

def train_model(
        Xs, 
        Xt, 
        config_file: str = "config.yaml"):
    # ==== Scaling ====
    X_train_all = np.vstack([Xs, Xt])
    scaler = StandardScaler().fit(X_train_all)
    Xs = scaler.transform(Xs)
    Xt = scaler.transform(Xt)

    # ==== Torch datasets ====
    xs = torch.from_numpy(Xs).float()
    xt = torch.from_numpy(Xt).float()

    source_dataset = TensorDataset(xs)
    target_dataset = TensorDataset(xt)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    exp_cfg = config["experiment"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]



    # ==== Generate data ====
    seed = exp_cfg["seed"]

    ns = Xs.shape[0]
    nt = Xt.shape[0]


    n_clusters = exp_cfg["n_clusters"]

    # ==== WDGRL model ====
    final_model = WDGRL(
        input_dim=data_cfg["n_features"],
        encoder_hidden_dims=model_cfg["encoder_hidden_dims"],
        decoder_hidden_dims=model_cfg["decoder_hidden_dims"],
        critic_hidden_dims=model_cfg["critic_hidden_dims"],
        alpha1=model_cfg["alpha1"],
        alpha2=model_cfg["alpha2"],
        seed=exp_cfg["model_random_state"],
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
        check_ari=False,
    )

    # ==== Save logs ====
    total_loss = log_loss["loss"]
    reconstructionloss = log_loss["decoder_loss"]
    log_metric = log_loss["log_ari"]

    return final_model

if __name__ == "__main__":
    ns, nt, d = 2000, 100, 10

    mu_s = np.array([3]*d).reshape(-1)
    mu_t = np.array([0]*d).reshape(-1)
    Xs = gendata.sample_normal_data(mu=mu_s, sigma=1)
    Xt = gendata.sample_normal_data(mu=mu_t, sigma=1)

    wdgrl = train_model(Xs, Xt)

    

