import numpy as np
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import yaml
import time
import os

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# import utils.construct_interval
import gendata
from models.wdgrl import WDGRL

def train_model(
        Xs, 
        Xt, 
        config_file: str = "config.yaml"):
    # # ==== Scaling ====
    # X_train_all = np.vstack([Xs, Xt])
    # scaler = StandardScaler().fit(X_train_all)
    # Xs = scaler.transform(Xs)
    # Xt = scaler.transform(Xt)

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

    d = Xs.shape[1]

    n_clusters = exp_cfg["n_clusters"]

    # ==== WDGRL model ====
    final_model = WDGRL(
        input_dim=d,
        encoder_hidden_dims=model_cfg["encoder_hidden_dims"],
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
        dc_iter=train_cfg["dc_iter"],
        batch_size=train_cfg["batch_size"],
        verbose=train_cfg["verbose"],
        check_ari=False,
        early_stopping=False
    )
    # save model
    # get time for save model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    saved_model_dir = os.path.join("trained_model", timestamp)
    os.makedirs(saved_model_dir, exist_ok=True)
    final_model.save_model(saved_model_dir)

    # ==== Save logs ====
    total_loss = log_loss["loss"]
    log_metric = log_loss["log_ari"]
    epochs = range(1, len(total_loss) + 1)

    epochs = range(1, len(total_loss) + 1)

    # Loss
    plt.figure(figsize=(14, 6))
    plt.plot(epochs, total_loss, linestyle='-', color='blue')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.ylim(0, 1.0)
    plt.grid(True)
    plt.savefig(os.path.join(saved_model_dir, "loss.png"))
    plt.close()


    xs_hat = final_model.extract_feature(xs.cuda())
    xt_hat = final_model.extract_feature(xt.cuda())
    xs_hat = xs_hat.cpu().numpy()
    xt_hat = xt_hat.cpu().numpy()
    print("Xs",np.mean(xs_hat, axis=1))
    print("Xt",np.mean(xt_hat, axis=1))
    gendata.visualby_tsne(Xs, Xt, path=os.path.join(saved_model_dir, "train_tsne.png"))
    return final_model

if __name__ == "__main__":
    ns, nt, d = 2000, 100, 1
    # n_clusters = 2
    mu_s = np.full((ns,d),2)
    mu_t = np.full((nt,d),0)
    Xs = gendata.sample_normal_data(mu=mu_s, sigma=1)
    Xt = gendata.sample_normal_data(mu=mu_t, sigma=1)
    wdgrl = train_model(Xs, Xt)
    # with open("config.yaml", "r") as f:
    #     config = yaml.safe_load(f)

    # exp_cfg = config["experiment"]
    # data_cfg = config["data"]
    # model_cfg = config["model"]
    # train_cfg = config["training"]
    # wdgrl = WDGRL(
    #     input_dim=d,
    #     encoder_hidden_dims=model_cfg["encoder_hidden_dims"],
    #     critic_hidden_dims=model_cfg["critic_hidden_dims"],
    #     alpha1=model_cfg["alpha1"],
    #     alpha2=model_cfg["alpha2"],
    #     seed=exp_cfg["model_random_state"],
    # )
    # wdgrl.load_model("trained_model/20250831-165108")
    # final_model.load_state_dict(torch.load("trained_model/wdgrl_model.pth"))
    # Xs_train, Xs_test = train_test_split(Xs, test_size=0.5, random_state=42)
    # Xt_train, Xt_test = train_test_split(Xt, test_size=0.5, random_state=42)
    # ns_test = Xs_test.shape[0]
    # nt_test = Xt_test.shape[0]
    # wdgrl = train_model(Xs_train, Xt_train)
    
    # Xs_test_torch = torch.from_numpy(Xs_test).float()
    # Xt_test_torch = torch.from_numpy(Xt_test).float()



    # # kmean clustering
    # X_transformed = np.vstack((xs_hat, xt_hat))

    # kmeans = KMeans(n_clusters=n_clusters, random_state=4)
    # kmeans.fit(X_transformed)
    # labels = kmeans.labels_[ns_test:]

    # cluster_id = list(range(n_clusters))
    # list_cluster = [np.where(labels == i)[0].tolist() for i in range(n_clusters)]

    # print(list_cluster)

    # DAinterval = utils.construct_interval.ReLUcondition(wdgrl.encoder, a, b, X)

