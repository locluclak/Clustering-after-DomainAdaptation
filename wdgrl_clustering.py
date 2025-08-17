import numpy as np  
import matplotlib.pyplot as plt
import gendata
import torch
from random import randint

from torch.utils.data import DataLoader, TensorDataset

from models.wdgrl_ae import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import optuna


def clustering(
        X, 
        n_cluster: int):
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels, kmeans

# def objective(trial):
#     # Hyperparameter search space
#     # encoder_hidden_dims = trial.suggest_categorical("encoder_hidden_dims", [[32], [10,10], [32,16]])
#     # critic_hidden_dims  = trial.suggest_categorical("critic_hidden_dims", [[10, 10], [32,16], [32]])
    
#     # n_encoder_layers = trial.suggest_int("n_encoder_layers", 1, 2)
#     # encoder_hidden_dims = []
#     # for i in range(n_encoder_layers):
#     #     # Suggest the number of nodes for each encoder layer
#     #     dim = trial.suggest_int(f"encoder_dim_l{i}", d//2, d*2, log=True)
#     #     encoder_hidden_dims.append(dim)
#     # # This is the line you need to add for the encoder
#     # trial.set_user_attr("encoder_hidden_dims", encoder_hidden_dims)
    
#     # # n_decoder_layers = trial.suggest_int("n_decoder_layers", 0, 2)
#     # # decoder_hidden_dims = []
#     # # for i in range(n_decoder_layers):
#     # #     dim = trial.suggest_int(f"decoder_dim_l{i}", d//2, d*2, log=True)
#     # #     decoder_hidden_dims.append(dim)
#     # decoder_hidden_dims = encoder_hidden_dims[::-1]
#     # trial.set_user_attr("decoder_hidden_dims", decoder_hidden_dims)


#     # # --- Critic Architecture Tuning ---
#     # # Suggest the number of hidden layers for the critic
#     # n_critic_layers = trial.suggest_int("n_critic_layers", 1, 3)
#     # critic_hidden_dims = []
#     # for i in range(n_critic_layers):
#     #     # Suggest the number of nodes for each critic layer
#     #     dim = trial.suggest_int(f"critic_dim_l{i}", d//2, d*2, log=True)
#     #     critic_hidden_dims.append(dim)
#     # # This is the line you need to add for the critic
#     # trial.set_user_attr("critic_hidden_dims", critic_hidden_dims)

    
#     # alpha2 = 1e-3
#     # alpha1 = 1e-3
    
#     alpha2 = trial.suggest_float("alpha2", 1e-4, 1e-2, log=True)
#     alpha1  = trial.suggest_float("alpha1", 1e-4, 1e-2, log=True)
#     gamma = trial.suggest_float("gamma", 0.01, 10.0)
#     lambda_ = trial.suggest_float("lambda_", 0.01, 10.0)
#     # dc_iter = trial.suggest_int("dc_iter", 5, 15)
#     batch_size = 32 #trial.suggest_categorical("batch_size", [16, 32, 64])

#     # Build WDGRL tune_model
#     tune_model = WDGRL(
#         input_dim=d,
#         encoder_hidden_dims=[32],
#         decoder_hidden_dims=[32],
#         critic_hidden_dims=[32,16],
#         use_decoder=True,
#         alpha2=alpha2,
#         alpha1=alpha1,
#         seed=42
#     )

#     # Train (shorter epochs for tuning)
#     loss = tune_model.train(
#         source_dataset_small, 
#         target_dataset_small,
#         num_epochs=60,
#         with_decoder=True,
#         gamma=gamma,
#         lambda_= lambda_,
#         dc_iter=10,#dc_iter,
#         batch_size=batch_size
#     )

#     xs_small_hat = tune_model.extract_feature(xs_small.cuda())
#     xt_small_hat = tune_model.extract_feature(xt_small.cuda())
#     xs_small_hat = xs_small_hat.cpu().numpy()
#     xt_small_hat = xt_small_hat.cpu().numpy()
    
#     # Combine and cluster
#     x_small_comb = np.vstack((xs_small_hat, xt_small_hat))
#     comb_sm_cluster_labels, _ = clustering(x_small_comb, n_clusters)
#     print("Loss: ",loss["loss"][-1])
#     # Compute silhouette_score on transported target domain
#     scr = silhouette_score(x_small_comb, comb_sm_cluster_labels)
#     return scr  # Optuna will maximize this


if __name__ == "__main__":
    ns, nt, d = 1500, 100, 16
    n_clusters = 2
    seed = randint(0, 2**32 - 1)
    # seed = None
    print("Randomly choose seed =",seed)

    dataset = gendata.gen_domain_adaptation_data2(
                ns = ns, 
                nt = nt, 
                n_features= d, 
                dist=3,
                std_source=1,
                std_target=3,
                shift=0,
                random_state=seed
                )
    Xs, Ys, cen_s = dataset["source"]
    Xt, Yt, cen_t = dataset["target"]

    cluster_labels, model1 = clustering(Xt,2)

    ari = adjusted_rand_score(Yt, cluster_labels)
    print(f'Adjusted Rand Index (ARI) only on target domain: {ari:.4f}')



    xs = torch.from_numpy(Xs).float()
    ys = torch.from_numpy(Ys).long()
    xt = torch.from_numpy(Xt).float()
    yt = torch.from_numpy(Yt).long()

    source_dataset = TensorDataset(xs)
    target_dataset = TensorDataset(xt)

    encoder_hidden_dims = [32]
    decoder_hidden_dims = [32]
    critic_hidden_dims = [32,16]

    model = WDGRL(
        input_dim=d, 
        encoder_hidden_dims=encoder_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        decoder_hidden_dims= decoder_hidden_dims,
        use_decoder=True,
        alpha1 = 1e-3,
        alpha2 = 1e-3,
        seed=42)

    losses = model.train(
        source_dataset, 
        target_dataset,
        num_epochs=100,
        gamma=5,
        lambda_=10,
        dc_iter= 8,
        with_decoder=True,
        batch_size=32,
        verbose=False
        )
    
    xs_hat = model.extract_feature(xs.cuda())
    xt_hat = model.extract_feature(xt.cuda())
    xs_hat = xs_hat.cpu().numpy()
    xt_hat = xt_hat.cpu().numpy()

    x_comb = np.vstack((xs_hat, xt_hat))
    comb_cluster_labels, model2 = clustering(x_comb, n_clusters)
    ari = adjusted_rand_score(Yt, comb_cluster_labels[ns:])
    print(f'Adjusted Rand Index (ARI) of target on transported domain: {ari:.4f}')



    # n_source_small = 800
    # n_target_small = 80

    # # Randomly pick indices without replacement
    # src_indices = torch.randperm(len(xs))[:n_source_small]
    # tgt_indices = torch.randperm(len(xt))[:n_target_small]

    # # Create smaller tensors
    # xs_small = xs[src_indices]
    # xt_small = xt[tgt_indices]

    # # Subset labels (Ys, Yt are numpy arrays)
    # Ys_small = Ys[src_indices.cpu().numpy()]
    # Yt_small = Yt[tgt_indices.cpu().numpy()]

    # # Create smaller datasets
    # source_dataset_small = TensorDataset(xs_small)
    # target_dataset_small = TensorDataset(xt_small)

    # import warnings
    # warnings.filterwarnings("ignore", category=UserWarning)


    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=40)

    # print("Best params:", study.best_params)
    # print("Best silhouette_score:", study.best_value)

    # # Assuming 'study' is your Optuna study object after optimization
    # best_trial = study.best_trial

    # print("Best Trial:")
    # print(f"  Value: {best_trial.value}")

    # print("\n  Directly Suggested Parameters:")
    # for key, value in best_trial.params.items():
    #     print(f"    {key}: {value}")

    # print("\n  Dynamically Generated Architecture (User Attributes):")
    # for key, value in best_trial.user_attrs.items():
    #     print(f"    {key}: {value}")


    # best_params = study.best_params

    # final_model = WDGRL(
    #     input_dim=d,
    #     encoder_hidden_dims=[32],#best_trial.user_attrs["encoder_hidden_dims"],
    #     decoder_hidden_dims=[32],#best_trial.user_attrs["decoder_hidden_dims"],
    #     critic_hidden_dims=[32,16],#best_trial.user_attrs["critic_hidden_dims"],
    #     alpha2=best_params["alpha2"],
    #     alpha1=best_params["alpha1"],
    #     use_decoder=True,
    #     seed=42
    # )   

    # # Train longer for final fit
    # log_loss = final_model.train(
    #     source_dataset,
    #     target_dataset,
    #     num_epochs=200,  # more epochs for final training
    #     gamma=best_params["gamma"],
    #     with_decoder=True,
    #     lambda_= best_params["lambda_"],
    #     dc_iter=15,#best_params["dc_iter"],
    #     batch_size=64,#best_params["batch_size"],
    #     verbose = False
    # )

    # xs_hat = final_model.extract_feature(xs.cuda())
    # xt_hat = final_model.extract_feature(xt.cuda())
    # xs_hat = xs_hat.cpu().numpy()
    # xt_hat = xt_hat.cpu().numpy()

    # x_comb = np.vstack((xs_hat, xt_hat))
    # comb_cluster_labels, model2 = clustering(x_comb,2)
    # ari = adjusted_rand_score(Yt, comb_cluster_labels[ns:])
    # print(f'Adjusted Rand Index (ARI) of target on transported domain: {ari:.4f}')