import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import numpy as np

import math
from tqdm import tqdm
from data_loader import load_data
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")


def predict(g: dgl.DGLGraph, features: torch.Tensor, mask: torch.Tensor, model: nn.Module, device: str = "cpu", info=False):
    """Predict with model"""
    if info:
        print("Predicting...")

    # === Copy data to device ===
    g = g.to(device)
    features = features.to(device)
    mask = mask.to(device)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        return indices


def evaluate(g: dgl.DGLGraph, features: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, model: nn.Module, device: str = "cpu", info=False):
    """Evaluate model accuracy"""
    if info:
        print("Evaluating...")

    # === Copy data to device ===
    g = g.to(device)
    features = features.to(device)
    labels = labels.to(device)
    mask = mask.to(device)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g: dgl.DGLGraph, features: torch.Tensor, train_labels: torch.Tensor,
          val_labels: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor,
          model: nn.Module, loss_fcn, optimizer, epochs: int, es_iters: int = None, device: str = "cpu", info=False):
    """Train model"""
    if info:
        print("Training...")

    if val_labels is not None and val_mask is not None:
        # If early stopping criteria, initialize relevant parameters
        if es_iters:
            print("Early stopping monitoring on")
            loss_min = 1e8
            es_i = 0

    # === Copy data to device ===
    g = g.to(device)
    features = features.to(device)
    train_labels = train_labels.to(device)
    if val_labels is not None:
        val_labels = val_labels.to(device)
    train_mask = train_mask.to(device)
    if val_mask is not None:
        val_mask = val_mask.to(device)
    model = model.to(device)

    # training loop
    epochs_progress = tqdm(range(epochs), desc='Epoch')
    train_log = tqdm(total=0, position=1, bar_format='{desc}')
    for epoch in epochs_progress:
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if val_labels is not None:
            acc = evaluate(g, features, val_labels, val_mask, model, device)
            train_log.set_description_str(
                "Current Epoch: {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                    epoch, loss.item(), acc
                )
            )
            val_loss = loss_fcn(logits[val_mask], val_labels).item()
            if es_iters:
                if val_loss < loss_min:
                    loss_min = val_loss
                    es_i = 0
                else:
                    es_i += 1

                if es_i >= es_iters:
                    epochs_progress.close()
                    train_log.close()
                    print(f"Early stopping at epoch={epoch+1}")
                    break
        else:
            train_log.set_description_str(
                "Current Epoch: {:05d} | Loss {:.4f} ".format(
                    epoch, loss.item()
                )
            )


def write_output(indices, filename='outputs/output.csv'):
    # Export predictions as csv file
    with open(filename, 'w') as f:
        f.write('Id,Predict\n')
        for idx, pred in enumerate(indices):
            f.write(f'{idx},{int(pred)}\n')
    # Please remember to upload your output.csv file to Kaggle for scoring


def process_labels(labels: torch.Tensor, masks: torch.Tensor):
    label_idx = 0
    ret = torch.zeros_like(masks, dtype=torch.int64)
    for i, mask in enumerate(masks):
        if mask:
            ret[i] = labels[label_idx]
            label_idx += 1
    if label_idx != labels.shape[0]:
        print("label mask not matched")
    return ret


def main():
    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--es_iters', type=int, help='num of iters to trigger early stopping')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--ensembles', type=int, default=1)
    parser.add_argument('--node2vec', action='store_true')
    args = parser.parse_args()

    es_iters = args.es_iters
    n_ensembles = args.ensembles
    use_gpu = args.use_gpu
    model_str = args.model
    train_all = args.train_all
    node2vec = args.node2vec
    tuning = args.tuning
    epochs = args.epochs

    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # === Load data ===
    features, graph, num_classes, \
        train_labels, val_labels, test_labels, \
        train_mask, val_mask, test_mask = load_data()

    if train_all:
        labels = (process_labels(train_labels, train_mask) + process_labels(val_labels, val_mask))
        mask = train_mask | val_mask
        train_mask = mask
        train_labels = labels[train_mask]
        val_labels = None
        val_mask = None

    # === Get dimensions ===
    in_size = features.shape[1]
    out_size = num_classes

    # === Initialize the model (Baseline Model: GCN) ===
    model: nn.Module = None

    if model_str == "GCN":
        from model import GCN

        if tuning:
            def objective(trial):
                # 0.786: {'hid_size': 225, 'dropout': 0.4998782070694462, 'lr': 0.15539955776920314, 'weight_decay': 0.0005277555782436347}
                hid_size = trial.suggest_int('hid_size', 1, 512)
                dropout = trial.suggest_float('dropout', 0.1, 0.6)
                lr = trial.suggest_float('lr', 0, 1)
                weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)

                model = GCN(in_size, hid_size, out_size, dropout).to(device)
                loss_fcn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                train(graph, features, train_labels, val_labels, train_mask, val_mask,
                      model, loss_fcn, optimizer, epochs, es_iters=es_iters, device=device)
                return evaluate(graph, features, val_labels, val_mask, model, device)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            print('Best score:', study.best_value)
            print('Best trial parameters:', study.best_trial.params)
        else:
            model = GCN(in_size, 225, out_size, 0.4998782070694462).to(device)
            loss_fcn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.15539955776920314, weight_decay=0.0005277555782436347)
            train(graph, features, train_labels, val_labels, train_mask, val_mask,
                  model, loss_fcn, optimizer, epochs, es_iters=es_iters, device=device, info=True)
            indices = predict(graph, features, test_mask, model, device=device, info=True)

    elif model_str == "SAGE":
        from model import SAGE

        if tuning:
            def objective(trial):
                hid_size = trial.suggest_int('hid_size', 1, 512)
                dropout = trial.suggest_float('dropout', 0.1, 0.6)
                lr = trial.suggest_float('lr', 0, 1)
                weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)

                model = SAGE(in_size, hid_size, out_size, dropout).to(device)
                loss_fcn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                train(graph, features, train_labels, val_labels, train_mask, val_mask,
                      model, loss_fcn, optimizer, epochs, es_iters=es_iters, device=device)
                return evaluate(graph, features, val_labels, val_mask, model, device=device)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            print('Best score:', study.best_value)
            print('Best trial parameters:', study.best_trial.params)
        else:
            # 0.804: {'hid_size': 204, 'dropout': 0.22420182392055, 'lr': 0.005031767855753401, 'weight_decay': 8.544936816998818e-05}
            model = SAGE(in_size, 204, out_size, 0.22420182392055).to(device)
            loss_fcn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005031767855753401, weight_decay=8.544936816998818e-05)
            train(graph, features, train_labels, val_labels, train_mask, val_mask,
                  model, loss_fcn, optimizer, epochs, es_iters=es_iters, device=device, info=True)
            indices = predict(graph, features, test_mask, model, device=device, info=True)

    elif model_str == "CHEB":
        from model import CHEB

        if tuning:
            def objective(trial):
                hid_size = trial.suggest_int('hid_size', 1, 512)
                k = trial.suggest_int('k', 1, 10)
                dropout = trial.suggest_float('dropout', 0.1, 0.9)
                lr = trial.suggest_float('lr', 0, 1e-2)
                weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)

                model = CHEB(in_size, hid_size, out_size, k, dropout).to(device)
                loss_fcn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                train(graph, features, train_labels, val_labels, train_mask, val_mask,
                      model, loss_fcn, optimizer, epochs, es_iters=es_iters, device=device)
                return evaluate(graph, features, val_labels, val_mask, model, device=device)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            print('Best score:', study.best_value)
            print('Best trial parameters:', study.best_trial.params)
        else:
            hid_size = 64
            k = 8
            dropout = 0.5
            lr = 5e-3
            weight_decay = 5e-5

            model = CHEB(in_size, hid_size, out_size, k, dropout).to(device)
            loss_fcn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            train(graph, features, train_labels, val_labels, train_mask, val_mask,
                  model, loss_fcn, optimizer, epochs, es_iters=es_iters, device=device, info=True)
            indices = predict(graph, features, test_mask, model, device=device, info=True)

    elif model_str == "GAT":
        from model import GAT
        # 0.802: {'hid_size': 15, 'num_heads': 78, 'dropout': 0.35504093201574133, 'lr': 0.0001596696063196532, 'weight_decay': 0.0007732094500682307}

        if tuning:
            def objective(trial):
                hid_size = trial.suggest_int('hid_size', 1, 64)
                num_heads = trial.suggest_int('num_heads', 1, 128)
                feat_drop = trial.suggest_float('feat_drop', 0.1, 0.6)
                attn_drop = trial.suggest_float('attn_drop', 0.1, 0.6)
                lr = trial.suggest_float('lr', 0, 1e-3)
                weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)

                model = GAT(in_size, hid_size, out_size, num_heads, feat_drop=feat_drop, attn_drop=attn_drop).to(device)
                loss_fcn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                train(graph, features, train_labels, val_labels, train_mask, val_mask,
                      model, loss_fcn, optimizer, epochs, es_iters=es_iters, device=device)
                return evaluate(graph, features, val_labels, val_mask, model, device=device)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            print('Best score:', study.best_value)
            print('Best trial parameters:', study.best_trial.params)
        else:
            model = GAT(in_size, 15, out_size, 78, 0.35504093201574133).to(device)
            loss_fcn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001596696063196532, weight_decay=0.0007732094500682307)
            train(graph, features, train_labels, val_labels, train_mask, val_mask,
                  model, loss_fcn, optimizer, epochs, es_iters=es_iters, device=device, info=True)
            indices = predict(graph, features, test_mask, model, device=device, info=True)

    elif model_str == "DotGAT":
        from model import DotGAT

        if tuning:
            def objective(trial):
                hid_size = trial.suggest_int('hid_size', 1, 64)
                num_heads = trial.suggest_int('num_heads', 1, 128)
                dropout = trial.suggest_float('dropout', 0.1, 0.6)
                lr = trial.suggest_float('lr', 0, 1e-3)
                weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)

                model = DotGAT(in_size, hid_size, out_size, num_heads, dropout).to(device)
                loss_fcn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                train(graph, features, train_labels, val_labels, train_mask, val_mask,
                      model, loss_fcn, optimizer, epochs, es_iters=es_iters, device=device)
                return evaluate(graph, features, val_labels, val_mask, model, device=device)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            print('Best score:', study.best_value)
            print('Best trial parameters:', study.best_trial.params)
        else:
            model = DotGAT(in_size, 15, out_size, 78, 0.35504093201574133).to(device)
            loss_fcn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001596696063196532, weight_decay=0.0007732094500682307)
            train(graph, features, train_labels, val_labels, train_mask, val_mask,
                  model, loss_fcn, optimizer, epochs, es_iters=es_iters, device=device, info=True)
            indices = predict(graph, features, test_mask, model, device=device, info=True)

    elif model_str == "Grace":
        import grace
        from grace import Grace

        # if node2vec:
        #     walk_length = 500
        #     walk_features: torch.Tensor = dgl.sampling.node2vec_random_walk(graph, graph.nodes(), 0.25, 0.25, walk_length=walk_length)  # (19717, walk_length+1)
        #     features = torch.hstack((features, walk_features))  # (19717, 500+walk_length+1)
        #     m = features.mean(0, keepdim=True)
        #     s = features.std(0, unbiased=False, keepdim=True)
        #     features = (features - m) / s
        #     in_size = features.shape[1]

        _features = features
        if tuning:
            def objective(trial):
                hid_size = trial.suggest_int('hid_size', 1, 512)
                out_size = trial.suggest_int('out_size', 1, 512)
                num_layers = trial.suggest_int('num_layers', 2, 10)
                act_fn_str = trial.suggest_categorical('act_fn_str', ['relu', 'prelu'])
                drop_edge_rate_1 = trial.suggest_float('drop_edge_rate_1', 0.1, 0.9)
                drop_edge_rate_2 = trial.suggest_float('drop_edge_rate_2', 0.1, 0.9)
                drop_feature_rate_1 = trial.suggest_float('drop_feature_rate_1', 0.1, 0.9)
                drop_feature_rate_2 = trial.suggest_float('drop_feature_rate_2', 0.1, 0.9)
                temp = trial.suggest_float('temp', 0.1, 0.6)
                lr = trial.suggest_float('lr', 0, 1e-3)
                weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)

                if node2vec:
                    walk_length = trial.suggest_int('walk_length', 0, 1000)
                    walk_p = trial.suggest_float('walk_p', 0, 1)
                    walk_q = trial.suggest_float('walk_q', 0, 1)

                    walk_features: torch.Tensor = dgl.sampling.node2vec_random_walk(graph, graph.nodes(), walk_p, walk_q, walk_length=walk_length)  # (19717, walk_length+1)
                    # features = torch.hstack((_features, walk_features))  # (19717, 500+walk_length+1)
                    # m = features.mean(0, keepdim=True)
                    # s = features.std(0, unbiased=False, keepdim=True)
                    # features = (features - m) / s
                    # in_size = features.shape[1]
                else:
                    walk_features = None

                batch_size = math.ceil(graph.num_nodes() * 0.5)
                # batch_size = graph.num_nodes()

                act_fn = ({"relu": nn.ReLU(), "prelu": nn.PReLU()})[act_fn_str]
                model = Grace(in_size, hid_size, out_size, num_layers, act_fn, temp).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                grace.train(graph, features, train_labels, val_labels, train_mask, val_mask,
                            model, optimizer, epochs, batch_size,
                            drop_feature_rate_1, drop_feature_rate_2, drop_edge_rate_1, drop_edge_rate_2, device=device)
                acc = grace.evaluate(graph, features, train_labels, val_labels, train_mask, val_mask, model, device=device)
                indices = grace.predict(graph, features, train_labels, train_mask, test_mask, model, device=device)
                write_output(indices, filename=f'outputs/output.{int(1000*acc)}.csv')
                return acc

            optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
            study_name = "optuna-grace"
            storage_name = "sqlite:///{}.db".format(study_name)
            study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)
            study.optimize(objective, n_trials=100)
            print('Best score:', study.best_value)
            print('Best trial parameters:', study.best_trial.params)
        else:
            # 0.854: {'hid_size': 266, 'out_size': 375, 'num_layers': 2, 'act_fn_str': 'prelu', 'drop_edge_rate_1': 0.10661225285451398, 'drop_edge_rate_2': 0.3833879662681614, 'drop_feature_rate_1': 0.4508433772266215, 'drop_feature_rate_2': 0.4303120406763847, 'temp': 0.47456804523077506, 'lr': 0.0009660679922312985, 'weight_decay': 0.0009953991461427538}

            # with node2vec
            # 0.84: {'act_fn_str': 'prelu', 'drop_edge_rate_1': 0.24362765977905837, 'drop_edge_rate_2': 0.6342788109844308, 'drop_feature_rate_1': 0.5361947735169645, 'drop_feature_rate_2': 0.4074135977825209, 'hid_size': 422, 'lr': 0.0006801087289555603, 'num_layers': 3, 'out_size': 474, 'temp': 0.40170123987706596, 'weight_decay': 0.0008015346838736527}
            # 0.848: {'act_fn_str': 'relu', 'drop_edge_rate_1': 0.3179447606672301, 'drop_edge_rate_2': 0.8426984175787502, 'drop_feature_rate_1': 0.5335787455853964, 'drop_feature_rate_2': 0.3826710505908456, 'hid_size': 230, 'lr': 0.00032165718913340933, 'num_layers': 10, 'out_size': 511, 'temp': 0.35364542318132747, 'walk_length': 433, 'walk_p': 0.8788506606911656, 'walk_q': 0.7999299625540663, 'weight_decay': 6.581158747208596e-05}
            # 0.856: {'act_fn_str': 'relu', 'drop_edge_rate_1': 0.5251482599492395, 'drop_edge_rate_2': 0.37782962566511036, 'drop_feature_rate_1': 0.5292745614691008, 'drop_feature_rate_2': 0.3579714315545969, 'hid_size': 355, 'lr': 0.000686604412691569, 'num_layers': 5, 'out_size': 291, 'temp': 0.5388011027084435, 'walk_length': 182, 'walk_p': 0.12214064945424387, 'walk_q': 0.3603707118954555, 'weight_decay': 0.00024982201371369795}

            param = {'act_fn_str': 'relu', 'drop_edge_rate_1': 0.3179447606672301, 'drop_edge_rate_2': 0.8426984175787502, 'drop_feature_rate_1': 0.5335787455853964, 'drop_feature_rate_2': 0.3826710505908456, 'hid_size': 230, 'lr': 0.00032165718913340933, 'num_layers': 10, 'out_size': 511, 'temp': 0.35364542318132747, 'walk_length': 433, 'walk_p': 0.8788506606911656, 'walk_q': 0.7999299625540663, 'weight_decay': 6.581158747208596e-05}

            hid_size = param['hid_size']
            out_size = param['out_size']
            num_layers = param['num_layers']
            act_fn_str = param['act_fn_str']
            drop_edge_rate_1 = param['drop_edge_rate_1']
            drop_edge_rate_2 = param['drop_edge_rate_2']
            drop_feature_rate_1 = param['drop_feature_rate_1']
            drop_feature_rate_2 = param['drop_feature_rate_2']
            temp = param['temp']
            lr = param['lr']
            weight_decay = param['weight_decay']

            if node2vec:
                walk_length = param['walk_length']
                walk_p = param['walk_p']
                walk_q = param['walk_q']

                walk_features: torch.Tensor = dgl.sampling.node2vec_random_walk(graph, graph.nodes(), walk_p, walk_q, walk_length=walk_length)  # (19717, walk_length+1)
                print(walk_features)
                # features = torch.hstack((_features, walk_features))  # (19717, 500+walk_length+1)
                # m = features.mean(0, keepdim=True)
                # s = features.std(0, unbiased=False, keepdim=True)
                # features = (features - m) / s
                # in_size = features.shape[1]
            else:
                walk_features = None

            batch_size = math.ceil(graph.num_nodes() * 0.5)
            # batch_size = graph.num_nodes()

            if n_ensembles == 1:
                act_fn = ({"relu": nn.ReLU(), "prelu": nn.PReLU()})[act_fn_str]
                model = Grace(in_size, hid_size, out_size, num_layers, act_fn, temp).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                grace.train(graph, features, train_labels, val_labels, train_mask, val_mask,
                            model, optimizer, epochs, batch_size,
                            drop_feature_rate_1, drop_feature_rate_2, drop_edge_rate_1, drop_edge_rate_2, device=device, info=True)
                indices = grace.predict(graph, features, train_labels, train_mask, test_mask, model, device=device, info=True)
            else:
                if val_labels is not None:
                    val_probas = np.zeros((val_labels.shape[0], num_classes))
                test_probas = np.zeros((test_labels.shape[0], num_classes))

                for _ in range(n_ensembles):
                    act_fn = ({"relu": nn.ReLU(), "prelu": nn.PReLU()})[act_fn_str]
                    model = Grace(in_size, hid_size, out_size, num_layers, act_fn, temp).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                    grace.train(graph, features, train_labels, val_labels, train_mask, val_mask,
                                model, optimizer, epochs, batch_size,
                                drop_feature_rate_1, drop_feature_rate_2, drop_edge_rate_1, drop_edge_rate_2, device=device, pbar=False)
                    if val_labels is not None:
                        val_probas += grace.predict(graph, features, train_labels, train_mask, val_mask, model, device=device, proba=True)
                    test_probas += grace.predict(graph, features, train_labels, train_mask, test_mask, model, device=device, proba=True)

                if val_labels is not None:
                    val_labels_pred = val_probas.argmax(axis=1).astype(np.int64)
                    print("Ensembled Acc: {:.4f}".format(accuracy_score(val_labels_pred, val_labels)))
                indices = test_probas.argmax(axis=1).astype(np.int64)

    elif model_str == "SSP":
        import ssp
        from ssp import Net

        edge_index = torch.vstack(graph.edges())

        if tuning:
            def objective(trial: optuna.Trial):
                hid_size = trial.suggest_int('hid_size', 1, 512)
                dropout = trial.suggest_float('dropout', 0.1, 0.9)
                eps = trial.suggest_float('eps', 0, 1e-1)
                update_freq = trial.suggest_int('update_freq', 1, 64)
                alpha = None
                gamma = trial.suggest_int('gamma', 1, 64)
                lr = trial.suggest_float('lr', 0, 1e-2)
                weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)
                # momentum = trial.suggest_float('momentum', 0, 1)
                momentum = 0.9
                # precond_str = trial.suggest_categorical('precond_str', ['kfac', None])
                precond_str = None
                # optim_str = trial.suggest_categorical('optim_str', ['sgd', 'adam'])
                optim_str = 'adam'

                model = Net(in_size, hid_size, out_size, dropout)
                model.to(device).reset_parameters()

                preconditioner = ssp.KFAC(
                    model,
                    eps,
                    sua=False,
                    pi=False,
                    update_freq=update_freq,
                    alpha=alpha if alpha is not None else 1.,
                    constraint_norm=False
                ) if precond_str == 'kfac' else None
                optimizer = {'adam': torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
                             'sgd': torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)}[optim_str]

                ssp.train(edge_index, features, train_labels, val_labels, train_mask, val_mask,
                          model, optimizer, gamma, epochs, es_iters=es_iters, device=device, preconditioner=preconditioner)
                return ssp.evaluate(edge_index, features, val_labels, val_mask, model, device=device)

            optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
            study_name = "optuna-ssp"
            storage_name = "sqlite:///{}.db".format(study_name)
            study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)
            study.optimize(objective, n_trials=100)
            print('Best score:', study.best_value)
            print('Best trial parameters:', study.best_trial.params)
        else:
            hid_size = 32
            dropout = 0.5
            eps = 0.01
            update_freq = 32
            alpha = None
            gamma = 5
            lr = 0.001
            weight_decay = 0.0005
            momentum = 0.9
            # precond_str = 'kfac'
            precond_str = None
            optim_str = 'adam'
            # optim_str = 'sgd'

            model = Net(in_size, hid_size, out_size, dropout)
            model.to(device).reset_parameters()

            preconditioner = ssp.KFAC(
                model,
                eps,
                sua=False,
                pi=False,
                update_freq=update_freq,
                alpha=alpha if alpha is not None else 1.,
                constraint_norm=False
            ) if precond_str == 'kfac' else None
            optimizer = {'adam': torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
                         'sgd': torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)}[optim_str]

            ssp.train(edge_index, features, train_labels, val_labels, train_mask, val_mask,
                      model, optimizer, gamma, epochs, es_iters=es_iters, device=device, preconditioner=preconditioner)
            indices = ssp.predict(edge_index, features, test_mask, model, device=device, info=True)

    else:
        print(f"Model not found: {model_str}")
        exit(1)

    if not tuning:
        print("Export predictions as csv file.")
        write_output(indices)


if __name__ == '__main__':
    main()
