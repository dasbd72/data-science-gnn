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
    print(f"Export predictions to {filename}.")
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


def features_standardize(features):
    m = features.mean(0, keepdim=True)
    s = features.std(0, unbiased=False, keepdim=True)
    features_std = (features - m) / s
    return features_std


if __name__ == '__main__':
    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--es_iters', type=int, help='num of iters to trigger early stopping')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--ensembles', type=int, default=1)
    parser.add_argument('--node2vec', action='store_true')
    args = parser.parse_args()

    es_iters = args.es_iters
    n_ensembles = args.ensembles
    use_gpu = args.use_gpu
    model_str = args.model
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

    # === Get dimensions ===
    in_size = features.shape[1]
    out_size = num_classes
    _features = features

    # === Initialize the model (Baseline Model: GCN) ===
    model: nn.Module = None

    if model_str == "GCN":
        from model import GCN

        if tuning:
            def objective(trial):
                hid_size = trial.suggest_int('hid_size', 1, 512)
                dropout = trial.suggest_float('dropout', 0.1, 0.6)
                lr = trial.suggest_float('lr', 0, 1)
                weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)

                if node2vec:
                    walk_length = trial.suggest_int('walk_length', 0, 1000)
                    walk_p = trial.suggest_float('walk_p', 0, 1)
                    walk_q = trial.suggest_float('walk_q', 0, 1)

                    features = _features
                    walk_features = dgl.sampling.node2vec_random_walk(graph, graph.nodes(), walk_p, walk_q, walk_length=walk_length).type(torch.float32)  # (19717, walk_length+1)
                    features = torch.hstack((features, walk_features))
                    features = features_standardize(features)
                    in_size = features.shape[1]
                else:
                    features = _features
                    in_size = features.shape[1]

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
            # 0.786: {'hid_size': 225, 'dropout': 0.4998782070694462, 'lr': 0.15539955776920314, 'weight_decay': 0.0005277555782436347}
            params = {'hid_size': 225, 'dropout': 0.4998782070694462, 'lr': 0.15539955776920314, 'weight_decay': 0.0005277555782436347}
            hid_size = params['hid_size']
            dropout = params['dropout']
            lr = params['lr']
            weight_decay = params['weight_decay']

            if node2vec:
                walk_length = params['walk_length']
                walk_p = params['walk_p']
                walk_q = params['walk_q']

                features = _features
                walk_features = dgl.sampling.node2vec_random_walk(graph, graph.nodes(), walk_p, walk_q, walk_length=walk_length).type(torch.float32)  # (19717, walk_length+1)
                features = torch.hstack((features, walk_features))
                features = features_standardize(features)
                in_size = features.shape[1]
            else:
                features = _features
                in_size = features.shape[1]

            model = GCN(in_size, hid_size, out_size, dropout).to(device)
            loss_fcn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            train(graph, features, train_labels, val_labels, train_mask, val_mask,
                  model, loss_fcn, optimizer, epochs, es_iters=es_iters, device=device, info=True)
            indices = predict(graph, features, test_mask, model, device=device, info=True)

    if model_str == "GCNII":
        from model import GCNII

        if tuning:
            def objective(trial):
                # hid_size = trial.suggest_int('hid_size', 1, 512)
                hid_size = 64
                # dropout = trial.suggest_float('dropout', 0.1, 0.6)
                dropout = 0.5
                # num_layers = trial.suggest_int('num_layers', 1, 20)
                num_layers = 8
                # lambda_ = trial.suggest_float('lambda_', 0, 1)
                lambda_ = 0.5
                # alpha = trial.suggest_float('alpha', 0, 1)
                alpha = 0.5
                # lr = trial.suggest_float('lr', 0, 1)
                lr = 0.1
                # weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)
                weight_decay = 5e-6

                if node2vec:
                    walk_length = trial.suggest_int('walk_length', 0, 1000)
                    walk_p = trial.suggest_float('walk_p', 0, 1)
                    walk_q = trial.suggest_float('walk_q', 0, 1)

                    features = _features
                    walk_features = dgl.sampling.node2vec_random_walk(graph, graph.nodes(), walk_p, walk_q, walk_length=walk_length).type(torch.float32)  # (19717, walk_length+1)
                    features = torch.hstack((features, walk_features))
                    features = features_standardize(features)
                    in_size = features.shape[1]
                else:
                    features = _features
                    in_size = features.shape[1]

                model = GCNII(in_size, hid_size, out_size, num_layers, dropout, lambda_, alpha).to(device)
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
            # 0.786: {'hid_size': 225, 'dropout': 0.4998782070694462, 'lr': 0.15539955776920314, 'weight_decay': 0.0005277555782436347}
            params = {'hid_size': 225, 'dropout': 0.4998782070694462, 'lr': 0.15539955776920314, 'weight_decay': 0.0005277555782436347}
            hid_size = params['hid_size']
            dropout = params['dropout']
            lr = params['lr']
            weight_decay = params['weight_decay']

            if node2vec:
                walk_length = params['walk_length']
                walk_p = params['walk_p']
                walk_q = params['walk_q']

                features = _features
                walk_features = dgl.sampling.node2vec_random_walk(graph, graph.nodes(), walk_p, walk_q, walk_length=walk_length).type(torch.float32)  # (19717, walk_length+1)
                features = torch.hstack((features, walk_features))
                features = features_standardize(features)
                in_size = features.shape[1]
            else:
                features = _features
                in_size = features.shape[1]

            model = GCNII(in_size, hid_size, out_size, dropout).to(device)
            loss_fcn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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

                if node2vec:
                    walk_length = trial.suggest_int('walk_length', 0, 1000)
                    walk_p = trial.suggest_float('walk_p', 0, 1)
                    walk_q = trial.suggest_float('walk_q', 0, 1)

                    features = _features
                    walk_features = dgl.sampling.node2vec_random_walk(graph, graph.nodes(), walk_p, walk_q, walk_length=walk_length).type(torch.float32)  # (19717, walk_length+1)
                    features = torch.hstack((features, walk_features))
                    features = features_standardize(features)
                    in_size = features.shape[1]
                else:
                    features = _features
                    in_size = features.shape[1]

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
            params = {'hid_size': 204, 'dropout': 0.22420182392055, 'lr': 0.005031767855753401, 'weight_decay': 8.544936816998818e-05}
            hid_size = params['hid_size']
            dropout = params['dropout']
            lr = params['lr']
            weight_decay = params['weight_decay']

            if node2vec:
                walk_length = params['walk_length']
                walk_p = params['walk_p']
                walk_q = params['walk_q']

                features = _features
                walk_features = dgl.sampling.node2vec_random_walk(graph, graph.nodes(), walk_p, walk_q, walk_length=walk_length).type(torch.float32)  # (19717, walk_length+1)
                features = torch.hstack((features, walk_features))
                features = features_standardize(features)
                in_size = features.shape[1]
            else:
                features = _features
                in_size = features.shape[1]

            model = SAGE(in_size, hid_size, out_size, dropout).to(device)
            loss_fcn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            train(graph, features, train_labels, val_labels, train_mask, val_mask,
                  model, loss_fcn, optimizer, epochs, es_iters=es_iters, device=device, info=True)
            indices = predict(graph, features, test_mask, model, device=device, info=True)

    elif model_str == "Grace":
        import grace
        from grace import Grace

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

                    features = _features
                    walk_features = dgl.sampling.node2vec_random_walk(graph, graph.nodes(), walk_p, walk_q, walk_length=walk_length).type(torch.float32)  # (19717, walk_length+1)
                    features = torch.hstack((features, walk_features))
                    features = features_standardize(features)
                    in_size = features.shape[1]
                else:
                    features = _features
                    in_size = features.shape[1]

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

            study_name = "optuna-grace"
            storage_name = "sqlite:///{}.db".format(study_name)
            study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)
            study.optimize(objective, n_trials=100)
            print('Best score:', study.best_value)
            print('Best trial parameters:', study.best_trial.params)
        else:
            # 0.854: {'hid_size': 266, 'out_size': 375, 'num_layers': 2, 'act_fn_str': 'prelu', 'drop_edge_rate_1': 0.10661225285451398, 'drop_edge_rate_2': 0.3833879662681614, 'drop_feature_rate_1': 0.4508433772266215, 'drop_feature_rate_2': 0.4303120406763847, 'temp': 0.47456804523077506, 'lr': 0.0009660679922312985, 'weight_decay': 0.0009953991461427538}

            # with node2vec
            # 0.834: {'hid_size': 506, 'out_size': 190, 'num_layers': 8, 'act_fn_str': 'relu', 'drop_edge_rate_1': 0.21223277464851337, 'drop_edge_rate_2': 0.6424141225292261, 'drop_feature_rate_1': 0.2720684410251818, 'drop_feature_rate_2': 0.5498122852125431, 'temp': 0.28089871726866567, 'lr': 0.0004800855348433543, 'weight_decay': 0.00036167043026865016, 'walk_length': 43, 'walk_p': 0.9573610957634249, 'walk_q': 0.01953251326085559}
            # 0.834: {'hid_size': 352, 'out_size': 147, 'num_layers': 8, 'act_fn_str': 'relu', 'drop_edge_rate_1': 0.14075583164139704, 'drop_edge_rate_2': 0.5746992754201669, 'drop_feature_rate_1': 0.2446741928656893, 'drop_feature_rate_2': 0.4594874443124306, 'temp': 0.27202500316338163, 'lr': 0.00047978437712733535, 'weight_decay': 0.0004421260262163701, 'walk_length': 59, 'walk_p': 0.7569845488227431, 'walk_q': 0.005770545679393149}
            # 0.841: {'act_fn_str': 'relu', 'drop_edge_rate_1': 0.4557229473572051, 'drop_edge_rate_2': 0.6621814364041027, 'drop_feature_rate_1': 0.45597986006744484, 'drop_feature_rate_2': 0.15278738671222483, 'hid_size': 116, 'lr': 0.0009023515728948979, 'num_layers': 6, 'out_size': 244, 'temp': 0.3194226847966415, 'walk_length': 287, 'walk_p': 0.42298370202749275, 'walk_q': 0.506942629940019, 'weight_decay': 0.0007246960055252379}

            params = {'act_fn_str': 'relu', 'drop_edge_rate_1': 0.4557229473572051, 'drop_edge_rate_2': 0.6621814364041027, 'drop_feature_rate_1': 0.45597986006744484, 'drop_feature_rate_2': 0.15278738671222483, 'hid_size': 116, 'lr': 0.0009023515728948979, 'num_layers': 6, 'out_size': 244, 'temp': 0.3194226847966415, 'walk_length': 287, 'walk_p': 0.42298370202749275, 'walk_q': 0.506942629940019, 'weight_decay': 0.0007246960055252379}

            hid_size = params['hid_size']
            out_size = params['out_size']
            num_layers = params['num_layers']
            act_fn_str = params['act_fn_str']
            drop_edge_rate_1 = params['drop_edge_rate_1']
            drop_edge_rate_2 = params['drop_edge_rate_2']
            drop_feature_rate_1 = params['drop_feature_rate_1']
            drop_feature_rate_2 = params['drop_feature_rate_2']
            temp = params['temp']
            lr = params['lr']
            weight_decay = params['weight_decay']

            if node2vec:
                walk_length = params['walk_length']
                walk_p = params['walk_p']
                walk_q = params['walk_q']

                features = _features
                walk_features = dgl.sampling.node2vec_random_walk(graph, graph.nodes(), walk_p, walk_q, walk_length=walk_length).type(torch.float32)  # (19717, walk_length+1)
                features = torch.hstack((features, walk_features))
                features = features_standardize(features)
                in_size = features.shape[1]
            else:
                features = _features
                in_size = features.shape[1]

            batch_size = math.ceil(graph.num_nodes() * 0.5)
            # batch_size = graph.num_nodes()
            act_fn = ({"relu": nn.ReLU(), "prelu": nn.PReLU()})[act_fn_str]

            if n_ensembles == 1:
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
                    model = Grace(in_size, hid_size, out_size, num_layers, act_fn, temp).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                    grace.train(graph, features, train_labels, val_labels, train_mask, val_mask,
                                model, optimizer, epochs, batch_size,
                                drop_feature_rate_1, drop_feature_rate_2, drop_edge_rate_1, drop_edge_rate_2, device=device, pbar=True)
                    if val_labels is not None:
                        val_probas += grace.predict(graph, features, train_labels, train_mask, val_mask, model, device=device, proba=True)
                    test_probas += grace.predict(graph, features, train_labels, train_mask, test_mask, model, device=device, proba=True)

                if val_labels is not None:
                    val_labels_pred = val_probas.argmax(axis=1).astype(np.int64)
                    print("Ensembled Acc: {:.4f}".format(accuracy_score(val_labels_pred, val_labels)))
                indices = test_probas.argmax(axis=1).astype(np.int64)

    else:
        print(f"Model not found: {model_str}")
        exit(1)

    if not tuning:
        write_output(indices)
