import os
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna

from tqdm import tqdm
from data_loader import load_data
from argparse import ArgumentParser

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
        val_labels = val_labels.to(device)
        train_mask = train_mask.to(device)
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
        # === Copy data to device ===
        g = g.to(device)
        features = features.to(device)
        train_labels = train_labels.to(device)
        train_mask = train_mask.to(device)
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

            train_log.set_description_str(
                "Current Epoch: {:05d} | Loss {:.4f} ".format(
                    epoch, loss.item()
                )
            )


def minibatch_train(g: dgl.DGLGraph, features: torch.Tensor, train_labels: torch.Tensor,
                    val_labels: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor,
                    model: nn.Module, loss_fcn, optimizer, epochs: int, es_iters: int = None, device: str = "cpu"):

    # If early stopping criteria, initialize relevant parameters
    if es_iters:
        print("Early stopping monitoring on")
        loss_min = 1e8
        es_i = 0

    epochs_progress = tqdm(range(epochs), desc='Epoch')
    train_log = tqdm(total=0, position=1, bar_format='{desc}')
    for epoch in epochs_progress:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.DataLoader(
            g, torch.arange(0, g.num_nodes()), sampler,
            batch_size=1024,
            shuffle=True,
            drop_last=False,
            num_workers=4)

        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device) for b in blocks]
            # input_features = blocks[0].srcdata['features']
            # output_labels = blocks[-1].dstdata['label']
            # output_predictions = model(blocks, input_features)
            # loss = compute_loss(output_labels, output_predictions)
            # opt.zero_grad()
            # loss.backward()
            # opt.step()

            model.train()
            logits = model(blocks, features)
            loss = loss_fcn(logits[train_mask], train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate(g, features, val_labels, val_mask, model, device)
        train_log.set_description_str(
            "Current Epoch: {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )

        # val_loss = loss_fcn(logits[val_mask], val_labels).item()
        # if es_iters:
        #     if val_loss < loss_min:
        #         loss_min = val_loss
        #         es_i = 0
        #     else:
        #         es_i += 1

        #     if es_i >= es_iters:
        #         epochs_progress.close()
        #         train_log.close()
        #         print(f"Early stopping at epoch={epoch+1}")
        #         break


if __name__ == '__main__':

    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--es_iters', type=int, help='num of iters to trigger early stopping')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--cheat', action='store_true')
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # === Load data ===
    features, graph, num_classes, \
        train_labels, val_labels, test_labels, \
        train_mask, val_mask, test_mask = load_data()

    # === Combine(Cheat) ===
    if args.cheat:
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
        train_labels = (process_labels(train_labels, train_mask) + process_labels(val_labels, val_mask))
        train_mask = train_mask | val_mask
        train_labels = train_labels[train_mask]
        val_labels = None
        val_mask = None

    # walk_features: torch.Tensor = dgl.sampling.node2vec_random_walk(graph, torch.arange(0, graph.num_nodes()), 1, 1, walk_length=500)

    # === Process features ===
    # walk_features = walk_features.type(torch.float32).to(device)  # (19717, walk_length+1)
    # m = walk_features.mean(0, keepdim=True)
    # s = walk_features.std(0, unbiased=False, keepdim=True)
    # walk_features = (walk_features - m) / s

    # features = torch.hstack((features, walk_features))  # (19717, 500+walk_length+1)
    # m = features.mean(0, keepdim=True)
    # s = features.std(0, unbiased=False, keepdim=True)
    # features = (features - m) / s

    # === Get dimensions ===
    in_size = features.shape[1]
    out_size = num_classes

    # === Initialize the model (Baseline Model: GCN) ===
    model: nn.Module = None

    if args.model == "GCN":
        from model import GCN

        if args.tuning:
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
                      model, loss_fcn, optimizer, args.epochs, es_iters=args.es_iters, device=device)
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
                  model, loss_fcn, optimizer, args.epochs, es_iters=args.es_iters, device=device, info=True)
            indices = predict(graph, features, test_mask, model, device=device, info=True)

    elif args.model == "SAGE":
        from model import SAGE

        if args.tuning:
            def objective(trial):
                # 0.808: {'hid_size': 156, 'dropout': 0.32691986979539955, 'lr': 0.04464424460744484, 'weight_decay': 0.0007347113513438519}
                hid_size = trial.suggest_int('hid_size', 1, 512)
                dropout = trial.suggest_float('dropout', 0.1, 0.6)
                lr = trial.suggest_float('lr', 0, 1)
                weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)

                model = SAGE(in_size, hid_size, out_size, dropout).to(device)
                loss_fcn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                train(graph, features, train_labels, val_labels, train_mask, val_mask,
                      model, loss_fcn, optimizer, args.epochs, es_iters=args.es_iters, device=device)
                return evaluate(graph, features, val_labels, val_mask, model, device=device)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            print('Best score:', study.best_value)
            print('Best trial parameters:', study.best_trial.params)
        else:
            model = SAGE(in_size, 156, out_size, 0.32691986979539955).to(device)
            loss_fcn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.04464424460744484, weight_decay=0.0007347113513438519)
            train(graph, features, train_labels, val_labels, train_mask, val_mask,
                  model, loss_fcn, optimizer, args.epochs, es_iters=args.es_iters, device=device, info=True)
            indices = predict(graph, features, test_mask, model, device=device, info=True)

    elif args.model == "GAT":
        from model import GAT
        # 0.802: {'hid_size': 15, 'num_heads': 78, 'dropout': 0.35504093201574133, 'lr': 0.0001596696063196532, 'weight_decay': 0.0007732094500682307}

        if args.tuning:
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
                      model, loss_fcn, optimizer, args.epochs, es_iters=args.es_iters, device=device)
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
                  model, loss_fcn, optimizer, args.epochs, es_iters=args.es_iters, device=device, info=True)
            indices = predict(graph, features, test_mask, model, device=device, info=True)

    elif args.model == "DotGAT":
        from model import DotGAT

        if args.tuning:
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
                      model, loss_fcn, optimizer, args.epochs, es_iters=args.es_iters, device=device)
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
                  model, loss_fcn, optimizer, args.epochs, es_iters=args.es_iters, device=device, info=True)
            indices = predict(graph, features, test_mask, model, device=device, info=True)

    elif args.model == "SSP":
        from model import SSP
        # 0.812: {'hid_size': 24, 'dropout': 0.29959723751104317, 'lr': 0.00021993423092887875, 'weight_decay': 0.0007062869243741213}

        if args.tuning:
            def objective(trial):
                hid_size = trial.suggest_int('hid_size', 1, 64)
                dropout = trial.suggest_float('dropout', 0.1, 0.6)
                lr = trial.suggest_float('lr', 0, 1e-3)
                weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)

                model = SSP(in_size, hid_size, out_size, dropout).to(device)

                loss_fcn = nn.CrossEntropyLoss()
                # loss_fcn = nn.NLLLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                train(graph, features, train_labels, val_labels, train_mask, val_mask,
                      model, loss_fcn, optimizer, args.epochs, args.es_iters, device)
                return evaluate(graph, features, val_labels, val_mask, model, device)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            print('Best score:', study.best_value)
            print('Best trial parameters:', study.best_trial.params)
        else:
            model = SSP(in_size, 24, out_size, 0.29959723751104317).to(device)
            loss_fcn = nn.CrossEntropyLoss()
            # loss_fcn = nn.NLLLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00021993423092887875, weight_decay=0.0007062869243741213)
            train(graph, features, train_labels, val_labels, train_mask, val_mask,
                  model, loss_fcn, optimizer, args.epochs, args.es_iters, device=device, info=True)
            indices = predict(graph, features, test_mask, model, device=device, info=True)

    elif args.model == "Grace":
        from model import Grace
        from aug import aug

        if args.tuning:
            pass
        else:
            pass

    else:
        print(f"Model not found: {args.model}")
        exit(1)

    if not args.tuning:
        # Export predictions as csv file
        print("Export predictions as csv file.")
        with open('outputs/output.csv', 'w') as f:
            f.write('Id,Predict\n')
            for idx, pred in enumerate(indices):
                f.write(f'{idx},{int(pred)}\n')
        # Please remember to upload your output.csv file to Kaggle for scoring
