import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

from tqdm import tqdm
import math
import random


# === Graph Augmentation ==================================================================
def aug(graph, features, feat_drop_rate, edge_mask_rate) -> tuple[dgl.DGLGraph, torch.Tensor]:
    edge_mask = torch.bernoulli(1 - torch.FloatTensor(np.ones(graph.num_edges()) * edge_mask_rate)).nonzero().squeeze(1)

    src = graph.edges()[0]
    dst = graph.edges()[1]

    g = dgl.graph((src[edge_mask], dst[edge_mask]), num_nodes=graph.num_nodes())
    g = g.add_self_loop()

    drop_mask = (torch.empty((features.size(1),), dtype=torch.float32, device=features.device).uniform_(0, 1) < feat_drop_rate)
    f = features.clone()
    f[:, drop_mask] = 0
    return g, f


# === Model ==================================================================
class GCN(nn.Module):
    def __init__(self, in_size, out_size, act_fn, num_layers=2):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.act_fn = act_fn
        self.convs = nn.ModuleList()

        base_layer = dglnn.SAGEConv

        if base_layer is dglnn.GraphConv:
            self.convs.append(dglnn.GraphConv(in_size, out_size * 2))
            for _ in range(self.num_layers - 2):
                self.convs.append(dglnn.GraphConv(out_size * 2, out_size * 2))
            self.convs.append(dglnn.GraphConv(out_size * 2, out_size))
        elif base_layer is dglnn.SAGEConv:
            self.convs.append(dglnn.SAGEConv(in_size, out_size * 2, "mean"))
            for _ in range(self.num_layers - 2):
                self.convs.append(dglnn.SAGEConv(out_size * 2, out_size * 2, "mean"))
            self.convs.append(dglnn.SAGEConv(out_size * 2, out_size, "mean"))

    def forward(self, g, h):
        for i in range(self.num_layers):
            h = self.convs[i](g, h)
            h = self.act_fn(h)
        return h


class MLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_size, out_size)
        self.fc2 = nn.Linear(out_size, in_size)

    def forward(self, h):
        h = F.elu(self.fc1(h))
        h = self.fc2(h)
        return h


class Grace(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, act_fn, tau):
        super(Grace, self).__init__()
        self.encoder = GCN(in_size, hid_size, act_fn, num_layers)
        self.tau = tau
        self.proj = MLP(hid_size, out_size)

    def forward(self, g, h: torch.Tensor) -> torch.Tensor:
        return self.encoder(g, h)

    def sim(self, h1, h2):
        return torch.mm(F.normalize(h1), F.normalize(h2).t())

    def semi_loss(self, h1: torch.Tensor, h2: torch.Tensor):
        refl_sim = torch.exp(self.sim(h1, h1) / self.tau)
        between_sim = torch.exp(self.sim(h1, h2) / self.tau)
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, h1: torch.Tensor, h2: torch.Tensor):
        h1 = self.proj(h1)
        h2 = self.proj(h2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret


def get_classifier(X_train, y_train_hot):
    logreg = LogisticRegression(solver="liblinear")
    c = 1.4 ** np.arange(-20, 20)
    classifier = GridSearchCV(
        estimator=OneVsRestClassifier(logreg),
        param_grid=dict(estimator__C=c),
        n_jobs=8,
        cv=10,
        verbose=0,
    )
    classifier.fit(X_train, y_train_hot)
    return classifier


def predict(g: dgl.DGLGraph, features: torch.Tensor, train_labels: torch.Tensor, train_mask: torch.Tensor, test_mask: torch.Tensor, model: Grace, device: str = "cpu", info=False, proba=False):
    if info:
        print("Predicting...")
    # get embeddings
    g = g.add_self_loop()
    g = g.to(device)
    feat = features.to(device)
    model.eval()
    with torch.no_grad():
        embeds = model(g, feat)

    # predict on embeddings
    X = embeds.detach().cpu().numpy()
    X = normalize(X, norm="l2")
    X_train = X[train_mask.detach().cpu().numpy()]
    X_test = X[test_mask.detach().cpu().numpy()]

    y_train = train_labels.detach().cpu().numpy().reshape(-1, 1)
    y_train_hot = OneHotEncoder(categories="auto").fit_transform(y_train).toarray().astype(np.bool)

    classifier = get_classifier(X_train, y_train_hot)

    if proba:
        return classifier.predict_proba(X_test)
    else:
        y_test_pred = classifier.predict_proba(X_test).argmax(axis=1).astype(np.int64)
        return y_test_pred


def evaluate(g: dgl.DGLGraph, features: torch.Tensor, train_labels: torch.Tensor, val_labels: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor, model: Grace, device: str = "cpu", info=False):
    if info:
        print("Evaluating...")
    # get embeddings
    g = g.add_self_loop()
    g = g.to(device)
    feat = features.to(device)
    model.eval()
    with torch.no_grad():
        embeds = model(g, feat)

    # predict on embeddings
    X = embeds.detach().cpu().numpy()
    X = normalize(X, norm="l2")
    X_train = X[train_mask.detach().cpu().numpy()]
    X_val = X[val_mask.detach().cpu().numpy()]

    y_train = train_labels.detach().cpu().numpy().reshape(-1, 1)
    y_val = val_labels.detach().cpu().numpy().reshape(-1, 1)
    y_train_hot = OneHotEncoder(categories="auto").fit_transform(y_train).toarray().astype(np.bool)

    classifier = get_classifier(X_train, y_train_hot)

    y_val_pred = classifier.predict_proba(X_val).argmax(axis=1).astype(np.int64)
    return accuracy_score(y_val, y_val_pred)


def gen_node_mask_lst(g: dgl.DGLGraph, batch_size: int, device: str = 'cpu'):
    """ 
    Generate mini batch of node masks
    """
    N = g.num_nodes()
    node_mask_lst = []
    for i in range(0, N, batch_size):
        node_mask_lst.append(torch.randint(0, N, (batch_size, ), device=device))

    return node_mask_lst


def gen_subgraph_lst(g: dgl.DGLGraph, features: torch.Tensor, batch_size: int, batches: int, device: str = 'cpu'):
    """ 
    Generate mini batch of node masks
    """
    g = g.to(device)
    features = features.to(device)
    N = g.num_nodes()
    idx = torch.randint(0, N, (batches * batch_size, ), device=device)
    node_mask = torch.arange(0, N, device=device)[idx]
    subgraph_lst = []
    for i in range(0, batches * batch_size, batch_size):
        mask = node_mask[i:i+batch_size]
        subgraph = g.subgraph(mask)
        sub_features = features[mask]
        subgraph_lst.append((subgraph, sub_features))

    return subgraph_lst


def train(g: dgl.DGLGraph, features: torch.Tensor, train_labels: torch.Tensor, val_labels: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor,
          model: Grace, optimizer, epochs: int, batch_size: int, drop_feature_rate_1, drop_edge_rate_1, drop_feature_rate_2, drop_edge_rate_2, device: str = "cpu", info=False, pbar=True):
    """
    Training on minibatches of nodes(subgraphs)
    """
    if info:
        print("Training...")

    if val_labels is not None:
        val_labels = val_labels.to(device)
    train_mask = train_mask.to(device)
    if val_mask is not None:
        val_mask = val_mask.to(device)
    model = model.to(device)

    epochs_progress = tqdm(total=epochs, desc='Epoch', disable=not pbar)
    train_log = tqdm(total=0, position=1, bar_format='{desc}', disable=not pbar)
    if val_labels is not None and val_mask is not None:
        acc_log = tqdm(total=0, position=2, bar_format='{desc}', disable=not pbar)

    subgraph_lst = gen_subgraph_lst(g, features, batch_size, 5 * math.ceil(g.num_nodes() / batch_size), device=device)
    for epoch in range(epochs):
        for subgraph, sub_features in random.choices(subgraph_lst, k=math.ceil(g.num_nodes() / batch_size)):
            model.train()
            optimizer.zero_grad()
            g1, features1 = aug(subgraph, sub_features, drop_feature_rate_1, drop_edge_rate_1)
            g2, features2 = aug(subgraph, sub_features, drop_feature_rate_2, drop_edge_rate_2)

            g1 = g1.to(device)
            g2 = g2.to(device)

            features1 = features1.to(device)
            features2 = features2.to(device)

            h1 = model(g1, features1)
            h2 = model(g2, features2)

            loss = model.loss(h1, h2)
            loss.backward()
            optimizer.step()

        if val_labels is not None and val_mask is not None and epoch % 10 == 0:
            acc = evaluate(g, features, train_labels, val_labels, train_mask, val_mask, model, device=device)
            acc_log.set_description_str("Accuracy {:.4f} ".format(acc))
        train_log.set_description_str("Current Epoch: {:05d} | Loss {:.4f} ".format(epoch, loss.item()))
        epochs_progress.update()

    epochs_progress.close()
    train_log.close()
    if val_labels is not None and val_mask is not None:
        acc = evaluate(g, features, train_labels, val_labels, train_mask, val_mask, model, device=device)
        acc_log.set_description_str("Accuracy {:.4f}".format(acc))
        acc_log.close()
