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
    n_node = graph.num_nodes()

    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = drop_feature(features, feat_drop_rate)

    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    return ng, feat


def drop_feature(features, drop_prob):
    drop_mask = (torch.empty((features.size(1),), dtype=torch.float32, device=features.device).uniform_(0, 1) < drop_prob)
    feat = features.clone()
    feat[:, drop_mask] = 0

    return feat


def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


# === Model ==================================================================
class GCN(nn.Module):
    def __init__(self, in_size, out_size, act_fn, num_layers=2):
        super(GCN, self).__init__()
        assert num_layers >= 2

        self.num_layers = num_layers
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
        self.act_fn = act_fn

    def forward(self, graph, h):
        for i in range(self.num_layers):
            h = self.act_fn(self.convs[i](graph, h))
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
    r"""
        GRACE model
    Parameters
    -----------
    in_size: int
        Input features size.
    hid_size: int
        Hidden features size.
    out_size: int
        Output features size.
    num_layers: int
        Number of the GNN encoder layers.
    act_fn: nn.Module
        Activation function.
    tau: float
        Temperature constant.
    """

    def __init__(self, in_size, hid_size, out_size, num_layers, act_fn, tau):
        super(Grace, self).__init__()
        self.encoder = GCN(in_size, hid_size, act_fn, num_layers)
        self.tau = tau
        self.proj = MLP(hid_size, out_size)

    def forward(self, g, h: torch.Tensor) -> torch.Tensor:
        return self.encoder(g, h)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)

    def sim(self, z1, z2):
        # normalize embeddings across features dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        def f(x): return torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        def f(x): return torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def get_classifier(X_train, y_train_hot):
    logreg = LogisticRegression(solver="liblinear")
    c = 1.4 ** np.arange(-20, 20)
    clf = GridSearchCV(
        estimator=OneVsRestClassifier(logreg),
        param_grid=dict(estimator__C=c),
        n_jobs=8,
        cv=10,
        verbose=0,
    )
    clf.fit(X_train, y_train_hot)
    return clf


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

    clf = get_classifier(X_train, y_train_hot)

    if proba:
        return clf.predict_proba(X_test)
    else:
        y_test_pred = clf.predict_proba(X_test).argmax(axis=1).astype(np.int64)
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

    clf = get_classifier(X_train, y_train_hot)

    y_val_pred = clf.predict_proba(X_val).argmax(axis=1).astype(np.int64)
    return accuracy_score(y_val, y_val_pred)


def gen_node_mask_lst(g: dgl.DGLGraph, batch_size: int, device: str = 'cpu'):
    """ 
    Generate mini batch of node masks
    """
    # N = g.num_nodes()
    # idx = torch.randperm(N)
    # node_mask = torch.arange(0, N)[idx]
    # node_mask_lst = []
    # for i in range(0, N, batch_size):
    #     if i + batch_size > N:
    #         node_mask_lst.append(node_mask[i:N])
    #     else:
    #         node_mask_lst.append(node_mask[i:i+batch_size])

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

            z1 = model(g1, features1)
            z2 = model(g2, features2)

            loss = model.loss(z1, z2)
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
