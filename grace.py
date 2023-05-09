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

        self.convs.append(dglnn.GraphConv(in_size, out_size * 2))
        for _ in range(self.num_layers - 2):
            self.convs.append(dglnn.GraphConv(out_size * 2, out_size * 2))

        self.convs.append(dglnn.GraphConv(out_size * 2, out_size))
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
    temp: float
        Temperature constant.
    """

    def __init__(self, in_size, hid_size, out_size, num_layers, act_fn, temp):
        super(Grace, self).__init__()
        self.encoder = GCN(in_size, hid_size, act_fn, num_layers)
        self.temp = temp
        self.proj = MLP(hid_size, out_size)

    def sim(self, z1, z2):
        # normalize embeddings across features dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = torch.mm(z1, z2.t())
        return s

    def get_loss(self, z1, z2):
        # calculate SimCLR loss
        def f(x): return torch.exp(x / self.temp)

        refl_sim = f(self.sim(z1, z1))  # intra-view pairs
        between_sim = f(self.sim(z1, z2))  # inter-view pairs

        # between_sim.diag(): positive pairs
        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        loss = -torch.log(between_sim.diag() / x1)

        return loss

    def get_embedding(self, graph, h):
        # get embeddings from the model for evaluation
        h = self.encoder(graph, h)

        return h.detach()

    def forward(self, g1, g2, h1, h2):
        # encoding
        h1 = self.encoder(g1, h1)
        h2 = self.encoder(g2, h2)

        # projection
        z1 = self.proj(h1)
        z2 = self.proj(h2)

        # get loss
        l1 = self.get_loss(z1, z2)
        l2 = self.get_loss(z2, z1)
        ret = (l1 + l2) * 0.5

        return ret.mean()


def predict(g: dgl.DGLGraph, features: torch.Tensor, train_labels: torch.Tensor, train_mask: torch.Tensor, test_mask: torch.Tensor, model: nn.Module, device: str = "cpu", info=False, proba=False):
    if info:
        print("Predicting...")
    # get embeddings
    g = g.add_self_loop()
    g = g.to(device)
    feat = features.to(device)
    embeds = model.get_embedding(g, feat)

    # predict on embeddings
    X = embeds.detach().cpu().numpy()
    X = normalize(X, norm="l2")
    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = train_labels.detach().cpu().numpy().reshape(-1, 1)
    y_train_hot = OneHotEncoder(categories="auto").fit_transform(y_train).toarray().astype(np.bool)

    logreg = LogisticRegression(solver="liblinear")
    c = 2.0 ** np.arange(-10, 10)
    clf = GridSearchCV(
        estimator=OneVsRestClassifier(logreg),
        param_grid=dict(estimator__C=c),
        n_jobs=8,
        cv=5,
        verbose=0,
    )
    clf.fit(X_train, y_train_hot)

    if proba:
        return clf.predict_proba(X_test)
    else:
        y_test_pred = clf.predict_proba(X_test).argmax(axis=1).astype(np.int64)
        return y_test_pred


def evaluate(g: dgl.DGLGraph, features: torch.Tensor, train_labels: torch.Tensor, val_labels: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor, model: nn.Module, device: str = "cpu", info=False):
    if info:
        print("Evaluating...")
    # get embeddings
    g = g.add_self_loop()
    g = g.to(device)
    model = model.to(device)
    feat = features.to(device)
    embeds = model.get_embedding(g, feat)

    # predict on embeddings
    X = embeds.detach().cpu().numpy()
    X = normalize(X, norm="l2")
    X_train = X[train_mask.detach().cpu().numpy()]
    X_val = X[val_mask.detach().cpu().numpy()]

    y_train = train_labels.detach().cpu().numpy().reshape(-1, 1)
    y_val = val_labels.detach().cpu().numpy().reshape(-1, 1)
    y_train_hot = OneHotEncoder(categories="auto").fit_transform(y_train).toarray().astype(np.bool)

    logreg = LogisticRegression(solver="liblinear")
    c = 2.0 ** np.arange(-10, 10)
    clf = GridSearchCV(
        estimator=OneVsRestClassifier(logreg),
        param_grid=dict(estimator__C=c),
        n_jobs=8,
        cv=5,
        verbose=0,
    )
    clf.fit(X_train, y_train_hot)

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


def train(g: dgl.DGLGraph, features: torch.Tensor, train_labels: torch.Tensor, val_labels: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor,
          model: nn.Module, optimizer, epochs: int, batch_size: int, drop_feature_rate_1, drop_edge_rate_1, drop_feature_rate_2, drop_edge_rate_2, device: str = "cpu", info=False, pbar=True):
    """
    Training on minibatches of nodes(subgraphs)
    """
    if info:
        print("Training...")

    g = g.to(device)
    features = features.to(device)
    train_labels = train_labels.to(device)
    if val_labels is not None:
        val_labels = val_labels.to(device)
    train_mask = train_mask.to(device)
    if val_mask is not None:
        val_mask = val_mask.to(device)
    model = model.to(device)

    epochs_progress = tqdm(total=epochs, desc='Epoch', disable=not pbar)
    train_log = tqdm(total=0, position=1, bar_format='{desc}', disable=not pbar)

    for epoch in range(epochs):
        node_mask_lst = gen_node_mask_lst(g, batch_size, device=device)
        for node_mask in node_mask_lst:
            sub_graph = g.subgraph(node_mask)
            sub_features = features[node_mask, :]
            model.train()
            optimizer.zero_grad()
            graph1, features1 = aug(sub_graph, sub_features, drop_feature_rate_1, drop_edge_rate_1)
            graph2, features2 = aug(sub_graph, sub_features, drop_feature_rate_2, drop_edge_rate_2)

            graph1 = graph1.to(device)
            graph2 = graph2.to(device)

            features1 = features1.to(device)
            features2 = features2.to(device)

            loss = model(graph1, graph2, features1, features2)
            loss.backward()
            optimizer.step()

        train_log.set_description_str("Current Epoch: {:05d} | Loss {:.4f} ".format(epoch, loss.item()))
        epochs_progress.update()

    epochs_progress.close()
    train_log.close()
    if val_labels is not None and val_mask is not None:
        acc = evaluate(g, features, train_labels, val_labels, train_mask, val_mask, model, device=device)
        print("Accuracy {:.4f}".format(acc))
