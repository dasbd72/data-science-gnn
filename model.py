import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn


class GCN(nn.Module):
    """
    Baseline Model:
    - A simple two-layer GCN model, similar to https://github.com/tkipf/pygcn
    - Implement with DGL package
    """

    def __init__(self, in_size, hid_size, out_size, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

# class YourGNNModel(nn.Module):
#     """
#     TODO: Use GCN model as reference, implement your own model here to achieve higher accuracy on testing data
#     """
#     def __init__(self, in_size, hid_size, out_size):
#         super().__init__()

#     def forward(self, g, features):
#         pass


class GCNII(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, dropout, lambda_, alpha, variant=False):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(1, num_layers+1):
            self.convs.append(dglnn.GCN2Conv(hid_size, i, alpha, lambda_))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_size, hid_size))
        self.fcs.append(nn.Linear(hid_size, out_size))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lambda_ = lambda_

    def forward(self, g, features):
        h = features
        h = F.dropout(features, self.dropout, training=self.training)
        _layers = []
        layer_inner = self.act_fn(self.fcs[0](h))
        _layers.append(layer_inner)
        for i, conv in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(conv(g, layer_inner, _layers[0]))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, dropout=0.5):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_size, hid_size, "mean")
        self.conv2 = dglnn.SAGEConv(hid_size, out_size, "mean")
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, feat_drop=0.6, attn_drop=0.6):
        super(GAT, self).__init__()
        self.layer1 = dglnn.GATConv(in_dim, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
        self.layer2 = dglnn.GATConv(hidden_dim*num_heads, out_dim, 1, feat_drop=feat_drop, attn_drop=attn_drop)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.elu(h)
        h = h.flatten(1)
        h = self.layer2(g, h)
        h = h.mean(1)
        return h


class GATv2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, feat_drop=0.6, attn_drop=0.6):
        super(GATv2, self).__init__()
        self.layer1 = dglnn.GATv2Conv(in_dim, hidden_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
        self.layer2 = dglnn.GATv2Conv(hidden_dim*num_heads, out_dim, 1, feat_drop=feat_drop, attn_drop=attn_drop)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.elu(h)
        h = h.flatten(1)
        h = self.layer2(g, h)
        h = h.mean(1)
        return h


class DotGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout=0.5):
        super(DotGAT, self).__init__()
        self.layer1 = dglnn.DotGatConv(in_dim, hidden_dim, num_heads)
        self.layer2 = dglnn.DotGatConv(hidden_dim*num_heads, out_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.elu(h)
        h = h.flatten(1)
        h = self.dropout(h)
        h = self.layer2(g, h)
        h = h.mean(1)
        return h


class DeepGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, num_hidden_layers=0):
        super(DeepGAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GATConv(in_dim, hidden_dim, num_heads, activation=F.relu)
        )
        for _ in range(num_hidden_layers):
            self.layers.append(
                dglnn.GATConv(hidden_dim*num_heads, hidden_dim, num_heads, activation=F.relu)
            )
        self.layers.append(dglnn.GATConv(hidden_dim*num_heads, out_dim, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, h):
        bs = h.shape[0]
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            h = h.reshape(bs, -1)
        return h


class CHEB(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, k, dropout):
        super(CHEB, self).__init__()
        self.layer1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.layer2 = dglnn.SAGEConv(hidden_dim, hidden_dim, "mean")
        self.layer3 = dglnn.ChebConv(hidden_dim, out_dim, k, activation=None)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer2(g, h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer3(g, h)
        return h
