import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn

from torch_geometric.nn import GCNConv


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


class CRD(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(CRD, self).__init__()
        self.conv = dglnn.GraphConv(in_dim, out_dim)
        # self.conv = dglnn.SAGEConv(in_dim, out_dim, "mean")
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h):
        h = self.conv(g, h)
        h = F.relu(h)
        h = self.dropout(h)
        return h


class CLS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CLS, self).__init__()
        self.conv = dglnn.GraphConv(in_dim, out_dim)
        # self.conv = dglnn.SAGEConv(in_dim, out_dim, "mean")

    def forward(self, g, h):
        h = self.conv(g, h)
        h = F.log_softmax(h, dim=1)
        return h


class SSP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(SSP, self).__init__()
        self.crd = CRD(in_dim, hidden_dim, dropout)
        self.cls = CLS(hidden_dim, out_dim)

    def forward(self, g, h):
        h = self.crd(g, h)
        h = self.cls(g, h)
        return h
