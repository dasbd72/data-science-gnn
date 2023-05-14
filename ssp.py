import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch.optim.optimizer import Optimizer
from torch_geometric.nn import GCNConv

import numpy as np
from tqdm import tqdm


class CRD(torch.nn.Module):
    def __init__(self, in_size, out_size, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(in_size, out_size, cached=True)
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x


class CLS(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(CLS, self).__init__()
        self.conv = GCNConv(in_size, out_size, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class Net(torch.nn.Module):
    def __init__(self, in_size, hid_size, out_size, dropout):
        super(Net, self).__init__()
        self.crd = CRD(in_size, hid_size, dropout)
        self.cls = CLS(hid_size, out_size)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, edge_index, h):
        h = self.crd(h, edge_index)
        h = self.cls(h, edge_index)
        return h


class KFAC(Optimizer):

    def __init__(self, net, eps, sua=False, pi=False, update_freq=1,
                 alpha=1.0, constraint_norm=False):
        """ K-FAC Preconditionner for Linear and Conv2d layers.
        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.
        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            constraint_norm (bool): Scale the gradients by the squared
                fisher norm.
        """
        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0

        for mod in net.children():
            mod_name = mod.__class__.__name__
            if mod_name not in ['CRD', 'CLS']:
                continue

            handle = mod.register_forward_pre_hook(self._save_input)
            self._fwd_handles.append(handle)

            for sub_mod in mod.children():
                sub_mod_name = sub_mod.__class__.__name__
                if sub_mod_name not in ['GCNConv']:
                    continue

                handle = sub_mod.register_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)

                params = []
                for sub_sub_mod in sub_mod.children():
                    sub_sub_mod_name = sub_sub_mod.__class__.__name__
                    if sub_sub_mod_name not in ['Linear']:
                        continue
                    params = [sub_sub_mod.weight]
                if sub_mod.bias is not None:
                    params.append(sub_mod.bias)
                d = {'params': params, 'mod': mod, 'sub_mod': sub_mod, 'sub_sub_mod': sub_sub_mod}
                self.params.append(d)

        super(KFAC, self).__init__(self.params, {})

    def step(self, update_stats=True, update_params=True, lam=0.):
        """Performs one step of preconditioning."""
        self.lam = lam
        fisher_norm = 0.
        for group in self.param_groups:

            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]

            # Update convariances and inverses
            if update_stats:
                if self._iteration_counter % self.update_freq == 0:
                    self._compute_covs(group, state)
                    ixxt, iggt = self._inv_covs(state['xxt'], state['ggt'],
                                                state['num_locations'])
                    state['ixxt'] = ixxt
                    state['iggt'] = iggt
                else:
                    if self.alpha != 1:
                        self._compute_covs(group, state)

            if update_params:
                gw, gb = self._precond(weight, bias, group, state)

                # Updating gradients
                if self.constraint_norm:
                    fisher_norm += (weight.grad * gw).sum()

                weight.grad.data = gw
                if bias is not None:
                    if self.constraint_norm:
                        fisher_norm += (bias.grad * gb).sum()
                    bias.grad.data = gb

            # Cleaning
            if 'x' in self.state[group['mod']]:
                del self.state[group['mod']]['x']
            if 'gy' in self.state[group['mod']]:
                del self.state[group['mod']]['gy']

        # Eventually scale the norm of the gradients
        if update_params and self.constraint_norm:
            scale = (1. / fisher_norm) ** 0.5
            for group in self.param_groups:
                for param in group['params']:
                    print(param.shape, param)
                    param.grad.data *= scale

        if update_stats:
            self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        # i = (x, edge_index)
        if mod.training:
            self.state[mod]['x'] = i[0]

            self.mask = i[-1]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if mod.training:
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(1)
            self._cached_edge_index = mod._cached_edge_index

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        ixxt = state['ixxt']  # [d_in x d_in]
        iggt = state['iggt']  # [d_out x d_out]
        g = weight.grad.data.T  # [d_in x d_out]
        s = g.shape

        g = g.contiguous().view(-1, g.shape[-1])

        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(1, gb.shape[0])], dim=0)

        g = torch.mm(ixxt, torch.mm(g, iggt))
        if bias is not None:
            gb = g[-1].contiguous().view(*bias.shape)
            g = g[:-1]
        else:
            gb = None
        g = g.contiguous().view(*s)
        return g.T, gb

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        sub_mod = group['sub_mod']
        sub_sub_mod = group['sub_sub_mod']
        x = self.state[group['mod']]['x']  # [n x d_in]
        gy = self.state[group['sub_mod']]['gy']  # [n x d_out]
        edge_index, edge_weight = self._cached_edge_index  # [2, n_edges], [n_edges]

        n = float(self.mask.sum() + self.lam*((~self.mask).sum()))

        x = scatter(x[edge_index[0]]*edge_weight[:, None], edge_index[1], dim=0)

        x = x.data.t()

        if sub_sub_mod.weight.ndim == 3:
            x = x.repeat(sub_sub_mod.weight.shape[0], 1)

        if sub_mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)

        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()) / n
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / n)

        gy = gy.data.t()  # [d_out x n]

        state['num_locations'] = 1
        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()) / n
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / n)

    def _inv_covs(self, xxt, ggt, num_locations):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0]
            tg = torch.trace(ggt) * xxt.shape[0]
            pi = (tx / tg)
        # Regularizes and inverse
        eps = self.eps / num_locations
        diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
        diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)
        ixxt = (xxt + torch.diag(diag_xxt)).inverse()
        iggt = (ggt + torch.diag(diag_ggt)).inverse()

        return ixxt, iggt

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()


# def predict(g: dgl.DGLGraph, features: torch.Tensor, mask: torch.Tensor, model: nn.Module, device: str = "cpu", info=False):
def predict(edge_index: torch.Tensor, features: torch.Tensor, test_mask: torch.Tensor, model: Net, device: str = "cpu", info=False):
    """Predict with model"""
    if info:
        print("Predicting...")

    # === Copy data to device ===
    edge_index = edge_index.to(device)
    features = features.to(device)
    test_mask = test_mask.to(device)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(edge_index, features)
        logits = logits[test_mask]
        _, indices = torch.max(logits, dim=1)
        return indices


def evaluate(edge_index: torch.Tensor, features: torch.Tensor, val_labels: torch.Tensor, val_mask: torch.Tensor, model: Net, device: str = "cpu", info=False):
    """Evaluate model accuracy"""
    if info:
        print("Evaluating...")

    # === Copy data to device ===
    edge_index = edge_index.to(device)
    features = features.to(device)
    val_labels = val_labels.to(device)
    val_mask = val_mask.to(device)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(edge_index, features)
        logits = logits[val_mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == val_labels)
        return correct.item() * 1.0 / len(val_labels)


def train(edge_index: torch.Tensor, features: torch.Tensor, train_labels: torch.Tensor,
          val_labels: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor,
          model: Net, optimizer, gamma, epochs: int, es_iters: int = None, device: str = "cpu", info=False, preconditioner: KFAC = None):
    """Train model"""
    if info:
        print("Training...")

    # If early stopping criteria, initialize relevant parameters
    if es_iters:
        print("Early stopping monitoring on")
        loss_min = 1e8
        es_i = 0

    # === Copy data to device ===
    edge_index = edge_index.to(device)
    features = features.to(device)
    train_labels = train_labels.to(device)
    if val_labels is not None:
        val_labels = val_labels.to(device)
    train_mask = train_mask.to(device)
    if val_labels is not None:
        val_mask = val_mask.to(device)
    model = model.to(device)

    # training loop
    epochs_progress = tqdm(total=epochs, desc='Epoch')
    train_log = tqdm(total=0, position=1, bar_format='{desc}')
    for epoch in range(epochs):
        lam = (float(epoch)/float(epochs))**gamma if gamma is not None else 0.

        model.train()
        optimizer.zero_grad()
        out = model(edge_index, features)
        label = out.max(1)[1]
        label[train_mask] = train_labels
        label.requires_grad = False

        loss = F.nll_loss(out[train_mask], label[train_mask])
        loss += lam * F.nll_loss(out[~train_mask], label[~train_mask])

        loss.backward()
        if preconditioner:
            preconditioner.step(lam=lam)
        optimizer.step()

        acc = evaluate(edge_index, features, val_labels, val_mask, model, device=device)
        train_log.set_description_str("Current Epoch: {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, loss.item(), acc))
        epochs_progress.update()

        if es_iters:
            with torch.no_grad():
                label[val_mask] = val_labels
                val_loss = (F.nll_loss(out[val_mask], label[val_mask]) + lam * F.nll_loss(out[~val_mask], label[~val_mask])).item()
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

    epochs_progress.close()
    train_log.close()
