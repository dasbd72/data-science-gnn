import pickle as pkl
import sys
import dgl
import torch


def load_data() -> tuple[torch.Tensor, dgl.DGLGraph, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    * Load data from pickle file in folder `dataset`.
    * No need to modify.

    * test_labels is an array of length 1000 with each element being -1.
    * train_mask, val_mask, and test_mask are used to indicate the index of each set of nodes.
    """
    names = ['features', 'graph', 'num_classes',
             'train_labels', 'val_labels', 'test_labels',
             'train_mask', 'val_mask', 'test_mask']

    objects = []
    for i in range(len(names)):
        with open("dataset/private_{}.pkl".format(names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    features, graph, num_classes, \
        train_labels, val_labels, test_labels, \
        train_mask, val_mask, test_mask = tuple(objects)

    return features, graph, num_classes, train_labels, val_labels, test_labels, train_mask, val_mask, test_mask


if __name__ == '__main__':
    features, graph, num_classes, \
        train_labels, val_labels, test_labels, \
        train_mask, val_mask, test_mask = load_data()

    # features = features.to(device)  # (19717, 500)
    # graph = graph.to(device)  # (19717, 19717)
    # train_labels = train_labels.to(device)  # (60)
    # val_labels = val_labels.to(device)  # (500)
    # train_mask = train_mask.to(device)  # (19717)
    # val_mask = val_mask.to(device)  # (19717)
    # test_mask = test_mask.to(device)  # (19717)
