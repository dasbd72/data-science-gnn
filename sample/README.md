# Data Science HW3

## TODO

* model.py
  * implement your own node classification model
* train.py
  * setup your model

## You can create the environment with anaconda

```
conda create --name hw3 pip -y
conda activate hw3
```

## Install packages

* scipy, networkx

```
pip install scipy networkx
```

* Install dgl
  * <https://www.dgl.ai/pages/start.html>

* Install pytorch
  * <https://pytorch.org/get-started/locally/>

## Run sample code

```python
python3 train.py --es_iters 30 --epochs 300 --use_gpu
```

## Dataset

* Unknown graph data
  * Label Split:
    * Train: 60, Valid: 500, Test: 1000
* File name description

```
  dataset
  │   ├── private_features.pkl # node feature
  │   ├── private_graph.pkl # graph edges
  │   ├── private_num_classes.pkl # number of classes
  │   ├── private_test_labels.pkl # X
  │   ├── private_test_mask.pkl # nodes indices of testing set
  │   ├── private_train_labels.pkl # nodes labels of training set
  │   ├── private_train_mask.pkl # nodes indices of training set
  │   ├── private_val_labels.pkl # nodes labels of validation set
  │   └── private_val_mask.pkl # nodes indices of validation set
```
