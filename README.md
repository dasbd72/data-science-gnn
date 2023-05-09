# Node classification

## How to run

### Parameters

* --use_gpu
  * Train with cuda gpu
* --tuning
  * Tune using optuna
* --es_iter
  * Set if wants to early stop, usually use with --tuning
* --epochs
  * Training iterations
* --node2vec
  * adds new features
* --model
  * Supports
    * GCN
    * SAGE
    * GAT
    * DotGAT
    * Grace
    * SSP
  * Best is Grace

### Examples

To tune variables

```bash=
python3 train.py --use_gpu --tuning --es_iter 50 --epochs 500 --model "Grace"
```

To train and validate

```bash=
python3 train.py --use_gpu --es_iter 50 --epochs 500 --model "Grace"
```

