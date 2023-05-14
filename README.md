# Node classification

## How to run

### Requirements

directory `./ouptuts` should exists

```bash=
mkdir -p outputs
```

dataset should be in `./dataset`

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
  * adds random walk features
* --model
  * Supports
    * GCN
    * SAGE
    * Grace
    * SSP
  * Best model is Grace

### Examples

To tune variables (also generates ./outputs/output_\[val_acc\].csv)

```bash=
python3 train.py --use_gpu --node2vec --tuning --epochs 1500 --model "Grace"
```

To train, validate then predict

```bash=
python3 train.py --use_gpu --node2vec --epochs 1500 --model "Grace"
```

At last, vote on output files in voting.ipynb