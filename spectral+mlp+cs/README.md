# Spec-MLP-Wide + C&S_pyg
This is a PyG (Pytorch Geometric) implement of the  [MLP + C&S](https://github.com/CUAI/CorrectAndSmooth)  model (using spectral embeddings), which is more simple and streamlined.

### ogbn-products

+ Check out the C&S paper：[C&S](https://arxiv.org/abs/2010.13993)

#### Environmental Requirements

+ pytorch == 1.8.1
+ pytorch_geometric == 1.7.2
+ ogb == 1.3.1
+ julia == 1.0.5
+ pyjulia == 0.5.6

[How to install Julia and PyJulia.](https://pyjulia.readthedocs.io/en/latest/installation.html) (generate spectral embeddings, more details can be found in C&S paper.)

#### Experiment Setup：
+ mkdir
```bash
mkdir embeddings/
mkdir outpus/
```

+ **Let the program run in the foreground.**

```bash
# first execute
python3 mlp_cs.py --device 0 --use_embed
# use cached embeddings
python3 mlp_cs.py --device 0 --use_embed --use_cached
```

+ **Or let the program run in the background** and save the results to a log file.

```bash
nohup python3 mlp_cs.py --device 0 --use_embed > ./outputs/mlp_cs.log 2>&1 &
# use cached embeddings
nohup python3 mlp_cs.py --device 0 --use_embed --use_cached > ./outputs/mlp_cs.log 2>&1 &
```

#### Detailed Hyperparameter:

```python
Namespace(device=2, dropout=0.5, epochs=200, hidden_channels=512, lr=0.01, num_layers=3, runs=10, use_cached=True, use_embed=True)

num_layers = 3
hidden_dim = 512
dropout = 0.5
lr = 0.01
runs = 10
epochs = 200
num_correction_layers = 50
correction_alpha = 1.0
num_smoothing_layers = 50
smoothing_alpha = 0.8
scale = 15.
A1 = 'DAD'
A2 = 'DA'
```

#### Result:

```bash
All runs:
Highest Train: 95.50 ± 0.06
Highest Valid: 91.32 ± 0.10
  Final Train: 95.50 ± 0.06
   Final Test: 84.51 ± 0.06
```

| Model                | Test Accuracy   | Valid Accuracy  | Parameters | Hardware          |
| -------------------- | --------------- | --------------- | ---------- | ----------------- |
| Spec-MLP-Wide + C&S_pyg | 0.8451 ± 0.0006 | 0.9132 ± 0.0010 | 406063     | Tesla V100 (32GB) |

