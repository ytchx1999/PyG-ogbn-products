# GraphSAGE w/NS + C&S
This is an improvement of the  [(NeighborSampling (SAGE aggr))](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_sage.py)  model, using the C&S method. 

### ogbn-products

+ Check out the model： [(NeighborSampling (SAGE aggr))](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_sage.py) 

+ Check out the C&S method：[C&S](https://arxiv.org/abs/2010.13993)

#### Improvement Strategy：

+ add C&S method
+ add BatchNorm

#### Environmental Requirements

+ pytorch == 1.8.1
+ pytorch_geometric == 1.6.3
+ ogb == 1.3.0

#### Experiment Setup：

+ **Let the program run in the foreground.**

```bash
python sage_cs_mini.py
```

+ **Or let the program run in the background** and save the results to a log file.

```bash
nohup python sage_cs_mini.py > ./sage_cs_mini.log 2>&1 &
```

#### Detailed Hyperparameter:

```bash
num_layers = 3
hidden_dim = 256
dropout = 0.5
lr = 0.003
batch_size = 1024
sizes = [15, 10, 5]
runs = 10
epochs = 25
num_correction_layers = 100
correction_alpha = 0.8
num_smoothing_layers = 100
smoothing_alpha = 0.8
scale = 10.
A1 = 'DAD'
A2 = 'DAD'
```

#### Result:

```bash
All runs:
Highest Train: 97.15 ± 0.08
Highest Valid: 92.38 ± 0.07
  Final Train: 97.15 ± 0.08
   Final Test: 80.41 ± 0.22
```

| Model                | Test Accuracy   | Valid Accuracy  | Parameters | Hardware          |
| -------------------- | --------------- | --------------- | ---------- | ----------------- |
| GraphSAGE w/NS + C&S | 0.8041 ± 0.0022 | 0.9238 ± 0.0007 | 207919     | Tesla V100 (32GB) |

