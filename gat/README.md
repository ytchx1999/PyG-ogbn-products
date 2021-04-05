# GAT w/NS + C&S
This is an improvement of the  [(GAT with NeighborSampling)](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_gat.py)  model, using the C&S method. 

### ogbn-products

+ Check out the model： [(GAT with NeighborSampling)](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_gat.py) 

+ Check out the C&S method：[C&S](https://arxiv.org/abs/2010.13993)

#### Improvement Strategy：

+ add C&S method

#### Environmental Requirements

+ pytorch == 1.8.1
+ pytorch_geometric == 1.6.3
+ ogb == 1.3.0

#### Experiment Setup：

+ **Let the program run in the foreground.**

```bash
python gat_cs_mini.py
```

+ **Or let the program run in the background** and save the results to a log file.

```bash
nohup python gat_cs_mini.py > ./gat_cs_mini.log 2>&1 &
```

#### Detailed Hyperparameter:

```bash
num_layers = 3
hidden_dim = 128
heads = 4
dropout = 0.5
lr = 0.001
batch_size = 512
sizes = [10, 10, 10]
runs = 10
epochs = 100
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
Highest Train: 97.28 ± 0.06
Highest Valid: 92.63 ± 0.08
  Final Train: 97.28 ± 0.06
   Final Test: 80.92 ± 0.37
```

| Model          | Test Accuracy   | Valid Accuracy  | Parameters | Hardware          |
| -------------- | --------------- | --------------- | ---------- | ----------------- |
| GAT w/NS + C&S | 0.8092 ± 0.0037 | 0.9263 ± 0.0008 | 753622     | Tesla V100 (32GB) |

