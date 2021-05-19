# GraphSAGE w/NS + C&S + node2vec
This is an improvement of the  [(NeighborSampling (SAGE aggr))](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_sage.py)  model, using the C&S method and node2vec embedding. 

[**Check out the OGBn-Products LeaderBoard!**](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-products)

Our paper is available at [https://arxiv.org/pdf/2105.08330.pdf](https://arxiv.org/pdf/2105.08330.pdf).

### ogbn-products

+ Check out the model： [(NeighborSampling (SAGE aggr))](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_products_sage.py) 
+ Check out the C&S method：[C&S](https://arxiv.org/abs/2010.13993)
+ Check out node2vec model：[node2vec](https://arxiv.org/abs/1607.00653)

#### Improvement Strategy：

+ add C&S method
+ add BatchNorm
+ add node2vec embedding

#### Environmental Requirements

+ pytorch == 1.8.1
+ pytorch_geometric == 1.6.3
+ ogb == 1.3.0

#### Experiment Setup：

1. Generate node2vec embeddings, which save in `embedding.pt`

   ```bash
   python node2vec_products.py
   ```

2. Run the real model

   + **Let the program run in the foreground.**

   ```bash
   python sage_cs_em.py
   ```

   + **Or let the program run in the background** and save the results to a log file.

   ```bash
   nohup python sage_cs_em.py > ./sage_cs_em.log 2>&1 &
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
epochs = 20
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
Highest Train: 97.13 ± 0.07
Highest Valid: 92.38 ± 0.06
  Final Train: 97.13 ± 0.07
   Final Test: 81.54 ± 0.50
```

| Model                           | Test Accuracy   | Valid Accuracy  | Parameters | Hardware          |
| ------------------------------- | --------------- | --------------- | ---------- | ----------------- |
| GraphSAGE w/NS + C&S + node2vec | 0.8154 ± 0.0050 | 0.9238 ± 0.0006 | 103983     | Tesla V100 (32GB) |

