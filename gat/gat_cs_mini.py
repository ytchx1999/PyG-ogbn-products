import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
import torch.optim as optim
import numpy as np

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, JumpingKnowledge
from torch_geometric.data import NeighborSampler
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
from tqdm import tqdm

from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor

from CorrectAndSmooth import CorrectAndSmooth

# 加载数据集
# dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/', transform=T.ToSparseTensor())
dataset = PygNodePropPredDataset(name='ogbn-products', root='./products/')
print(dataset, flush=True)
data = dataset[0]
print(data, flush=True)

# 划分数据集
split_idx = dataset.get_idx_split()

# 定义评估器
# evaluator = Evaluator(name='ogbn-arxiv')
evaluator = Evaluator(name='ogbn-products')

train_idx = split_idx['train']
test_idx = split_idx['test']

# 邻居采样
# 用于训练
train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                               sizes=[10, 10, 10], batch_size=512,
                               shuffle=True, num_workers=12)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=1024, shuffle=False,
                                  num_workers=12)


class GAT(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, num_layers=3, heads=4):
        super(GAT, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GATConv(dataset.num_features, hidden_channels, heads))
        self.bns.append(nn.BatchNorm1d(heads * hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(heads * hidden_channels, hidden_channels, heads))
            self.bns.append(nn.BatchNorm1d(heads * hidden_channels))

        self.convs.append(GATConv(heads * hidden_channels, dataset.num_classes, heads, concat=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(nn.Linear(dataset.num_features, hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.skips.append(nn.Linear(hidden_channels * heads, hidden_channels * heads))

        self.skips.append(nn.Linear(hidden_channels * heads, dataset.num_classes))

    # def reset_masks(self):
    #     self.masks = [None] * (self.num_layers - 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (adj_t, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), adj_t)
            x = x + self.skips[i](x_target)
            if i != self.num_layers - 1:
                x = self.bns[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                adj_t, _, size = adj.to(device)
                total_edges += adj_t.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), adj_t)
                x = x + self.skips[i](x_target)

                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = F.elu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


# 实例化模型
model = GAT(dataset=dataset, hidden_channels=128, num_layers=3, heads=4)
print(model, flush=True)

# 转换为cpu或cuda格式
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, flush=True)
model.to(device)
data = data.to(device)
# data.adj_t = data.adj_t.to_symmetric()  # 对称归一化 yichu
# train_idx = train_idx.to(device)

x, y = data.x.to(device), data.y.squeeze().to(device)
train_idx = split_idx['train'].to(device)
val_idx = split_idx['valid'].to(device)
test_idx = split_idx['test'].to(device)
x_train, y_train = x[train_idx], y[train_idx]


def process_adj(data):
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index

    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    return adj


DAD = process_adj(data).to(device)

# adj_t = data.adj_t.to(device)
# deg = adj_t.sum(dim=1).to(torch.float)
# deg_inv_sqrt = deg.pow_(-0.5)
# deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
# DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
# DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t
# AD = adj_t * deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1)

# 定义损失函数和优化器
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 定义训练函数(minibatch)
def train():
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(train_idx.size(0))

    return loss, approx_acc


# 定义测试函数
@torch.no_grad()
def test(out=None):
    model.eval()

    out = model.inference(x) if out is None else out
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc, out


# 程序入口
if __name__ == '__main__':
    runs = 10
    logger = Logger(runs)

    for run in range(runs):
        print(sum(p.numel() for p in model.parameters()), flush=True)

        print('', flush=True)
        print(f'Run {run + 1:02d}:', flush=True)
        print('', flush=True)

        model.reset_parameters()

        best_val_acc = 0

        for epoch in range(100):
            loss, acc = train()
            # print('Epoch {:03d} train_loss: {:.4f}'.format(epoch, loss))
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}', flush=True)

            if (epoch + 1) > 50 and (epoch + 1) % 10 == 0:

                train_acc, val_acc, test_acc, out = test()
                result = (train_acc, val_acc, test_acc)
                # print(f'Train: {train_acc:.4f}, Val: {valid_acc:.4f}, 'f'Test: {test_acc:.4f}')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    y_soft = out.softmax(dim=-1).to(device)

                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * val_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%', flush=True)

            # logger.add_result(run, result)

        # post = CorrectAndSmooth(num_correction_layers=50, correction_alpha=1.0,
        #                         num_smoothing_layers=50, smoothing_alpha=0.8,
        #                         autoscale=False, scale=20.)

        post = CorrectAndSmooth(num_correction_layers=100, correction_alpha=0.8,
                                num_smoothing_layers=100, smoothing_alpha=0.8,
                                autoscale=False, scale=10.)

        print('Correct and smooth...', flush=True)
        # y_soft = post.correct(y_soft, y_train, train_idx, DAD)
        # y_soft = post.smooth(y_soft, y_train, train_idx, DA)
        y_soft = post.correct(y_soft, y_train, train_idx, DAD)
        y_soft = post.smooth(y_soft, y_train, train_idx, DAD)
        print('Done!', flush=True)
        train_acc, val_acc, test_acc, _ = test(y_soft)
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}', flush=True)

        result = (train_acc, val_acc, test_acc)
        logger.add_result(run, result)

    logger.print_statistics()
