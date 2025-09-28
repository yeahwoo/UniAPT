from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def create_norm(name):
    """创建归一化层"""
    name = name.lower()
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type=name)
    else:
        return None

def create_activation(name):
    """创建激活函数"""
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

# 1、归一化
# 2、对于每个样本：x（重建特征）和y（原始特征）逐元素相乘
# 3、loss = 所有样本相加/样本数
def sce_loss(x, y, alpha=3):
    """计算基于余弦相似度的损失函数"""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    # 计算余弦相似度
    # alpha参数控制损失的敏感性，可以放大差异
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha) 
    loss = loss.mean()
    return loss

def transform_graph(g, node_feature_dim, edge_feature_dim):
    """将特征转为one-hot编码"""
    new_g = g.clone()
    new_g.ndata["attr"] = F.one_hot(g.ndata["type"].view(-1), num_classes=node_feature_dim).float() # 将8个结点类型转成8维向量
    new_g.edata["attr"] = F.one_hot(g.edata["type"].view(-1), num_classes=edge_feature_dim).float() # 将4种边类型转成4维向量
    return new_g
    
def transform_graph_with_time(g, node_feature_dim, edge_feature_dim, time_dim):
    """将节点/边类型做one-hot，并将边的time字段进行TimeEncode，再拼接到边特征上"""
    new_g = g.clone()

    # 节点 one-hot 特征
    new_g.ndata["attr"] = F.one_hot(
        g.ndata["type"].view(-1), num_classes=node_feature_dim
    ).float()

    # 边 one-hot 特征
    edge_onehot = F.one_hot(
        g.edata["type"].view(-1), num_classes=edge_feature_dim
    ).float()

    # 取出时间字段 [E] -> [E,1]
    time_values = g.edata["time"].float().view(-1, 1)

    # 创建时间编码器并编码
    time_encoder = TimeEncode(expand_dim=time_dim)
    with torch.no_grad():  # 这里作为特征提取，不参与梯度
        time_encoded = time_encoder(time_values)  # [E,1,time_dim]
        time_encoded = time_encoded.squeeze(1)    # [E,time_dim]

    # 拼接边特征
    new_g.edata["attr"] = torch.cat([edge_onehot, time_encoded], dim=-1)

    return new_g


# 归一化层
class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


# 图池化层
class Pooling(nn.Module):
    def __init__(self, pooler):
        super(Pooling, self).__init__()
        self.pooler = pooler

    # graph: 图结构
    # feat: 结点特征
    # n_types: 结点类型数量
    def forward(self, graph, feat, n_types=None):
        # Implement node type-specific pooling
        # graph.local_scope() 创建一个临时上下文环境 ，在这个环境中对图数据的任何修改都 不会影响到原始的图对象
        with graph.local_scope():
            # 不指定结点类型，直接对所有结点特征进行池化
            if not n_types:
                if self.pooler == 'mean':
                    return feat.mean(0, keepdim=True)
                elif self.pooler == 'sum':
                    return feat.sum(0, keepdim=True)
                elif self.pooler == 'max':
                    return feat.max(0, keepdim=True)
                else:
                    raise NotImplementedError
            # 指定结点类型，按照结点类型分别池化，最后拼接
            else:
                result = []
                for i in range(n_types):
                    mask = (graph.ndata['type'] == i)
                    if not mask.any():
                        result.append(torch.zeros((1, feat.shape[-1]), device=feat.device))
                    elif self.pooler == 'mean':
                        result.append(feat[mask].mean(0, keepdim=True))
                    elif self.pooler == 'sum':
                        result.append(feat[mask].sum(0, keepdim=True))
                    elif self.pooler == 'max':
                        result.append(feat[mask].max(0, keepdim=True))
                    else:
                        raise NotImplementedError
                result = torch.cat(result, dim=-1)
                return result


# 早停
class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-4, threshold=None):
        self.patience = patience
        self.min_delta = min_delta
        self.start_threshold = threshold
        self.counter = 0
        self.best_loss = float('inf')
        self.started = threshold is None  # 如果没有设置门槛，则立即启动早停机制

    def should_stop(self, current_loss):
        if not self.started:
            if current_loss < self.start_threshold:
                self.started = True
            else:
                return False  # 还没达到起始门槛，不能早停

        if self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# 时间编码器
class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)
        #torch.nn.init.xavier_normal_(self.dense.weight)
        
    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        
        harmonic = torch.cos(map_ts)
        return harmonic #self.dense(harmonic)


# def test_early_stopper():
#     stopper = EarlyStopper(patience=3, min_delta=0.01, start_threshold=0.98)
#     # 构造一个模拟的loss序列
#     losses = [1.0, 0.99, 0.985, 0.981, 0.979, 0.978, 0.9775, 0.9774, 0.9773]
#     print("Testing EarlyStopper:")
#     for i, loss in enumerate(losses):
#         stop = stopper.should_stop(loss)
#         print(f"Epoch {i}, Loss: {loss:.4f}, Should Stop: {stop}")
#         if stop:
#             print(f"==> Early stopping triggered at epoch {i}")
#             break


# if __name__ == "__main__":
#     test_early_stopper()