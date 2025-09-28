import torch
import torch.nn as nn
from dgl.ops import edge_softmax
import dgl.function as fn
from dgl.utils import expand_as_pair
from .utils import create_activation

# =========================
# 自定义图注意力网络（GAT）主结构
# =========================
class GAT(nn.Module):
    def __init__(self,
                 n_dim,           # 输入节点特征维度
                 e_dim,           # 输入边特征维度
                 hidden_dim,      # 隐藏层特征维度
                 out_dim,         # 输出特征维度
                 n_layers,        # GAT层数
                 n_heads,         # 每层多头注意力头数
                 n_heads_out,     # 最后一层的多头数
                 activation,      # 激活函数类型
                 feat_drop,       # 特征dropout比例
                 attn_drop,       # 注意力dropout比例
                 negative_slope,  # LeakyReLU负斜率
                 residual,        # 是否使用残差连接
                 norm,            # 归一化方式
                 concat_out=False,# 是否拼接多头输出
                 encoding=False   # 是否为编码阶段，影响最后一层结构
                 ):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.gats = nn.ModuleList()  # 存放多层GATConv
        self.concat_out = concat_out
        # 创建最后一层的激活函数（仅编码阶段用）
        last_activation = create_activation(activation) if encoding else None
        # 创建最后一层的残差连接
        last_residual = (encoding and residual)
        # 创建最后一层的归一化
        last_norm = norm if encoding else None
        # 如果只有一层，直接创建GATConv
        if self.n_layers == 1:
            self.gats.append(GATConv(
                n_dim, e_dim, out_dim, n_heads_out, feat_drop, attn_drop, negative_slope,
                last_residual, norm=last_norm, concat_out=self.concat_out
            ))
        else:
            # 第一层
            self.gats.append(GATConv(
                n_dim, e_dim, hidden_dim, n_heads, feat_drop, attn_drop, negative_slope,
                residual, create_activation(activation),
                norm=norm, concat_out=self.concat_out
            ))
            # 中间层
            for _ in range(1, self.n_layers - 1):
                self.gats.append(GATConv(
                    hidden_dim * self.n_heads, e_dim, hidden_dim, n_heads,
                    feat_drop, attn_drop, negative_slope,
                    residual, create_activation(activation),
                    norm=norm, concat_out=self.concat_out
                ))
            # 最后一层
            self.gats.append(GATConv(
                hidden_dim * self.n_heads, e_dim, out_dim, n_heads_out,
                feat_drop, attn_drop, negative_slope,
                last_residual, last_activation, norm=last_norm, concat_out=self.concat_out
            ))
        self.head = nn.Identity()  # 分类头，默认不变换

    def forward(self, g, input_feature, return_hidden=False):
        """
        g: DGLGraph，输入图
        input_feature: 节点特征
        return_hidden: 是否返回所有层的输出
        """
        h = input_feature
        hidden_list = []  # 用于收集每层输出
        for layer in range(self.n_layers):
            h = self.gats[layer](g, h)  # 依次通过每一层GATConv
            hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list  # 返回最后一层和所有层输出
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        # 更换分类头
        self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)

# =========================
# 单层图注意力卷积（GATConv）
# =========================
class GATConv(nn.Module):
    def __init__(self,
                 in_dim,           # 输入特征维度
                 e_dim,            # 边特征维度
                 out_dim,          # 输出特征维度
                 n_heads,          # 多头数
                 feat_drop=0.0,    # 特征dropout
                 attn_drop=0.0,    # 注意力dropout
                 negative_slope=0.2, # LeakyReLU负斜率
                 residual=False,   # 是否残差
                 activation=None,  # 激活函数
                 allow_zero_in_degree=False, # 是否允许入度为0
                 bias=True,        # 是否加偏置
                 norm=None,        # 归一化方式
                 concat_out=True   # 是否拼接多头输出
                 ):
        super(GATConv, self).__init__()
        self.n_heads = n_heads
        self.src_feat, self.dst_feat = expand_as_pair(in_dim)
        self.edge_feat = e_dim
        self.out_feat = out_dim
        self.allow_zero_in_degree = allow_zero_in_degree
        self.concat_out = concat_out

        # 节点特征线性变换
        if isinstance(in_dim, tuple):
            # 如果是异构图，则对源节点和目标节点分别进行线性变换
            self.fc_node_embedding = nn.Linear(
                self.src_feat, self.out_feat * self.n_heads, bias=False) # 这一层是多余的，后续没有用到，只用到了fc_src和fc_dst
            self.fc_src = nn.Linear(self.src_feat, self.out_feat * self.n_heads, bias=False)
            self.fc_dst = nn.Linear(self.dst_feat, self.out_feat * self.n_heads, bias=False)
        else:
            # 如果是同构图，则对节点特征进行线性变换
            self.fc_node_embedding = nn.Linear(
                self.src_feat, self.out_feat * self.n_heads, bias=False)
            self.fc = nn.Linear(self.src_feat, self.out_feat * self.n_heads, bias=False)
        # 边特征线性变换
        self.edge_fc = nn.Linear(self.edge_feat, self.out_feat * self.n_heads, bias=False)
        # 初始化注意力参数
        # nn.Parameter会将张量标记为模型参数，会被更新，可被保存
        #NOTE：注意力参数的计算：e_ij = LeakyReLU(a_h^T * W*h_i + a_t^T * W*h_j + a_e^T * edge_feat)
        # 这里的attn对应的就是a_h, a_t, a_e
        self.attn_h = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat))) # 节点源
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat))) # 边
        self.attn_t = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat))) # 节点目标
        # dropout层，feat_drop是丢弃概率，会将输入向量中的一部分随机置为0
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        # LeakyReLU激活函数，negative_slope是负斜率
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        # 如果bias为True，则创建一个可训练的偏置参数，否则注册一个None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))
        else:
            # 使用 register_buffer 注册一个 固定属性 bias = None，避免self.bias 不存在引发错误
            self.register_buffer('bias', None)
        # 残差连接
        if residual:
            # 如果维度不匹配，则创建一个线性变换，否则直接使用恒等映射
            if self.dst_feat != self.n_heads * self.out_feat:
                self.res_fc = nn.Linear(
                    self.dst_feat, self.n_heads * self.out_feat, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.norm = norm
        # 归一化
        if norm is not None:
            self.norm = norm(self.n_heads * self.out_feat)

    def reset_parameters(self):
        # 参数初始化
        gain = nn.init.calculate_gain('relu')
        # 使用xavier_normal_初始化权重，gain是缩放因子
        nn.init.xavier_normal_(self.edge_fc.weight, gain=gain)
        if hasattr(self, 'fc'):
            # 如果有fc说明是同构图，初始化一个矩阵即可
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            # 如果是异构图，则需要初始化两个矩阵
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_h, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.attn_t, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            # 如果res_fc是线性变换，则初始化权重（恒等映射不需要初始化）
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
    # 设置是否允许入度为0的节点
    def set_allow_zero_in_degree(self, set_value):
        self.allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        """
        graph: DGLGraph，输入图
        feat: 节点特征
        get_attention: 是否返回注意力权重
        """
        edge_feature = graph.edata['attr']
        # 创建一个局部作用域，避免对外部 graph 的属性产生永久性修改。
        with graph.local_scope():
            # 节点特征dropout
            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1] # 去掉最后一个维度（特征）
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    # 如果不是异构图（结点特征维度是相同的）
                    # 先用fc进行一次映射得到输入向量，此时每个结点的特征维度为out_feat * n_heads
                    # 然后reshape为(batch_size, n_heads, out_feat)
                    # 即把单个结点的特征维度扩展为n_heads个，每个特征维度为out_feat，也就是一个头处理的特征维度数
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self.n_heads, self.out_feat)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self.n_heads, self.out_feat)
                else:
                    # 如果是异构图就要用不同的线性层做映射
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self.n_heads, self.out_feat)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self.n_heads, self.out_feat)
            else:
                # 不是采样的子图的话就所有的结点都会参与聚合
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self.n_heads, self.out_feat)
                # 如果是子图，则需要对特征进行裁剪
                # create_block 或 dataloader.sample() 返回的是子图对象，是对全图的采样
                if graph.is_block:
                    # DGL 在构建 block 图的时候，会将目标节点放在源节点 tensor 的前部，因此可以通过这种方式方便地提取出来。
                    # 此时feat_dst是目标结点经过多头划分后的特征
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    # h_dst是目标结点的特征
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    # (graph.number_of_dst_nodes(),) 是一个元组，只有一个元素
                    # 将dst_prefix_shape的第一个维度替换成目标结点的数量
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            # 边特征线性变换
            edge_prefix_shape = edge_feature.shape[:-1]
            # 对于每个结点，每个头会计算一个注意力分数（标量）
            eh = (feat_src * self.attn_h).sum(-1).unsqueeze(-1)  # 源节点注意力 a_h^T * W*h_i
            et = (feat_dst * self.attn_t).sum(-1).unsqueeze(-1)  # 目标节点注意力 a_t^T * W*h_j

            graph.srcdata.update({'hs': feat_src, 'eh': eh})
            graph.dstdata.update({'et': et})

            feat_edge = self.edge_fc(edge_feature).view(
                *edge_prefix_shape, self.n_heads, self.out_feat)
            ee = (feat_edge * self.attn_e).sum(-1).unsqueeze(-1)  # 边注意力  a_e^T * edge_feat

            graph.edata.update({'ee': ee})
            # 这两行相当于计算eh+ee+et，存在e中
            graph.apply_edges(fn.u_add_e('eh', 'ee', 'ee'))  # 源节点+边
            graph.apply_edges(fn.e_add_v('ee', 'et', 'e'))   # +目标节点
            """
            graph.apply_edges(fn.u_add_v('eh', 'et', 'e'))
            """
            # 将以上的权重过激活函数
            e = self.leaky_relu(graph.edata.pop('e'))  # LeakyReLU激活
            # 将以上的权重进行softmax归一化
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))  # softmax归一化+dropout
            # 消息传递
            # 对每条边，将源节点特征'hs'与边的注意力系数'a'相乘，得到消息'm'，shape为[num_edges, n_heads, out_feat]
            # 然后对每个目标节点，将所有入边的消息'm'按边聚合（求和），结果存到目标节点的'hs'属性中，shape为[n_heads, out_feat]
            graph.update_all(fn.u_mul_e('hs', 'a', 'm'),
                             fn.sum('m', 'hs'))

            # 将所有目标节点的新特征'hs'取出，保证shape为[num_dst_nodes, n_heads, out_feat]
            # 这里view主要是防止shape被意外改变，确保后续处理不会出错
            rst = graph.dstdata['hs'].view(-1, self.n_heads, self.out_feat)

            # 将更新后的结点特征加上偏置项
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self.n_heads, self.out_feat)

            # 残差连接
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self.out_feat)
                rst = rst + resval

            # 多头拼接或聚合
            if self.concat_out:
                rst = rst.flatten(1)
            else:
                # mean聚合会将特征维度降为原来的1/n_heads
                rst = torch.mean(rst, dim=1)

            # 归一化
            if self.norm is not None:
                rst = self.norm(rst)

            # 激活函数
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']  # 返回特征和注意力权重
            else:
                return rst  # 只返回特征
