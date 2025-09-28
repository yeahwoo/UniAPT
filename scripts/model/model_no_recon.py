from .gat import GAT
from .utils import create_norm, sce_loss


from functools import partial
from itertools import chain

import torch
import torch.nn as nn
import dgl
import random


class MYModel(nn.Module):
    def __init__(
        self,
        n_dim,
        e_dim,
        hidden_dim,
        n_layers,
        n_heads,
        activation,
        feat_drop,
        negative_slope,
        residual,
        norm,
        mask_rate=0.5,
        loss_fn="sce",
        alpha_l=2,
    ):
        super(MYModel, self).__init__()
        self._mask_rate = mask_rate
        self._output_hidden_size = hidden_dim
        self.recon_loss = nn.BCELoss(reduction="mean")

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant_(m.bias, 0)

        self.edge_recon_fc = nn.Sequential(
            nn.Linear(hidden_dim * n_layers * 2, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.edge_recon_fc.apply(init_weights)

        assert hidden_dim % n_heads == 0
        enc_num_hidden = hidden_dim // n_heads
        enc_nhead = n_heads

        dec_in_dim = hidden_dim
        dec_num_hidden = hidden_dim

        # build encoder
        self.encoder = GAT(
            n_dim=n_dim,
            e_dim=e_dim,
            hidden_dim=enc_num_hidden,
            out_dim=enc_num_hidden,
            n_layers=n_layers,
            n_heads=enc_nhead,
            n_heads_out=enc_nhead,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=True,
        )

        # build decoder for attribute prediction
        self.decoder = GAT(
            n_dim=dec_in_dim,
            e_dim=e_dim,
            hidden_dim=dec_num_hidden,
            out_dim=n_dim,
            n_layers=1,
            n_heads=n_heads,
            n_heads_out=1,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=False,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_dim))
        self.encoder_to_decoder = nn.Linear(
            dec_in_dim * n_layers, dec_in_dim, bias=False
        )

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        """
        设置损失函数
        目前支持SCE（Supervised Contrastive Encoding）损失
        使用partial固定损失函数的alpha参数
        """
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, mask_rate=0.3):
        """
        对图进行随机掩码
        随机打乱节点顺序
        根据掩码率确定需要掩码的节点数量
        将打乱顺序后的节点顺序中前num_mask_nodes个节点设置为掩码节点
        返回掩码节点和未掩码节点
        """
        new_g = g.clone()
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=g.device)  # 打乱节点顺序

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]  # 掩码节点
        keep_nodes = perm[num_mask_nodes:]  # 未掩码节点

        new_g.ndata["attr"][
            mask_nodes
        ] = self.enc_mask_token  # 掩码向量：[1, n_dim]，可学习

        return new_g, (mask_nodes, keep_nodes)

    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        """
        计算损失
        包括特征重建损失和结构重建损失
        """
        # Feature Reconstruction
        pre_use_g, (mask_nodes, keep_nodes) = self.encoding_mask_noise(
            g, self._mask_rate
        )  # pre_use_g：经过掩码的图
        pre_use_x = pre_use_g.ndata["attr"].to(pre_use_g.device)
        use_g = pre_use_g
        enc_rep, all_hidden = self.encoder(
            use_g, pre_use_x, return_hidden=True
        )  # 返回隐藏层
        enc_rep = torch.cat(all_hidden, dim=1)  # 拼接隐藏层（保留多跳结构信息）
        rep = self.encoder_to_decoder(enc_rep)

        recon = self.decoder(pre_use_g, rep)
        x_init = g.ndata["attr"][mask_nodes]  # 掩码结点的原始特征
        x_rec = recon[mask_nodes]  # 掩码结点的重建特征
        loss = self.criterion(x_rec, x_init)

        # Structural Reconstruction
        threshold = min(10000, g.num_nodes())

        negative_edge_pairs = dgl.sampling.global_uniform_negative_sampling(
            g, threshold
        )  # 负采样，采样threshold个不存在的边
        positive_edge_pairs = random.sample(
            range(g.number_of_edges()), threshold
        )  # 随机采样threshold个存在的边的索引
        positive_edge_pairs = (
            g.edges()[0][positive_edge_pairs],
            g.edges()[1][positive_edge_pairs],
        )  # 将采样到的边转换为源节点和目标节点
        sample_src = enc_rep[
            torch.cat([positive_edge_pairs[0], negative_edge_pairs[0]])
        ].to(g.device)
        sample_dst = enc_rep[
            torch.cat([positive_edge_pairs[1], negative_edge_pairs[1]])
        ].to(g.device)
        y_pred = self.edge_recon_fc(
            torch.cat([sample_src, sample_dst], dim=-1)
        ).squeeze(-1)
        y = torch.cat(
            [
                torch.ones(len(positive_edge_pairs[0])),
                torch.zeros(len(negative_edge_pairs[0])),
            ]
        ).to(g.device)

        return loss

    def embed(self, g):
        """
        提取图的特征
        """
        x = g.ndata["attr"].to(g.device)
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        """
        返回编码器参数
        """
        return self.encoder.parameters()

    @property
    def dec_params(self):
        """
        返回解码器参数
        """
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
