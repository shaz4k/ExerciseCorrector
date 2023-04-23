import math
from torch import nn
from torch.nn.parameter import Parameter
import torch
import numpy as np
from utils.GraphLayers import GraphConvolution
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat=True, dropout=0.6, alpha=0.2):
        """
        :param in_features: no. input features per node
        :param out_features: no. output features per node
        :param n_heads:
        :param is_concat:whether the multi-head results should be concatenated or averaged
        :param dropout:
        :param alpha: negative slope for leaky ReLU activation
        """
        super(GraphAttentionLayer, self).__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        # dims per head
        if is_concat:
            # for multiple heads
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            # avg multiple heads
            self.n_hidden = out_features

        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)  # embeddings
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)  # attention
        self.activation = nn.LeakyReLU(negative_slope=alpha)  # actf for attention scores
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor, batch_size: int):
        # h.shape = [nodes_n, in_features]
        # adj_mat = [nodes_n, nodes_n, n_heads] --> same for each head so [nodes_n, nodes_n, 1]

        n_nodes = h.shape[1]
        g = self.linear(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)  # embed and split for each head

        # Calculate attention scores
        g_repeat = g.repeat(1, n_nodes, 1, 1)  # [g1,...,gN,g1, ...gN, etc.]
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=1)  # [g1, g1,..., g2, ...g2]
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(batch_size, n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)

        # Check dims
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == n_nodes
        assert adj_mat.shape[3] == 1 or adj_mat.shape[3] == self.n_heads

        e = e.masked_fill(adj_mat == 0, float('-inf'))  # masked score based on adjacency matrix
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('bijh,bjhf->bihf', a, g)

        # Concatenate the heads
        if self.is_concat:
            attn_res = attn_res.reshape(batch_size, n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            attn_res = attn_res.mean(dim=2)  # [n_nodes, self.n_hidden]

        if batch_size == 1:
            attn_res = attn_res.squeeze(0)
        return attn_res


class GAT_Block(nn.Module):
    def __init__(self, in_features: int, n_heads: int, dropout: float, nodes_n: int):
        super(GAT_Block, self).__init__()
        self.in_features = in_features

        self.layer1 = GraphAttentionLayer(in_features, in_features, n_heads, is_concat=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(nodes_n * in_features)

        self.layer2 = GraphAttentionLayer(in_features, in_features, n_heads, is_concat=True, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(nodes_n * in_features)

        # maybe have layers of concat + concat and then mean?
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        if len(x.shape) == 2:
            n, f = x.shape
            x = x.unsqueeze(0)
        b, _, _ = x.shape
        y = self.layer1(x, adj_mat, b)
        if len(y.shape) == 3:
            b, n, f = y.shape
        else:
            b = 1
            n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act(y)
        y = self.dropout(y)

        y = self.layer2(y, adj_mat, b)
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act(y)
        y = self.dropout(y)
        y = y + x
        return y


class GAT(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, n_heads: int, nodes_n=57, num_stage=2, is_concat=True,
                 dropout=0.5):
        super(GAT, self).__init__()
        self.num_stage = num_stage
        self.ga1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(nodes_n * n_hidden)

        self.gabs = []
        for i in range(num_stage):
            self.gabs.append(GAT_Block(n_hidden, n_heads=n_heads, dropout=dropout, nodes_n=57))
        self.gabs = nn.ModuleList(self.gabs)
        self.ga2 = GraphAttentionLayer(n_hidden, in_features, n_heads=1, is_concat=False, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        # self.activation = nn.ReLU()
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, adj_mat):
        if len(x.shape) == 2:
            x.unsqueeze(0)
        b, _, _ = x.shape
        y = self.ga1(x, adj_mat, b)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.activation(y)
        y = self.dropout(y)

        for i in range(self.num_stage):
            y = self.gabs[i](y, adj_mat, b)

        y = self.ga2(y, adj_mat, b)
        # y = self.activation(y)
        # y = self.dropout(y)
        return y


class GraphAttentionLayerV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
        super(GraphAttentionLayerV2, self).__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weight = share_weights
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        n_batch, n_nodes = h.shape[0], h.shape[1]
        g_l = self.linear_l(h).view(n_batch, n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_batch, n_nodes, self.n_heads, self.n_hidden)

        g_l_repeat = g_l.unsqueeze(1).expand(n_batch, n_nodes, n_nodes, self.n_heads, self.n_hidden)
        g_r_repeat_interleave = g_r.unsqueeze(2).expand(n_batch, n_nodes, n_nodes, self.n_heads, self.n_hidden)

        g_sum = g_l_repeat + g_r_repeat_interleave
        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)

        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_batch
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == n_nodes
        assert adj_mat.shape[3] == 1 or adj_mat.shape[3] == self.n_heads

        e = e.masked_fill(adj_mat == 0, float('-inf'))
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('bijh,bjhf->bihf', a, g_r)

        if self.is_concat:
            attn_res = attn_res.reshape(n_batch, n_nodes, self.n_heads * self.n_hidden)
        else:
            attn_res = attn_res.mean(dim=2)
        return attn_res


class GATv2(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, n_heads: int, nodes_n=57, is_concat=True,
                 dropout=0.5, share_weights=True):
        super(GATv2, self).__init__()
        self.input_linear = nn.Linear(in_features, in_features)
        self.bin = nn.BatchNorm1d(nodes_n, in_features)
        self.ga1 = GraphAttentionLayerV2(in_features, n_hidden, n_heads,
                                         is_concat=True, dropout=dropout, share_weights=share_weights)
        self.bn1 = nn.BatchNorm1d(nodes_n * n_hidden)

        self.ga2 = GraphAttentionLayerV2(n_hidden, n_hidden, n_heads,
                                         is_concat=True, dropout=dropout, share_weights=share_weights)
        self.bn2 = nn.BatchNorm1d(nodes_n * n_hidden)

        self.ga3 = GraphAttentionLayerV2(n_hidden, n_hidden, n_heads,
                                         is_concat=False, dropout=dropout, share_weights=share_weights)
        self.bn3 = nn.BatchNorm1d(nodes_n * n_hidden)

        self.output_linear = nn.Linear(n_hidden, in_features)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        # self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, inputs, adj_mat):
        if len(inputs.shape) == 2:
            inputs.unsqueeze(0)

        x = self.input_linear(inputs)
        x = self.bin(x)
        x = self.activation(x)
        y = self.ga1(x, adj_mat)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.activation(y)
        y = self.dropout(y)

        y = self.ga2(y, adj_mat)
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.activation(y)
        y = self.dropout(y)

        y = self.ga3(y, adj_mat)
        # b, n, f = inputs.shape
        y = self.bn3(y.view(b, -1)).view(b, n, f)
        # y = self.activation(y)
        y = self.dropout(y)

        y = self.output_linear(y)
        y = y + x
        return y


class GraphAttentionLayerV3(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 n_nodes=57, bias=True, is_concat=True, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayerV3, self).__init__()
        self.n_nodes = n_nodes

        # Learnable adj matrix initialisation---------------------------------------
        self.adj = Parameter(torch.FloatTensor(n_nodes, n_nodes))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # --------------------------------------------------------------------------

        # Graph Attention-----------------------------------------------------------
        self.is_concat = is_concat
        self.n_heads = n_heads
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)  # embeddings
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)  # attention
        self.activation = nn.LeakyReLU(negative_slope=alpha)  # actf for attention scores
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        # ----------------------------------------------------------------------------
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.adj.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        attn_stdv = 1. / math.sqrt(self.attn.weight.size(1))
        self.attn.weight.data.uniform_(-attn_stdv, attn_stdv)

    def forward(self, h: torch.Tensor):
        batch_size, n_nodes = h.shape[0], h.shape[1]
        g = self.linear(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)  # embed and split for each head

        # Calculate attention scores
        g_repeat = g.repeat(1, n_nodes, 1, 1)  # [g1,...,gN,g1, ...gN, etc.]
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=1)  # [g1, g1,..., g2, ...g2]
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(batch_size, n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)

        e = torch.matmul(self.adj, e)
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('bijh,bjhf->bihf', a, g)

        # Concatenate the heads
        if self.is_concat:
            attn_res = attn_res.reshape(batch_size, n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            attn_res = attn_res.mean(dim=2)  # [n_nodes, self.n_hidden]

        if batch_size == 1:
            attn_res = attn_res.squeeze(0)

        return attn_res


class GATv3(nn.Module):
    def __init__(self, in_features: int, n_hidden: int, n_heads: int, nodes_n=57, is_concat=True,
                 dropout=0.5):
        super(GATv3, self).__init__()
        self.input_linear = nn.Linear(in_features, in_features)
        self.bin = nn.Linear(nodes_n, in_features)
        self.ga1 = GraphAttentionLayerV3(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(nodes_n * n_hidden)

        self.ga2 = GraphAttentionLayerV3(n_hidden, n_hidden, n_heads, is_concat=True, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(nodes_n * n_hidden)

        self.ga3 = GraphAttentionLayerV3(n_hidden, n_hidden, 1, is_concat=False, dropout=dropout)
        self.bn3 = nn.BatchNorm1d(nodes_n * n_hidden)

        self.output_linear = nn.Linear(n_hidden, in_features)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.activation1 = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, inputs):
        if len(inputs.shape) == 2:
            inputs.unsqueeze(0)

        x = self.input_linear(inputs)
        x = self.bin(x)
        x = self.activation1(x)
        y = self.ga1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.activation(y)
        # y = self.dropout(y)

        y = self.ga2(y)
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.activation(y)
        # y = self.dropout(y)

        y = self.ga3(y)
        # b, n, f = inputs.shape
        y = self.bn3(y.view(b, -1)).view(b, n, f)
        # y = self.activation(y)
        # y = self.dropout(y)

        y = self.output_linear(y)
        y = y + x
        return y


class GraphAttentionLayerV3_2(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 n_nodes=57, bias=True, is_concat=True, dropout=0.6, alpha=0.2,
                 share_weights=True, return_adj=False):
        super(GraphAttentionLayerV3_2, self).__init__()
        self.n_nodes = n_nodes
        self.return_adj = return_adj

        # Learnable adj matrix initialisation---------------------------------------
        self.adj = Parameter(torch.FloatTensor(n_nodes, n_nodes))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Graph Attention-----------------------------------------------------------
        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weight = share_weights
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        # self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)  # attention
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=alpha)  # actf for attention scores
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)
        # ----------------------------------------------------------------------------
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear_l.weight.size(1))
        self.linear_l.weight.data.uniform_(-stdv, stdv)
        if not self.share_weight:
            self.linear_r.weight.data.uniform_(-stdv, stdv)

        attn_stdv = 1. / math.sqrt(self.attn.weight.size(1))
        self.attn.weight.data.uniform_(-attn_stdv, attn_stdv)

        adj_stdv = 1. / math.sqrt(self.adj.size(1))
        self.adj.data = torch.abs(torch.randn(self.adj.size()) * adj_stdv)

    def forward(self, h: torch.Tensor):
        n_batch, n_nodes = h.shape[0], h.shape[1]
        g_l = self.linear_l(h).view(n_batch, n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_batch, n_nodes, self.n_heads, self.n_hidden)

        # Calculate attention scores
        g_l_repeat = g_l.unsqueeze(1).expand(n_batch, n_nodes, n_nodes, self.n_heads,
                                             self.n_hidden)  # [g1,...,gN,g1, ...gN, etc.]
        g_r_repeat_interleave = g_r.unsqueeze(2).expand(n_batch, n_nodes, n_nodes, self.n_heads,
                                                        self.n_hidden)  # [g1, g1,..., g2, ...g2]
        g_sum = g_l_repeat + g_r_repeat_interleave
        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)

        epsilon = 1e-5
        e = torch.matmul(self.adj, e)

        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('bijh,bjhf->bihf', a, g_r)

        # Concatenate the heads
        if self.is_concat:
            attn_res = attn_res.reshape(n_batch, n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            attn_res = attn_res.mean(dim=2)  # [n_nodes, self.n_hidden]

        if self.return_adj:
            return attn_res, self.adj
        else:
            return attn_res


class GATv3_2(nn.Module):
    # Add share weight
    def __init__(self, in_features: int, n_hidden: int, n_heads: int, nodes_n=57, is_concat=True,
                 dropout=0.5, share_weights=True):
        super(GATv3_2, self).__init__()
        self.share_weights = share_weights
        self.input_linear = nn.Linear(in_features, in_features)
        self.bin = nn.BatchNorm1d(nodes_n * in_features)
        self.ga1 = GraphAttentionLayerV3_2(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout,
                                           share_weights=True)
        self.bn1 = nn.BatchNorm1d(nodes_n * n_hidden)

        self.ga2 = GraphAttentionLayerV3_2(n_hidden, n_hidden, n_heads, is_concat=True, dropout=dropout,
                                           share_weights=True)
        self.bn2 = nn.BatchNorm1d(nodes_n * n_hidden)

        self.ga3 = GraphAttentionLayerV3_2(n_hidden, n_hidden, 1, is_concat=True, dropout=dropout, share_weights=True)
        self.bn3 = nn.BatchNorm1d(nodes_n * n_hidden)

        self.output_linear = nn.Linear(n_hidden, in_features)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.activation1 = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, inputs):
        if len(inputs.shape) == 2:
            inputs.unsqueeze(0)

        x = self.input_linear(inputs)
        b, n, f = inputs.shape
        x = self.bin(x.view(b, -1)).view(b, n, f)
        # x = self.activation1(x)
        y = self.ga1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.activation(y)
        # y = self.dropout(y)

        y = self.ga2(y)
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.activation(y)
        # y = self.dropout(y)

        y = self.ga3(y)
        # b, n, f = inputs.shape
        y = self.bn3(y.view(b, -1)).view(b, n, f)
        y = self.activation(y)
        # y = self.dropout(y)

        y = self.output_linear(y)
        y = y + x
        return y


class GAT_BlockV3(nn.Module):
    def __init__(self, n_hidden, n_heads):
        self.ga1 = GraphAttentionLayerV3_2(n_hidden, n_hidden, n_heads, is_concat=True, dropout=dropout,
                                           share_weights=True)
        self.ln1 = nn.LayerNorm(n_hidden)
        self.linear1 = nn.Linear(n_hidden)

    def forward(self, x):
        y = self.ga1(x)
        y = self.ln1(y)
        y = self.linear(y)
        y = y + x
        return y


class GATv3_3(nn.Module):
    # Add share weight
    def __init__(self, in_features: int, n_hidden: int, n_heads: int, nodes_n=57, is_concat=True,
                 dropout=0.5, share_weights=True):
        super(GATv3_3, self).__init__()
        self.share_weights = share_weights
        self.input_linear = nn.Linear(in_features, in_features)
        self.lnin = nn.LayerNorm(in_features)
        self.ga1 = GraphAttentionLayerV3_2(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout,
                                           share_weights=True)
        self.ln1 = nn.LayerNorm(n_hidden)

        self.ga2 = GraphAttentionLayerV3_2(n_hidden, n_hidden, n_heads, is_concat=True, dropout=dropout,
                                           share_weights=True)
        self.ln2 = nn.LayerNorm(n_hidden)

        self.ga3 = GraphAttentionLayerV3_2(n_hidden, n_hidden, n_heads, is_concat=True, dropout=dropout, share_weights=True)
        self.ln3 = nn.LayerNorm(n_hidden)
        self.output_linear = nn.Linear(n_hidden, in_features)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.activation1 = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, inputs):
        if len(inputs.shape) == 2:
            inputs.unsqueeze(0)
        x = self.input_linear(inputs)
        b, n, f = inputs.shape
        x = self.lnin(x)
        # x = self.activation1(x)
        y = self.ga1(x)
        y = self.ln1(y)
        y = self.activation(y)
        # y = self.dropout(y)

        y = self.ga2(y)
        y = self.ln2(y)
        y = self.activation(y)
        # y = self.dropout(y)

        y = self.ga3(y)
        # b, n, f = inputs.shape
        y = self.ln3(y)
        y = self.activation(y)
        # y = self.dropout(y)
        y = self.output_linear(y)
        y = y + x

        return y


class GraphAttentionLayerV4(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
        super(GraphAttentionLayerV4, self).__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weight = share_weights
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        n_batch, n_nodes = h.shape[0], h.shape[1]
        g_l = self.linear_l(h).view(n_batch, n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_batch, n_nodes, self.n_heads, self.n_hidden)

        g_l_repeat = g_l.unsqueeze(1).expand(n_batch, n_nodes, n_nodes, self.n_heads, self.n_hidden)
        g_r_repeat_interleave = g_r.unsqueeze(2).expand(n_batch, n_nodes, n_nodes, self.n_heads, self.n_hidden)

        g_sum = g_l_repeat + g_r_repeat_interleave
        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)

        if len(adj_mat.shape) == 2:
            adj_mat = adj_mat.unsqueeze(-1).repeat(n_batch, 1, 1, 1)

        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_batch
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == n_nodes
        assert adj_mat.shape[3] == 1 or adj_mat.shape[3] == self.n_heads

        # e = e.masked_fill(adj_mat == 0, float('-inf'))

        normalized_adj_mat = torch.softmax(adj_mat, dim=0)
        e = e * normalized_adj_mat
        # e = e * adj_mat
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('bijh,bjhf->bihf', a, g_r)

        if self.is_concat:
            attn_res = attn_res.reshape(n_batch, n_nodes, self.n_heads * self.n_hidden)
        else:
            attn_res = attn_res.mean(dim=2)
        return attn_res


class Simple_GCN_Attention(nn.Module):
    def __init__(self, in_features=25, hidden_features=1024, n_heads=4, p_dropout=0.5, node_n=57):
        super(Simple_GCN_Attention, self).__init__()
        # Input Graph Conv
        self.gcin = GraphConvolution(in_features, hidden_features, node_n=node_n, bias=True, return_adj=False)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_features)

        # Graph Attention
        self.gat1 = GraphAttentionLayerV3_2(hidden_features, hidden_features, n_heads, is_concat=True, dropout=p_dropout,
                                            share_weights=True)
        # Output Conv
        self.gcout = GraphConvolution(hidden_features, in_features, node_n=node_n, bias=True)
        # self.bn2 = nn.BatchNorm1d(node_n * hidden_features)
        self.do = nn.Dropout(p_dropout)
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.gcin(x)
        if len(x.shape) == 3:
            b, n, f = x.shape
        else:
            b = 1
            n, f = x.shape
        x = self.bn1(x.view(b, -1)).view(b, n, f)
        x = self.act(x)
        x = self.do(x)

        x = self.gat1(x)

        x = self.gcout(x)

        return x


class GCN_Attention(nn.Module):
    def __init__(self, input_feature, hidden_feature, n_heads, p_dropout, node_n=57):
        super(GCN_Attention, self).__init__()

        # Input Graph Conv
        self.gcin = GraphConvolution(input_feature, hidden_feature, node_n=node_n, bias=True, return_adj=True)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        # Graph Attention + GC Layers
        self.ga1 = GraphAttentionLayerV4(hidden_feature, hidden_feature, n_heads, is_concat=True, share_weights=True)
        self.gc1 = GraphConvolution(hidden_feature, hidden_feature, node_n=node_n, bias=True, return_adj=True)
        self.ga2 = GraphAttentionLayerV4(hidden_feature, hidden_feature, n_heads, is_concat=True, share_weights=True)

        # Output Graph Conv
        self.gcout = GraphConvolution(hidden_feature, input_feature, node_n=node_n, bias=True)
        self.bn2 = nn.BatchNorm1d(node_n * hidden_feature)

        self.do = nn.Dropout(p_dropout)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        # self.act = nn.ReLU()
        # self.act = nn.GELU()

    def forward(self, x):
        # Basic version had gcin, gat and gcout no adjacency matrix passed to ga
        x, adj_mat = self.gcin(x)
        if len(x.shape) == 3:
            b, n, f = x.shape
        else:
            b = 1
            n, f = x.shape
        x = self.bn1(x.view(b, -1)).view(b, n, f)
        x = self.act(x)
        x = self.do(x)

        # Graph Attention Layers using learned adj
        y = self.ga1(x, adj_mat)
        b, n, f = y.shape
        y, adj_mat = self.gc1(y)
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act(y)
        y = self.do(y)
        # y = self.ga2(y, adj_mat)
        z = y + x
        z = self.do(z)

        # Output Graph Conv
        out = self.gcout(z)
        return out


if __name__ == '__main__':
    bs = 32
    nodes_n = 57
    feat_dim = 25
    n_hidden = 256
    n_heads = 4
    dropout = 0.6
    test_input = torch.randn(bs, nodes_n, feat_dim)
    model = GraphAttentionLayerV3(in_features=feat_dim, out_features=n_hidden, n_heads=n_heads,
                                  n_nodes=57)
    out = model(test_input)

    adj_mat = np.ones((nodes_n, nodes_n), dtype=bool)
    np.fill_diagonal(adj_mat, False)  # set diagonal elements to False
    adj_matrix = torch.Tensor(adj_mat.astype(float)).unsqueeze(-1)  # convert NumPy array to PyTorch tensor
    model = GAT(in_features=feat_dim, n_hidden=n_hidden, n_heads=n_heads, dropout=dropout)
    output = model(test_input, adj_matrix)
