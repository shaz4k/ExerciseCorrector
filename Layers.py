import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import copy


# Graph Convolutional Layers ------------------------------------------------------------------------------------------#
class GraphConvolution(nn.Module):
    """
    A graph convolutional layer.

    This class implements a graph convolutional layer, which performs a linear transformation on
    the input features using a learnable weight matrix, and aggregates the features of neighboring
    nodes in the graph using a learnable adjacency matrix. Optionally, a bias term can be added to
    the output features. The layer can be used in a larger graph neural network for tasks such as
    node classification or graph classification.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): If True, add a learnable bias term to the output features. Default is True.
        node_n (int): Number of nodes in the graph. Default is 57.

    Attributes:
        weight (Parameter): Learnable weight matrix of shape (in_features, out_features).
        adj (Parameter): Learnable adjacency matrix of shape (node_n, node_n).
        bias (Parameter or None): Learnable bias term of shape (out_features) or None.

    Methods:
        reset_parameters(): Initialize weight and bias parameters.
        forward(input): Perform a forward pass through the graph convolutional layer.

    """

    def __init__(self, in_features, out_features, bias=True, node_n=57):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.adj = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weight and bias parameters.

        This method initializes the weight and bias parameters using a uniform distribution.
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.adj.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        Perform a forward pass through the graph convolutional layer.

        This method applies a linear transformation to the input features using the learnable weight matrix,
        and aggregates the features of neighboring nodes in the graph using the learnable adjacency matrix.
        Optionally, a bias term can be added to the output features.

        Args:
            input (torch.Tensor): Input features of shape (batch_size, node_n, in_features).

        Returns:
            output (torch.Tensor): Output features of shape (batch_size, node_n, out_features).
        """
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    """
    A residual block of graph convolutional layers.

    This class defines a residual block of graph convolutional layers, which performs two graph convolutional
    operations and adds the original input to the output (element-wise addition). Batch normalization and dropout
    are used to improve the performance of the network. The block can be used in a larger graph neural network
    for tasks such as node classification or graph classification.

    Args:
        in_features (int): Number of input features.
        p_dropout (float): Dropout probability.
        bias (bool): If True, add a learnable bias term to the output features of the graph convolutional layers.
            Default is True.
        node_n (int): Number of nodes in the graph. Default is 57.

    Attributes:
        gc1 (GraphConvolution): The first graph convolutional layer.
        bn1 (BatchNorm1d): Batch normalization layer for the output features of the first graph convolutional layer.
        gc2 (GraphConvolution): The second graph convolutional layer.
        bn2 (BatchNorm1d): Batch normalization layer for the output features of the second graph convolutional layer.
        do (Dropout): Dropout layer.
        act_f (ReLU): ReLU activation function.

    Methods:
        forward(x): Perform a forward pass through the residual block.
        __repr__(): Return a string representation of the residual block.

    """

    def __init__(self, in_features, p_dropout, bias=True, node_n=57):

        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()

    def forward(self, x):
        """
        Perform a forward pass through the residual block.

        This method applies the two graph convolutional layers, followed by batch normalization,
        ReLU activation, and dropout, and adds the original input to the output (element-wise addition).

        Args:
            x (torch.Tensor): Input features of shape (batch_size, node_n, in_features).

        Returns:
            output (torch.Tensor): Output features of shape (batch_size, node_n, in_features).
        """
        y = self.gc1(x)
        if len(y.shape) == 3:
            b, n, f = y.shape
        else:
            b = 1
            n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# Transformer layers --------------------------------------------------------------------------------------------------#
# 1/3 - Lowest-layer containing embedder and positional encoding layer

# (1) Embedding Layer
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


# (2) Positional Encoding
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # create matrix for positional embeddings
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):  # step size = 2
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # save pe to be used later

        def forward(self, x):
            # make embeddings relatively larger using square root of dimensionality
            x = x * math.sqrt(self.d_model)
            # select contguous sub-tensor of positional encoding
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False).cuda()
            # x.shape = (batch_size, seq_length, embedding_size)
            return x


# 2/3 - intermediate layer
# Contains: attention function, feedforward, layer normalisation and multi-head attention
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # If mask provided
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    # Pass attn score through softmax to get probability dist of scores
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.2):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        # query, key, val
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # batch size
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear.view(bs, -1, self.h, self.d_k)
        q = self.k_linear.view(bs, -1, self.h, self.d_k)
        v = self.k_linear.view(bs, -1, self.h, self.d_k)
        # dimensions bs * seqlen * head * d_model

        # transpose to get dimensions batchsize * head * seqlen * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = (q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # d_ff is number of neurons in hidden layer (default=2048)
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout
        x = self.linear_2
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable params to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


# Layer 3/3
# Contains: single encoder, single decoder, clone function to stack encoder/decoder layers

# Generate multiple layers using clone function
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# Encoder Layer
# 1 multi-head attn and 1 ff
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = Feedforward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # SubLayer 1: Multi-Head Attention
        x2 = self.norm_1(x)
        attn_output = self.attn(x2, x2, x2, mask)
        x = x + x.dropout(attn_output)

        # SubLayer 2: Feed-forward
        x2 = self.norm_2(x)
        ff_output = self.ff(x2)
        x = x + self.dropout_2(ff_output)

        return x


# Decoder Layer
# 2 multi-head attn and 1 ff
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(d_model, heads)
        self.attn_1 = MultiHeadAttention(d_model, heads)

        self.ff = Feedforward(d_model).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):  # e_outputs = encoder outputs, input into decoder
        # Apply first multi-head attention layer
        x2 = self.norm_1(x)
        attn_output_1 = self.attn_1(x2, x2, x2, trg_mask)
        dropout_1 = self.dropout_1(attn_output_1)
        x = x + dropout_1

        # Apply the second multi-head attention layer
        x2 = self.norm_2(x)
        attn_output_2 = self.attn_2(x2, e_outputs, e_outputs, src_mask)
        dropout_2 = self.dropout_2(attn_output_2)
        x = x + dropout_2

        # Apply feedforward layer
        x2 = self.norm_3(x)
        ff_out = self.ff(x2)
        dropout_3 = self.dropout_3(ff_out)
        x = x + dropout_3

        return x
