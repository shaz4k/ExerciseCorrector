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
        node_n (int): Number of nodes in the graph. Default is 57 for 19 joints.

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

    Ref:- Mao et. al: https://github.com/wei-mao-2019/LearnTrajDep

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
