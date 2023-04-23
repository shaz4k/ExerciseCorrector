import math
import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, MaxPool2d
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
from torch.nn import MultiheadAttention
import torch.nn.init as init

# Custom modules
from utils.GraphLayers import GraphConvolution, GC_Block, GC_BlockV2



# Classification-------------------------------------------------------------------------------------------------------#
class Simple_GCN_Classifier(nn.Module):
    """
    A simple GCN-based graph classifier.

    This class defines a simple graph convolutional neural network for classification tasks.
    It consists of two graph convolutional layers, batch normalization layers, a linear layer,
    and a dropout layer. The activation function is ReLU, and the final layer uses LogSoftmax.

    """

    def __init__(self, input_feature=25, hidden_feature=32, p_dropout=0.5, node_n=57, classes=12):
        super(Simple_GCN_Classifier, self).__init__()
        torch.manual_seed(42)

        # Initialise layers
        self.gcin = GraphConvolution(input_feature, hidden_feature)
        self.gcout = GraphConvolution(hidden_feature, input_feature)
        self.bnin = BatchNorm1d(node_n * hidden_feature)
        self.bnout = BatchNorm1d(node_n * input_feature)
        self.lin = nn.Linear(node_n * input_feature, classes)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()
        self.act_flin = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if len(x.shape) == 3:
            b, n, f = x.shape
        else:
            b = 1
            n, f = x.shape
        # x = [32, 57, 25] --> gcin --> y = [32, 57, 32] --> bn -->[32, 57*32=1824]
        y = self.gcin(x)
        if b > 1:
            y = self.bnin(y.view(b, -1)).view(y.shape)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gcout(y)
        if b > 1:
            y = self.bnout(y.view(b, -1)).view(y.shape)
        y = self.act_f(y)
        y = self.do(y)

        y = y.view(-1, n * f)
        y = self.lin(y)
        y = self.act_flin(y)

        return y


class CNN_Classifier(nn.Module):
    """
        CNN Classifier:
            inputs (tensor): shape (batch_size, channels, joints, frames)
            outputs (tensor): shape (batch_size, classes)

        Attributes:
            Conv2D Layers, Max Pooling, Batch Normalisation, Global Average Pooling, Dense layers, Leaky ReLU
    """
    def __init__(self, in_channels, num_classes=12):
        super(CNN_Classifier, self).__init__()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=32, kernel_size=2, padding=1)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=1)
        self.maxPool1 = MaxPool2d(kernel_size=2, stride=2)
        self.batchNorm1 = BatchNorm2d(64)

        self.conv3 = Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=1)
        self.maxPool2 = MaxPool2d(kernel_size=2, stride=2)
        self.batchNorm2 = BatchNorm2d(128)

        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, num_classes)
        # self.act_f = nn.ReLU()
        # self.act_f = nn.Tanh()
        self.act_f = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, inputs):
        # input shape [batch, channels, joints, frames]
        if len(inputs.shape) == 3:
            b, n, f = inputs.shape
            inputs = inputs.view(-1, 1, n, f)
        # change relu to leaky relu
        x = self.act_f(self.conv1(inputs))
        x = self.act_f(self.conv2(x))
        x = self.maxPool1(x)
        x = self.batchNorm1(x)
        x = self.conv3(x)
        x = self.maxPool2(x)
        x = self.batchNorm2(x)
        x = self.globalAvgPool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.act_f(self.dense1(x))
        x = F.log_softmax(self.dense2(x), dim=1)

        return x


class CNN_Classifier_v2(nn.Module):
    def __init__(self, in_channels, num_classes=12):
        super(CNN_Classifier_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding=1)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.batchNorm2 = nn.BatchNorm2d(256)
        self.batchNorm3 = nn.BatchNorm2d(512)

        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(256, 128)
        self.dense2 = nn.Linear(128, num_classes)

        # self.act_f = nn.ReLU()
        self.act_f = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, inputs):
        if len(inputs.shape) == 3:
            b, n, f = inputs.shape
            inputs = inputs.view(-1, 1, n, f)
        x = self.act_f(self.conv1(inputs))
        x = self.act_f(self.conv2(x))
        x = self.maxPool(x)
        x = self.batchNorm1(x)
        x = self.act_f(self.conv3(x))
        x = self.act_f(self.conv4(x))
        x = self.maxPool(x)
        x = self.batchNorm2(x)
        x = self.globalAvgPool(x)
        x = torch.flatten(x, 1)
        x = self.act_f(self.dense1(x))
        x = F.log_softmax(self.dense2(x), dim=1)
        return x


# Correction ----------------------------------------------------------------------------------------------------------#
class GCN_Corrector(nn.Module):
    # Separated Corrector
    def __init__(self, input_feature=25, hidden_feature=256, p_dropout=0.5, num_stage=2, node_n=57):
        """
        Args:
            :param input_feature: num of input feature
            :param hidden_feature: num of hidden feature
            :param p_dropout: drop out prob.
            :param num_stage: number of residual blocks
            :param node_n: number of nodes in graph
            :return out: residuals

        Attributes:
            Graph Convolution, GC_Block, ReLU
        """
        super(GCN_Corrector, self).__init__()
        self.num_stage = num_stage

        self.gcin = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gcout = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.gcatt = GraphConvolution(hidden_feature, 1, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()
        self.act_fatt = nn.Sigmoid()

    def forward(self, x):
        y = self.gcin(x)
        if len(y.shape) == 3:
            b, n, f = y.shape
        else:
            b = 1
            n, f = y.shape

        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        out = self.gcout(y)

        att = self.gcatt(y)
        att = self.act_fatt(att)

        return out, att


class GCN_Corrector_Attention(nn.Module):
    # Separated Corrector
    def __init__(self, input_feature=25, hidden_feature=256, p_dropout=0.5, num_stage=2, num_heads=2, node_n=57):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        :return out: residuals
        """
        super(GCN_Corrector_Attention, self).__init__()
        self.num_stage = num_stage

        self.gcin = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)
        self.ln1 = nn.LayerNorm(hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_BlockV2(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        # Add the attention layer
        self.attention = MultiheadAttention(embed_dim=hidden_feature, num_heads=num_heads, dropout=0.,
                                            batch_first=True)
        self.attention2 = MultiheadAttention(embed_dim=hidden_feature, num_heads=num_heads, dropout=0.,
                                            batch_first=True)

        self.linear = nn.Linear(hidden_feature, input_feature)
        self.linear2 = nn.Linear(hidden_feature, hidden_feature)

        self.gcout = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()
        self.act_fatt = nn.GELU()
        # self.act_fatt = nn.LeakyReLU(negative_slope=0.1)

        stdv = 1. / torch.sqrt(torch.tensor(self.attention.embed_dim, dtype=torch.float32))
        self.attention.in_proj_weight.data.uniform_(-stdv, stdv)
        self.attention.out_proj.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        y = self.gcin(x)
        if len(y.shape) == 3:
            b, n, f = y.shape
        else:
            b = 1
            n, f = y.shape

        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        # Attention Layer
        attn_output, _ = self.attention(y, y, y)
        att = self.linear(attn_output)
        att = self.act_fatt(att)

        out = self.gcout(y)

        return out + att