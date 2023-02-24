import math
import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d

# Custom modules
from Layers import Embedder, PositionalEncoder, Norm, EncoderLayer, DecoderLayer, get_clones
from Layers import GraphConvolution, GC_Block


# Classification-------------------------------------------------------------------------------------------------------#
class Simple_GCN_Classifier(nn.Module):
    """
    A simple GCN-based graph classifier.

    This class defines a simple graph convolutional neural network for classification tasks.
    It consists of two graph convolutional layers, batch normalization layers, a linear layer,
    and a dropout layer. The activation function is ReLU, and the final layer uses LogSoftmax.

    Args:
        input_feature (int): Number of input features.
        hidden_feature (int): Number of hidden features.
        p_dropout (float): Dropout probability.
        node_n (int): Number of nodes in the graph.
        classes (int): Number of classes for classification.

    Attributes:
        gcin (GraphConvolution): The first graph convolutional layer.
        gcout (GraphConvolution): The second graph convolutional layer.
        bnin (BatchNorm1d): Batch normalization layer for the input features.
        bnout (BatchNorm1d): Batch normalization layer for the output features.
        lin (Linear): Linear layer for classification.
        do (Dropout): Dropout layer.
        act_f (ReLU): ReLU activation function.
        act_flin (LogSoftmax): LogSoftmax activation function.

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


# Correction ----------------------------------------------------------------------------------------------------------#
class GCN_Corrector(nn.Module):
    # Separated Corrector
    def __init__(self, input_feature=25, hidden_feature=256, p_dropout=0.5, num_stage=2, node_n=57):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
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

# Transformer, Encoder and Decoder ----------------------------------------------=-------------------------------------#
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_outputs = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_outputs)
        return output


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)

        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


if __name__ == '__main__':
    model = Simple_GCN_Classifier()
