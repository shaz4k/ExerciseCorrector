import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, MaxPool2d
import torch.nn.functional as F

# Custom modules
from utils.custom_layers import Embedder, PositionalEncoder, Norm, EncoderLayer, DecoderLayer, get_clones
from utils.custom_layers import GraphConvolution, GC_Block


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


class CNN_Classifier(nn.Module):
    def __init__(self, num_classes=12):
        super(CNN_Classifier, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.maxPool1 = MaxPool2d(kernel_size=2, stride=2)
        self.batchNorm1 = BatchNorm2d(64)

        self.conv3 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.maxPool2 = MaxPool2d(kernel_size=2, stride=2)
        self.batchNorm2 = BatchNorm2d(128)

        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, num_classes)

    def forward(self, inputs):
        inputs = inputs.view(-1, 1, 57, 25)  # reshape input to [batch_size, channels, height, width]
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = self.maxPool1(x)
        x = self.batchNorm1(x)
        x = self.conv3(x)
        x = self.maxPool2(x)
        x = self.batchNorm2(x)
        x = self.globalAvgPool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.dense1(x))
        x = F.softmax(self.dense2(x), dim=1)
        stop=1

        return x


# Correction ----------------------------------------------------------------------------------------------------------#
class GCN_Corrector(nn.Module):
    # Separated Corrector
    def __init__(self, input_feature=25, hidden_feature=256, p_dropout=0.5, num_stage=2, node_n=57):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks (Mao = 12 blocks)
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


# Simple PyTorch Transformer ------------------------------------------------------------------------------------------#
class Simple_Transformer(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=512, num_layers=6, num_heads=8, dropout_prob=0.1, num_classes=12):
        super(Simple_Transformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes

        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=2048,
                                                   dropout=dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)
        encoded = self.transformer_encoder(embedded)
        pooled = encoded.mean(dim=0)
        logits = self.output(pooled)

        return logits


# Custom Transformer, Encoder and Decoder -----------------------------------------------------------------------------#
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
