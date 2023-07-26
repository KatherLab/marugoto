import torch
import torch.nn as nn
import timm
class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention_dim = attention_dim
        self.input_dim = input_dim

        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x):
        attention_weights = self.attention_layer(x)
        attention_weights = torch.softmax(attention_weights, dim=1)
        return attention_weights
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MILModel(nn.Module):
    def __init__(self, n_classes, input_dim=1024, attention_dim=64, d_model=512, nhead=8, num_layers=6):
        super(MILModel, self).__init__()

        # 1D CNN
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Linear layer to adapt input dimensions
        conv1_output_dim = input_dim // 4
        self.linear = nn.Linear(64 * conv1_output_dim, d_model)

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # BiLSTM for sequence modeling
        self.bilstm = nn.LSTM(input_size=d_model, hidden_size=512, bidirectional=True, batch_first=True)

        # Attention mechanism
        self.attention = Attention(input_dim=1024, attention_dim=attention_dim)

        # Classifier head
        self.classifier = nn.Linear(1024, n_classes)

    def forward(self, x, register_hook=False):
        batch_size, seq_length, input_dim = x.size()

        # 1D CNN
        x = x.view(batch_size * seq_length, 1, input_dim)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(batch_size, seq_length, -1)

        # Calculate the size of the feature dimension after the 1D CNN layers
        _, _, feature_dim = x.size()

        # Linear layer
        x = x.view(batch_size, seq_length, -1)
        x = self.linear(x)

        # Transformer Encoder
        x = x.transpose(0, 1)  # Transpose for Transformer: (seq_length, batch_size, input_dim)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # Transpose back: (batch_size, seq_length, input_dim)

        # BiLSTM
        x, _ = self.bilstm(x)

        attention_weights = self.attention(x)
        x = torch.sum(x * attention_weights, dim=1)

        x = self.classifier(x)
        return x
