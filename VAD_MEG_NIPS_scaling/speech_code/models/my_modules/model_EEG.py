import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        pe = self.pe.repeat(x.size(0), 1, 1)
        x = torch.cat((x, pe[:, :x.size(1)]), dim=2)
        return x

class CNN_baseline(nn.Module):
    def __init__(self):
        super(CNN_baseline, self).__init__()
        self.layernorm = nn.LayerNorm(64)
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=25, padding=12, stride=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=25, padding=12, stride=2)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=25)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=25, padding=8)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(17,64), padding=(8, 0))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(128*3, 1))
        self.fc1 = nn.Linear(in_features=100, out_features=100)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=100, out_features=768)

    def forward(self, x):
        x = self.layernorm(x)
        x = x.transpose(1, 2)
        conv1_out = self.conv1(x)
        relu1_out = self.relu(conv1_out)
        conv2_out = self.conv2(relu1_out)
        relu2_out = self.relu(conv2_out)
        conv3_out = self.conv3(relu2_out)
        relu3_out = self.relu(conv3_out)
        conv4_out = self.conv4(relu3_out)
        relu4_out = self.relu(conv4_out)
        conv5_out = self.conv5(relu4_out)
        relu5_out = self.sigmoid(conv5_out)
        relu5_out = relu5_out.transpose(1, 2)
        return relu5_out[:,:-1,:]



# --------------------------CNN-baseline-----------------------------
class CNN_baseline0(nn.Module):
    def __init__(self):
        super(CNN_baseline0, self).__init__()
        self.layernorm = nn.LayerNorm(64)
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(17,64), padding=(8, 0))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(128*3, 1))
        self.fc1 = nn.Linear(in_features=100, out_features=100)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=100, out_features=256)

    def forward(self, x):
        x = self.layernorm(x)
        x = x.unsqueeze(dim=1)
        conv_out = self.conv_layer(x)
        relu_out = self.relu(conv_out)
        avg_pool_out = self.avg_pool(relu_out)
        flatten_out = torch.flatten(avg_pool_out, start_dim=1)
        fc1_out = self.fc1(flatten_out)
        sigmoid_out = self.sigmoid(fc1_out)
        fc2_out = self.fc2(sigmoid_out)
        return fc2_out

class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.layernorm1 = nn.LayerNorm(64)
        self.conv_layer1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding='same')
        self.layernorm2 = nn.LayerNorm(32)
        self.conv_layer2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool1d(kernel_size=6, stride=6)
        self.pe = PositionalEncoding(d_model=32, max_len=64)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=64*2, dropout=0.5),
            num_layers=2
        )
        self.fc1 = nn.Linear(in_features=64, out_features=256)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=256, out_features=256)

    def forward(self, x):
        x = self.layernorm1(x)
        x = x.transpose(1, 2)
        conv_out = self.conv_layer1(x)
        relu_out = self.relu(conv_out)
        relu_out = self.dropout(relu_out)

        relu_out = relu_out.transpose(1, 2)
        relu_out = self.layernorm2(relu_out)
        relu_out = relu_out.transpose(1, 2)
        conv_out = self.conv_layer2(relu_out)
        relu_out = self.relu(conv_out)
        relu_out = self.dropout(relu_out)

        avg_pool_out = self.avg_pool(relu_out)
        avg_pool_out = avg_pool_out.transpose(1, 2)
        pe_out = self.pe(avg_pool_out)
        transformer_out = self.transformer(pe_out)
        out = self.fc1(transformer_out)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out[:, :-1, :]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same')
        self.layernorm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        conv_out = self.conv(x)
        relu_out = self.relu(conv_out)
        relu_out = relu_out.transpose(1, 2)
        layernorm_out = self.layernorm(relu_out)
        layernorm_out = layernorm_out.transpose(1, 2)
        dropout_out = self.dropout(layernorm_out)
        return dropout_out



class ConcatCovNet(nn.Module):
    def __init__(self):
        super(ConcatCovNet, self).__init__()
        self.layernorm = nn.LayerNorm(62)
        self.conv_pre = nn.Conv1d(in_channels=62, out_channels=32, kernel_size=3, padding='same')
        self.avg_pool = nn.AvgPool1d(kernel_size=6, stride=6)
        self.pe = PositionalEncoding(d_model=32, max_len=64)
        self.conv_block1 = ConvBlock(in_channels=64, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=256, out_channels=256)
        # self.conv_block4 = ConvBlock(in_channels=256, out_channels=256)

        # self.conv_layer2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=256, out_features=256)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=256, out_features=256)

    def forward(self, x):
        x = self.layernorm(x)
        x = x.transpose(1, 2)
        x = self.conv_pre(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x_pe = self.pe(x)
        x_pe = x_pe.transpose(1, 2)
        x = self.conv_block1(x_pe)
        x_in = torch.cat([x, x_pe], dim=1)
        x = self.conv_block2(x_in)
        x_in = torch.cat([x, x_in], dim=1)
        x = self.conv_block3(x_in)
        x = x.transpose(1, 2)
        # x_in = torch.cat([x, x_in], dim=1)
        # x = self.conv_block4(x)

        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out[:, :-1, :]
