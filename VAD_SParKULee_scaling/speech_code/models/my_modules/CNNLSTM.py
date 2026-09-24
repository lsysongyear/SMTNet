import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNExtractor(nn.Module):
    def __init__(self, input_channels=64, num_kernels=5, kernel_time=16):
        super(CNNExtractor, self).__init__()
        self.conv = nn.Conv1d(input_channels, num_kernels, kernel_size=kernel_time, padding='same')

    def forward(self, x):
        return self.conv(x)

class LSTMIntegrator(nn.Module):
    def __init__(self, num_kernels=5, lstm_hidden=4):
        super(LSTMIntegrator, self).__init__()
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=1, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
            for _ in range(num_kernels)
        ])

    def forward(self, x):
        B, num_kernels, T = x.shape
        outputs = []
        for i in range(num_kernels):
            channel_input = x[:, i, :].unsqueeze(-1)
            lstm_out, _ = self.lstms[i](channel_input)
            outputs.append(lstm_out)
        return torch.cat(outputs, dim=2)

class CNNLSTM(nn.Module):
    def __init__(self, input_channels=64, num_kernels=5, kernel_time=16,
                 lstm_hidden=4, output_dim=1, n_subjects=25):
        super(CNNLSTM, self).__init__()
        self.n_subjects = n_subjects
        self.input_channels = input_channels

        self.spatial_attentions = nn.ModuleList([
            self._build_spatial_attention(input_channels)
            for _ in range(n_subjects)
        ])

        self.cnn_extractor = CNNExtractor(input_channels, num_kernels, kernel_time)
        self.lstm_integrator = LSTMIntegrator(num_kernels, lstm_hidden)
        self.fc = nn.Linear(num_kernels * lstm_hidden, output_dim)

    @staticmethod
    def _build_spatial_attention(input_dim):
        hidden = 4 * input_dim
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, input_dim),
            nn.GELU(),
        )

    def forward(self, eeg, subject_ids=None):
        # eeg: (B, C, T)
        B = eeg.size(0)
        x = eeg.transpose(1, 2)                   # (B, T, C)

        if subject_ids is not None:
            outs = []
            for i in range(B):
                idx = subject_ids[i].item()
                outs.append(self.spatial_attentions[idx](x[i:i+1]))
            x = torch.cat(outs, dim=0)
        else:
            x = self.spatial_attentions[0](x)

        x = x.transpose(1, 2)                     # (B, C, T)
        x = self.cnn_extractor(x)
        x = self.lstm_integrator(x)
        x = self.fc(x)
        return x
