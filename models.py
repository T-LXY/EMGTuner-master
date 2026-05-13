import torch
import torch.nn as nn

# Pulled directly from V8
class CNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        conv_channels: list[int] | tuple[int, ...] = (64, 128),
        kernel_size: int = 7,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_size       = input_size
        self.num_classes      = num_classes
        self.conv_channels    = list(conv_channels)
        self.kernel_size      = kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers  = lstm_num_layers
        self.dropout          = dropout
        self.bidirectional    = bidirectional

        conv_layers = []
        in_channels = input_size
        for i, out_channels in enumerate(self.conv_channels):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
            ])
            if i == 0:
                conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.cnn = nn.Sequential(*conv_layers)

        self.lstm = nn.LSTM(
            input_size   = in_channels,
            hidden_size  = lstm_hidden_size,
            num_layers   = lstm_num_layers,
            batch_first  = True,
            dropout      = dropout if lstm_num_layers > 1 else 0.0,
            bidirectional= bidirectional,
        )

        lstm_out_size = lstm_hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(lstm_out_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, channels)
        x = x.transpose(1, 2)          # (batch, channels, seq_len)  for CNN
        x = self.cnn(x)
        x = x.transpose(1, 2)          # (batch, seq_len, features)  for LSTM
        lstm_out, _ = self.lstm(x)
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        features = torch.cat([avg_pool, max_pool], dim=1)
        return self.head(features)