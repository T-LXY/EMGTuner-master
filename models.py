import torch
import torch.nn as nn

# Pulled directly from V9
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
        """
        Initiate the CNNLSTMClassifier with the specified architecture parameters, including the input size (number of channels), number of output classes, convolutional layer configuration, LSTM configuration, dropout rate, and whether to use bidirectional LSTMs.
        """
        
        # Call the superclass constructor to initialize the nn.Module
        super().__init__()

        # Store the parameters as instance variables
        self.input_size = input_size
        self.num_classes = num_classes
        self.conv_channels = list(conv_channels)
        self.kernel_size = kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # CNN expects shape: (batch, channels, time)
        conv_layers = []
        in_channels = input_size

        # Build the convolutional layers based on the specified configuration
        for i, out_channels in enumerate(self.conv_channels):
            # Each convolutional layer consists of a Conv1d, followed by BatchNorm1d, ReLU activation, and MaxPool1d to reduce the temporal dimension. 
            conv_layers.extend([
                nn.Conv1d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    padding = kernel_size // 2
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            ])

            # Only pool once, after the first conv block
            if i == 0:
                conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

            # Update in_channels for the next layer to be the out_channels of the current layer
            in_channels = out_channels
        
        # Combine the convolutional layers into a sequential module
        self.cnn = nn.Sequential(*conv_layers)

        # LSTM input size becomes the final number of CNN output channels
        self.lstm = nn.LSTM(
            input_size = in_channels,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_num_layers,
            batch_first = True,
            dropout = dropout if lstm_num_layers > 1 else 0.0,
            bidirectional = bidirectional
        )

        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)

        # avg pool + max pool => 2 * lstm_output_size
        self.head = nn.Sequential(
            nn.Linear(lstm_output_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, num_channels)

        # CNN expects (batch, channels, time)
        x = x.transpose(1, 2)

        # CNN feature extraction
        x = self.cnn(x)

        # Back to LSTM shape: (batch, seq_len, features)
        x = x.transpose(1, 2)

        # LSTM output for every time step
        lstm_out, _ = self.lstm(x)

        # Global average pooling over time
        avg_pool = torch.mean(lstm_out, dim=1)

        # Global max pooling over time
        max_pool, _ = torch.max(lstm_out, dim=1)

        # Combine both summaries
        features = torch.cat([avg_pool, max_pool], dim=1)

        logits = self.head(features)
        return logits
    
    