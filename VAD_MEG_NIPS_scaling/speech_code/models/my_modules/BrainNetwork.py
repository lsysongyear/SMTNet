import torch.nn as nn
import torch


class Brain_Magic_speech(nn.Module):
    """Brain signal regression network for 100Hz sampling rate."""
    
    def __init__(self, input_dim=306, attention_dim=128, output_dim=1, 
                 kernel_size=5, depthwise_kernel=15, dropout=0.1):
        """
        
        Args:
            input_dim: Input feature dimension (e.g., 306 EEG channels)
            attention_dim: Dimension after spatial attention
            output_dim: Output dimension (e.g., 39 phoneme classes)
            kernel_size: Kernel size for feature encoder blocks
            depthwise_kernel: Kernel size for depthwise separable convolution
            dropout: Dropout probability
        """
        super().__init__()
        
        # Spatial attention/transformer
        hidden_dims = [input_dim, 4 * input_dim]
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())  # or nn.ReLU()
            current_dim = hidden_dim  
        
        # Final projection to attention dimension
        layers.append(nn.Linear(current_dim, attention_dim))
        self.spatial_attention = nn.Sequential(*layers)
        
        # Feature encoder stack
        encoder_channels = [attention_dim, attention_dim, attention_dim, 2 * attention_dim]
        self.feature_encoder = nn.Sequential(
            Feature_Block(1, encoder_channels, kernel_size, dropout),
            Feature_Block(2, encoder_channels, kernel_size, dropout),
            Feature_Block(3, encoder_channels, kernel_size, dropout),
            Feature_Block(4, encoder_channels, kernel_size, dropout),
            Feature_Block(5, encoder_channels, kernel_size, dropout),
        )
         # Multi-scale temporal convolution block
        self.short_conv_block = DeepMultiScaleBlock(
            in_channels=attention_dim,
            out_channels=attention_dim,
            kernel_sizes=[3, 5, 7, 9],
            num_layers=12,
            dropout=dropout
        )
        self.lstm = nn.LSTM(
            input_size=attention_dim,  # 每个时间步输入特征维度
            hidden_size=attention_dim, # 输出维度
            num_layers=1,
            batch_first=True,
            bidirectional=True,     # 如果想双向，可设为 True
            dropout=dropout
        )
        # Depthwise separable convolution
        self.depthwise_conv = nn.Sequential(
            # Depthwise convolution
            nn.Conv1d(
                in_channels=2 * attention_dim,
                out_channels=2 * attention_dim,
                kernel_size=depthwise_kernel,
                padding='same',
                groups=2 * attention_dim,
                bias=False
            ),
            nn.BatchNorm1d(2 * attention_dim),
            nn.GELU(),
            
            # Pointwise convolution
            nn.Conv1d(
                in_channels=2 * attention_dim,
                out_channels=attention_dim,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm1d(attention_dim),
            nn.GELU()
        )
        
        # Final classification convolutions
        self.final_conv1 = nn.Conv1d(attention_dim, 4 * attention_dim, kernel_size=1)
        self.final_conv2 = nn.Conv1d(4 * attention_dim, output_dim, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, input_dim, sequence_length] e.g., [batch, 204, 1200]
            
        Returns:
            Output tensor [batch, 1200] e.g., [batch, 1200]
        """
        batch_size, input_dim, seq_len = x.shape
        
        # Spatial attention: [batch, input_dim, seq_len] -> [batch, seq_len, attention_dim]
        x = x.transpose(1, 2)  # [batch, seq_len, input_dim]
        x = self.spatial_attention(x)  # [batch, seq_len, attention_dim]
        x = x.transpose(1, 2)  # [batch, attention_dim, seq_len]
        
        # Multi-scale feature extraction
        short_features = self.short_conv_block(x)  # [batch, attention_dim, seq_len]
        
        # Deep feature encoding
        encoded_features = self.feature_encoder(x)  # [batch, attention_dim, seq_len]

        # Lstm feature encoding
        lstm_features, _ = self.lstm(x.transpose(1,2))
        
        # Feature fusion
        fused_features = torch.cat([encoded_features, short_features], dim=1) + lstm_features.transpose(1,2)  # [batch, 2*attention_dim, seq_len]
        
        # Depthwise separable convolution
        x = self.depthwise_conv(fused_features)  # [batch, attention_dim, seq_len]
        
        # Final convolutions
        x = self.final_conv1(x)  # [batch, 4*attention_dim, seq_len]
        x = self.final_conv2(x)  # [batch, output_dim, seq_len]
    
        output = x.squeeze(1)
        
        return output

class Feature_Block(nn.Module):
    """Feature encoder block with dilated convolutions and residual connections."""
    
    def __init__(self, layer_index, channels, kernel_size=10, dropout=0.5):
        """
        Initialize FeatureEncoderBlock1.
        
        Args:
            layer_index: Index of the layer (k) for dilation computation
            channels: List of channel dimensions [in, hidden1, hidden2, out]
            kernel_size: Size of convolution kernels
            dropout: Dropout probability
        """
        super().__init__()
        self.layer_index = layer_index
        
        # Compute dilation rates based on layer index and kernel size
        if kernel_size == 3:
            dilation1 = int(2 ** ((2 * layer_index) % 5))
            dilation2 = int(2 ** ((2 * layer_index + 1) % 5))
        else:
            dilation1 = int(2 ** (layer_index % 6))
            dilation2 = int(2 ** ((layer_index + 1) % 6))
        
        # Convolution layers with appropriate dilations
        self.conv1 = nn.Conv1d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=kernel_size,
            padding='same',
            dilation=dilation1
        )
        self.conv2 = nn.Conv1d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=kernel_size,
            padding='same',
            dilation=dilation2
        )
        self.conv3 = nn.Conv1d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=kernel_size,
            padding='same',
            dilation=2
        )
        
        # Normalization layers
        self.batch_norm1 = nn.BatchNorm1d(channels[1], eps=1e-05, momentum=0.1)
        self.batch_norm2 = nn.BatchNorm1d(channels[2], eps=1e-05, momentum=0.1)
        
        # Activation and regularization
        self.activation = nn.GELU()
        self.glu = nn.GLU(dim=-2)  # GLU over channel dimension
        self.dropout = nn.Dropout1d(dropout)
        
    def forward(self, x):
        """
        Forward pass with optional residual connection.
        
        Args:
            x: Input tensor [batch, channels[0], sequence_length]
            
        Returns:
            Output tensor [batch, channels[3]//2, sequence_length] (after GLU)
        """
        input_x = x  # Store for residual connection
        
        # First conv block
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Third conv block with GLU
        x = self.conv3(x)
        x = self.glu(x)
        x = self.dropout(x)
        
        # Add residual connection for layers > 1
        if self.layer_index > 1:
            # Adjust dimensions if necessary (assuming input matches after GLU)
            x = input_x + x
            
        return x
    
class ShortTermTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, dropout=0.2):
        """
        Short-term temporal feature convolution module

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel (default: 3)
            stride: Stride of the convolution (default: 1)
            dilation: Dilation rate (default: 1)

        dropout: Dropout probability (default: 0.2)
        """
        super(ShortTermTemporalConv, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size + (kernel_size-1)*(dilation-1) - 1) // 2,  
            dilation=dilation
        )
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        """
        input: (batch_size, in_channels, seq_len)
        output: (batch_size, out_channels, seq_len)
        """
        residual = x
        if self.residual is not None:
            residual = self.residual(residual)
            
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = residual + out
        
        return out

class MultiScaleTemporalLayer(nn.Module):
    """
    Multi-scale temporal feature block, capturing short-term features at different time scales
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dilation=1, dropout=0.2):
        super(MultiScaleTemporalLayer, self).__init__()
        assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by number of kernel sizes"
        self.convs = nn.ModuleList([
            ShortTermTemporalConv(
                in_channels, 
                out_channels // len(kernel_sizes),  
                kernel_size=ks,
                dilation=dilation,
                dropout=dropout
            ) for ks in kernel_sizes
        ])
        
    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.convs], dim=1)
        return out
    
class DeepMultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], 
                 num_layers=3, dropout=0.2, dilation_growth=1):
        """
        Deep stacked multi-scale temporal convolution block
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels per layer
            kernel_sizes: List of convolution kernel sizes to use
            num_layers: Number of stacked layers
            dropout: Dropout probability
            dilation_growth: Growth factor for dilation rate (1 indicates no change)
        """
        super(DeepMultiScaleBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        self.skip_cons = nn.ModuleList()  # Skip connections
        
        # Create multi-layer structure
        for i in range(num_layers):
            layer_in_channels = in_channels if i == 0 else out_channels
            dilation = dilation_growth ** i
            
            self.layers.append(
                MultiScaleTemporalLayer(
                    layer_in_channels,
                    out_channels,
                    kernel_sizes=kernel_sizes,
                    dilation=dilation,
                    dropout=dropout
                )
            )
            
            # Skip connections (use 1x1 convolution to match dimensions)
            if layer_in_channels != out_channels:
                self.skip_cons.append(nn.Conv1d(layer_in_channels, out_channels, 1))
            else:
                self.skip_cons.append(nn.Identity())
        
        # Final fusion and normalization
        self.final_norm = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        for i, (layer, skip_conv) in enumerate(zip(self.layers, self.skip_cons)):
            x = layer(x)
            if i > 0:  # Add skip connection starting from the second layer
                x = skip_conv(residual) + x
                residual = x
                
        x = self.final_norm(x)
        x = self.final_relu(x)
        return x

    
if __name__ == "__main__":
    device='cpu'
    model = Brain_Magic_speech(input_dim=204, attention_dim=128,kernel_size=25,dropout=0.01).to(device)
    x = torch.rand(1, 204, 1000).to(device)
    y = model(x)
    print(y.shape)
