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
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

class DriverRiskTransformer(nn.Module):
    def __init__(self, num_features=7, d_model=64, nhead=4, num_layers=4, dim_feedforward=128, dropout=0.1):
        super(DriverRiskTransformer, self).__init__()
        
        # Project tabular features into d_model dimension
        self.feature_projection = nn.Linear(num_features, d_model)
        
        # Positional Encoding for sequence awareness
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Classification Head
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, src):
        # src shape: (batch_size, seq_len, num_features)
        
        # Project and transpose for Transformer (seq_len, batch_size, d_model)
        x = self.feature_projection(src)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        x = self.transformer_encoder(x) # (seq_len, batch_size, d_model)
        
        # Pool the sequence (use the output of the last timestep)
        last_timestep_out = x[-1, :, :] # (batch_size, d_model)
        
        # Classification
        out = self.fc_out(last_timestep_out) # (batch_size, 1)
        
        return out.squeeze()

# Fast checks
if __name__ == '__main__':
    model = DriverRiskTransformer(num_features=7)
    dummy_input = torch.randn(32, 10, 7) # batch_size=32, seq_len=10, features=7
    out = model(dummy_input)
    print("Model Output Shape:", out.shape)
