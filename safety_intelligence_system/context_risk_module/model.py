import torch
import torch.nn as nn
import math

class TemporalAttentionTransformer(nn.Module):
    def __init__(self, num_features=6, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        """
        A lightweight surrogate for Temporal Fusion Transformer targeting 
        time-based contextual modeling constraints.
        """
        super(TemporalAttentionTransformer, self).__init__()
        
        # Encode historical tabular features
        self.fc_in = nn.Linear(num_features, d_model)
        
        # Simple positional encoding for time awareness
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, d_model))
        
        # Self-Attention Blocks
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Binary Classification Readout (predict context risk)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        seq_len = x.size(1)
        
        # Project and Add positional encoding
        out = self.fc_in(x)
        out = out + self.positional_encoding[:, :seq_len, :]
        
        # Pass memory through transformer (batch_first=True)
        out = self.transformer(out)
        
        # Extract the representation of the final temporal step
        last_hidden_state = out[:, -1, :] # (batch_size, d_model)
        
        # Map to Risk Score Output [0, 1]
        risk_score = self.fc_out(last_hidden_state).squeeze(-1)
        return risk_score

if __name__ == "__main__":
    model = TemporalAttentionTransformer(num_features=6)
    dummy_input = torch.randn(16, 10, 6) # batch 16, window 10, features 6
    risk = model(dummy_input)
    print("Task 3 Model Output Shape:", risk.shape)
