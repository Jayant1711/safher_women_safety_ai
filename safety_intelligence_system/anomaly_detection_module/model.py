import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder (seq_len, batch, input_dim -> hidden_dim)
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                               num_layers=num_layers, batch_first=True, dropout=0.1)
        
        # Decoder
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
                               num_layers=num_layers, batch_first=True, dropout=0.1)
        
        # Reconstructor payload mapping
        self.reconstructor = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # Encoder mapping
        # We only care about the final hidden state or just mapping the sequence backward
        enc_out, (h_n, c_n) = self.encoder(x)
        
        # Take the last time-step's output from the encoder as the sequence embedding vector
        seq_embedding = enc_out[:, -1, :].unsqueeze(1) # shape (batch, 1, hidden_dim)
        
        # Repeat embedding vector for seq_len as input to decoder
        dec_in = seq_embedding.repeat(1, seq_len, 1) # shape (batch, seq_len, hidden_dim)
        
        # Decoder maps embedding sequence back out
        dec_out, _ = self.decoder(dec_in) # shape (batch, seq_len, hidden_dim)
        
        # Reconstruct exactly matched output shapes
        reconstructed = self.reconstructor(dec_out) # shape (batch, seq_len, input_dim)
        return reconstructed

if __name__ == "__main__":
    model = LSTMAutoencoder(input_dim=3)
    dummy_input = torch.randn(64, 20, 3) # batch sizes=64, seq=20, features=lat, lon, speed
    out = model(dummy_input)
    print("Autoencoder Reconstruction Shape:", out.shape)
