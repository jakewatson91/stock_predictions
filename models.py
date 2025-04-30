import torch
import torch.nn as nn

class PriceLSTM(nn.Module):
    def __init__(self, M, D, Dprime=128, hidden_size=64, num_layers=1):
        super().__init__()
        # 1) Encode each day's (M, D) tensor into Dprime dims
        self.day_encoder = nn.Sequential(
            nn.Linear(D, Dprime),
            nn.ReLU(),
            nn.Linear(Dprime, Dprime),
        )

        # 2) GRU over days
        self.gru = nn.GRU(
            input_size=Dprime,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # 2) LSTM over those day embeddings
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # 3) Final regressor from hidden state → price
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x_seq):
        """
        x_seq: (batch, seq_len, M, D)
        """
        B, S, M, D = x_seq.size()

        # build mask of which markets are real
        mask = (x_seq.abs().sum(dim=-1) > 0).float()  # (B, S, M)

        # collapse to (B*S, M, D) for day-encoder
        flat = x_seq.view(B*S, M, D)                  # (B*S, M, D)
        h = self.day_encoder(flat)                # (B*S, Dprime)

        # print(f"  [day_encoder] h: min {h.min().item():.4f}, max {h.max().item():.4f}, mean {h.mean().item():.4f}")
        
        # reshape back to (B, S, M, Dprime)
        h = h.view(B, S, M, -1)                       # (B, S, M, Dprime)

        # masked‐mean over the M dimension
        mask = mask.unsqueeze(-1)                     # (B, S, M, 1)
        h_sum = (h * mask).sum(dim=2)                 # (B, S, Dprime)
        cnt   = mask.sum(dim=2).clamp(min=1e-3)        # (B, S, 1)
        day_embs = h_sum / cnt                        # (B, S, Dprime)

        gru_out, _  = self.gru(day_embs)              # (B, S, gru_hidden)
        lstm_out, (hn, _) = self.lstm(gru_out)              # hn[-1] is (B, hidden_size)

        # map to price
        return self.regressor(hn[-1]).squeeze(-1)      # (B,)