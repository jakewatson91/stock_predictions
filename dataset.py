from torch.utils.data import Dataset, DataLoader

class MarketPriceDataset(Dataset):
    def __init__(self, X, y, dates):
        """
        X: torch.Tensor of shape (n_days, max_markets, D_total)
        y: torch.Tensor of shape (n_days,)
        dates: list of length n_days
        """
        self.X = X
        self.y = y
        self.dates = dates

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# usage
dataset = MarketPriceDataset(market_tensor, y, dates)
loader  = DataLoader(dataset, batch_size=16, shuffle=True)

for X_batch, y_batch in loader:
    # X_batch: (16, max_markets, D_total)
    # y_batch: (16,)
    preds = model(X_batch)       # your model should map (B, M, D) → (B,)
    loss  = loss_fn(preds, y_batch)
    …