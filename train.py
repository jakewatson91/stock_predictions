import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import warnings
warnings.filterwarnings("ignore")

from spark_utils import read_parquet, prep_spark, top_k_markets_per_day
from encode import preprocess_markets
from models import PriceLSTM

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
embedding_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device).half()

# 2) Sequence dataset: given window of days → predict next-day price
class DaySequenceDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, seq_len: int):
        """
        X: (n_days, M, D), y: (n_days,)
        we produce samples 0..seq_len-1 → target at seq_len,
        then 1..seq_len → target at seq_len+1, etc.
        """
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        # last index we can start at is n_days - seq_len - 1
        return self.X.size(0) - self.seq_len

    def __getitem__(self, idx):
        # get window of days [idx, idx+seq_len)
        x_seq = self.X[idx : idx + self.seq_len]          # (seq_len, M, D)
        y_target = self.y[idx + self.seq_len]             # scalar
        return x_seq, y_target

def prep_split(target_df, ticker, market_tensor, dates, start, end):
    window = target_df.loc[start:end, ["NVDA", "SPY"]]
    prices_aligned = window.reindex(dates).bfill().ffill()

    X = market_tensor
    
    y = torch.tensor(prices_aligned.loc[dates, ticker].values, dtype=torch.float32)
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std

    # 2) Split into train/test arrays
    X_train, X_test, y_train, y_test = train_test_split(X, y_norm, test_size=0.2, shuffle=True, random_state=42)

    return X_train, X_test, y_train, y_test, y_std, y_mean

# 4) Training loop
def train(model, loader, ticker, epochs=1):
    for epoch in tqdm(range(1, epochs+1), desc="Training"):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # x_batch: (B, seq_len, M, D), y_batch: (B,)
            # print(f"x_batch.shape = {x_batch.shape}, y_batch.shape = {y_batch.shape}")
            # print("  x_batch: nan?", torch.isnan(x_batch).any().item(),
            #       "inf?", torch.isinf(x_batch).any().item(),
            #       "min", x_batch.min().item(), "max", x_batch.max().item())
            # print("  y_batch: min", y_batch.min().item(), "max", y_batch.max().item(),
            #       "mean", y_batch.mean().item())

            optimizer.zero_grad()
            preds = model(x_batch)

            # print("  preds: nan?", torch.isnan(preds).any().item(),
            #       "min", preds.min().item(), "max", preds.max().item(),
            #       "mean", preds.mean().item())

            loss  = loss_fn(preds, y_batch)
            # print("  loss:", loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        avg_loss = running_loss / len(loader.dataset)
        if epoch % 20 == 1:
            print(f"Epoch {epoch}/{epochs} — Train MSE: {avg_loss:.4f}")
    torch.save(model.state_dict(), f"{ticker}_model.pth")

def evaluate(model, loader, epochs=20):
    model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            total_loss += loss.item() * x_batch.size(0)

            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())

        mse = total_loss / len(loader.dataset)
        rmse = torch.sqrt(torch.tensor(mse))

        # flatten lists into tensors
        preds   = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        return mse, rmse, preds, targets

def plot_preds_vs_target(preds: np.ndarray, targets: np.ndarray, ticker):
    """
    preds: 1D array of your model predictions
    targets: 1D array of the true values (same length as preds)
    """
    plt.figure()
    plt.plot(preds, label="Predictions")
    plt.plot(targets, label="True Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Price")
    plt.title(f"{ticker}_preds_vs_target")
    plt.legend()
    plt.savefig(f"{ticker}_preds_vs_target.png")

if __name__ == "__main__":

    # ticker = "NVDA"
    ticker = "SPY"
    start, end = "2025-01-01", "2025-04-24"

    #--------------- Load and prep data --------------------#
    data = read_parquet()

    categorical_cols = ["category"]
    text_cols = ["market_title","market_subtitle","market_desc","event_title","event_subtitle"]
    exclude = ["open_time", "close_time", "created_date", "market_type"]
    numeric_cols = [col for col in data.columns if col not in (categorical_cols + text_cols + exclude)]
    # print(numeric_cols)

    df = prep_spark(data, numeric_cols)
    df = top_k_markets_per_day(df, k=100)

    df = df.toPandas()

    market_tensor, dates = preprocess_markets(
        df,
        numeric_cols,
        categorical_cols,
        text_cols,
        tokenizer,
        embedding_model,
        device,
        start,
        end,
        load_embeddings=False
    )
    dates = [d.normalize() for d in dates]

    # print("Tensor shape:", market_tensor.shape)  # (days, max_markets, D_total)
    # print("Dates:", dates)

    # Add target data
    prices_df = pd.read_csv("stock_prices_2010-2025.csv", parse_dates=["date"]).set_index("date")
    prices_df.index = prices_df.index.normalize()

    X_train, X_test, y_train, y_test, y_std, y_mean = prep_split(prices_df, ticker, market_tensor, dates, start, end)

    train_dataset = DaySequenceDataset(X_train, y_train, seq_len=7)
    test_dataset  = DaySequenceDataset(X_test,  y_test, seq_len=7)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)

    # instantiate
    n_days, M, D = market_tensor.shape
    price_model = PriceLSTM(M, D).to(device)

    optimizer = torch.optim.Adam(price_model.parameters(), lr=1e-4)    
    
    loss_fn = nn.MSELoss()

    train(price_model, train_loader, ticker, epochs=5000)

    # Load existing model?
    load_model = False
    if load_model:
        torch.load(price_model.state_dict(), f"{ticker}_model.pth")
    mse, rmse, preds, y_true = evaluate(price_model, test_loader)
    print(f"Test  — MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    preds  = preds * y_std + y_mean # scale back up
    y_true = y_true * y_std + y_mean

    preds = preds.numpy()
    y_true = y_true.numpy()
    print(f"Preds: {preds}")
    print(f"Y True: {y_true}")

    plot_preds_vs_target(preds, y_true, ticker)