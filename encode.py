import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv
load_dotenv()

def create_day_tensor(day_df, max_markets, feature_columns):
    # day_df = df[df["created_date"] == day]
    arr = day_df[feature_columns].values.astype(np.float32)                 # (n_markets_today, D)
    n_today = arr.shape[0]
    if n_today < max_markets:
        padding = np.zeros((max_markets - n_today, arr.shape[1]), dtype=np.float32)
        arr = np.vstack([arr, padding])
    # print(arr.shape)
    return torch.from_numpy(arr)            # (max_markets, D)

def embed_texts(texts, model, batch_size=32):
    """
    texts: N strings
    returns: torch.FloatTensor of shape (N, H)
    """
    # 'convert_to_tensor=True' gives you a torch.Tensor on the right device
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    return embs  # already on `device`, dtype=torch.float32

def preprocess_markets(
    df: pd.DataFrame,
    numeric_cols:    list[str],
    categorical_cols:list[str],
    text_cols:       list[str],
    model,
    device:          torch.device,
    start,
    end,
    load_embeddings = False # load saved embeddings
) -> tuple[torch.Tensor, list[pd.Timestamp]]:
    """
    Returns:
      - tensor of shape (n_days, max_markets, D_total)
      - the sorted list of unique dates
    """
    df = df.copy()
    df["created_date"] = pd.to_datetime(df["created_date"]).dt.normalize()
    df = df.set_index("created_date").sort_index()

    df = df.loc[start:end]
    df.index = df.index.normalize()
    # print(df["created_date"].dtype)

    # news preds
    # news_df = pd.read_csv("news_sentiment_preds.csv", parse_dates=["date"], index_col="date")
    # news_df.index = news_df.index.normalize()

    # df = df.join(news_df, how="left").fillna({"average_prediction": 0})

    # stock prices last 30 days
    prices_df = pd.read_csv("stock_prices_2010-2025.csv", parse_dates=["date"]).set_index("date")
    prices_df.index = prices_df.index.normalize()

    # Fit + transform with StandardScaler
    scaler = StandardScaler()
    spy_prices = prices_df[['SPY']] # 2d
    print(spy_prices)
    prices_df['SPY_norm'] = scaler.fit_transform(spy_prices)

    df['stock_price_today'] = prices_df[['SPY_norm']].reindex(df.index).ffill().bfill()

    print(df.columns)
    print(df.head(3))
    
    # 1) one-hot encode categoricals
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    dummy_cols = [c for c in df.columns
                  if any(c.startswith(cat + "_") for cat in categorical_cols)]

    # 2) normalize numeric columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    means = df[numeric_cols].mean()
    stds  = df[numeric_cols].std().replace(0,1)
    df[numeric_cols] = (df[numeric_cols] - means) / stds
    print(f"Numeric cols: {df[numeric_cols]}\n")

    # 3) embed text columns → new columns col+"_emb0"...col+"_emb{H-1}"
    # collapse all text cols into one and embed once
    # df["all_text"] = (
    #     df[text_cols]                          # your list: ["market_title", …]
    #       .fillna("")                          # drop NaNs
    #       .astype(str)
    #       .agg(" [SEP] ".join, axis=1)        # join each row’s texts
    # )
    df['all_text'] = (
        "The market is: "       + df['market_title'].fillna('')   + ", "
        "the market description is: " + df['market_desc'].fillna('blank') + ", "
        "the event name is: "   + df['event_title'].fillna('')   + ", "
        "the daily yes volume is: "  + df['daily_yes_volume'].fillna(0).astype(str)  + ", "
        "the daily no volume is: "   + df['daily_no_volume'].fillna(0).astype(str)   + ", "
        "the 7 day yes momentum is: " + df['7_day_yes_momentum'].fillna(0).astype(str) + ", "
        "the 7 day no momentum is: "  + df['7_day_no_momentum'].fillna(0).astype(str)
    )

    for i in range(3):
        print(f"All text: {df['all_text'][i]}\n") # debug

    df.index.name = "created_date"
    
    daily_text = (
    df.groupby(level="created_date")["all_text"]
      .agg(" [SEP] ".join)      # join all market texts on that day
    )

    # b) embed those ~750 strings
    daily_embs = embed_texts(
        daily_text.tolist(),
        model,
        batch_size=32
    )  # shape (n_days, H)
    print(f"Daily embs shape: {daily_embs.shape}")

    # c) turn into a DataFrame indexed by date
    H = daily_embs.shape[1]
    emb_df = pd.DataFrame(
        daily_embs.cpu().numpy(),
        index=daily_text.index,
        columns=[f"all_text_emb{i}" for i in range(H)]
    )
    print(f"Embeds shape: {emb_df.shape}\n")

    # d) join back onto every market-row
    df = (
        df
        .join(emb_df, how="left")
        .reset_index()
    )

    # now drop the raw text columns
    df.drop(columns=text_cols + ["all_text"], inplace=True)
    print(f"df columns: {df.columns}")

    # ——— record the new embedding column names ———
    # one col per embedding dim from CLS token
    text_emb_cols = [f"all_text_emb{i}" for i in range(H)]
        
    # final feature list
    feature_cols = numeric_cols + dummy_cols + text_emb_cols + ['stock_price_today']
    
    df = df.reset_index(drop=True)
    # get the sorted array of actual dates
    dates = pd.to_datetime(df["created_date"]).dt.normalize().sort_values().unique()
    max_markets = int(df.groupby("created_date").size().max())

    # 2) For each day: slice your original df, grab the lag row, and call create_day_tensor
    day_tensors = []
    for day in dates:
        day_df = df[df['created_date'] == day]
        day_tensors.append(
            create_day_tensor(
                # need to reset index so that create_day_tensor’s df["created_date"] matches:
                day_df,
                max_markets,
                feature_columns=feature_cols
            )
        )

    # stack into a single (n_days, max_markets, n_features) tensor
    tensor = torch.stack(day_tensors, dim=0)
    print(tensor[0][0])
    print("tensor.shape:", tensor.shape)  # (n_days, max_markets, n_features)

    return tensor, dates

if __name__=="__main__":
    from transformers import AutoTokenizer, AutoModel
    from data_processing.spark_utils import read_parquet, prep_spark

    data = read_parquet()

    categorical_cols = ["category"]
    text_cols = ["market_title","market_subtitle","market_desc","event_title","event_subtitle"]
    exclude = ["open_time", "close_time", "created_date", "market_type"]
    numeric_cols = [col for col in data.columns if col not in (categorical_cols + text_cols + exclude)]


    df = prep_spark(data, numeric_cols).toPandas()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("sentence-transformers/all-distilroberta-v1", 
                                      token=os.getenv("HF_API_KEY"),
                                      device=device)  

    market_tensor, dates = preprocess_markets(
        df,
        numeric_cols,
        categorical_cols,
        text_cols,
        embedding_model,
        device,
        "2024-11-01",
        "2024-11-02",
        load_embeddings=False
    )

    print("Tensor shape:", market_tensor.shape)  # (days, max_markets, D_total)
    print("Dates:", dates)