import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

def create_day_tensor(df, day, max_markets, feature_columns):
    day_df = df[df["created_date"] == day]
    arr = day_df[feature_columns].values.astype(np.float32)                 # (n_markets_today, D)
    n_today = arr.shape[0]
    if n_today < max_markets:
        padding = np.zeros((max_markets - n_today, arr.shape[1]), dtype=np.float32)
        arr = np.vstack([arr, padding])
    return torch.from_numpy(arr)            # (max_markets, D)

def embed_texts(texts, tokenizer, model, device, batch_size=32):
    model.eval()
    print(f">>> embed_texts running on {device}, CUDA available={torch.cuda.is_available()}")

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
        return_attention_mask=True
    )
    # 2) Move once to GPU
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

    batch_embs = []
    # 3) Chunk your already-tokenized tensors
    total = enc["input_ids"].size(0) # total = N
    for i in tqdm(range(0, total, batch_size), desc="Encoding", unit="batch"):
        batch = {k: v[i : i + batch_size] for k, v in enc.items()} # batch["input_ids"].shape = (Bᵢ, L)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            out = model(**batch) # out.last_hidden_state.shape = (Bᵢ, L, H)
            cls = out.last_hidden_state[:, 0, :]  # (B, H)
        batch_embs.append(cls.cpu())
        # debug print for first batch
        if i == 0:
            print(f"batch 0 size: {cls.size()}, dtype: {cls.dtype}")

    embs = torch.cat(batch_embs, dim=0)  # embs.shape = (∑ᵢ Bᵢ, H) = (N, H)
    
    print("Saving embeddings...\n")
    torch.save(embs, "all_text_embeddings.pt")
    print("Embeddings saved as all_text_embeddings.pt\n")
    
    # print(f"ALL EMBS SHAPE: {embs.size()}\n\n")
    # print(f"CAT SHAPE: {embs.size()}\n\n")
    # 4) Concatenate and return
    return embs

def preprocess_markets(
    df: pd.DataFrame,
    numeric_cols:    list[str],
    categorical_cols:list[str],
    text_cols:       list[str],
    tokenizer,
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
    df["created_date"] = pd.to_datetime(df["created_date"])
    mask = df["created_date"].between(start, end)
    df = df.loc[mask]

    # 1) one-hot encode categoricals
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    dummy_cols = [c for c in df.columns
                  if any(c.startswith(cat + "_") for cat in categorical_cols)]

    # 2) normalize numeric columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    means = df[numeric_cols].mean()
    stds  = df[numeric_cols].std().replace(0,1)
    df[numeric_cols] = (df[numeric_cols] - means) / stds

    # 3) embed text columns → new columns col+"_emb0"...col+"_emb{H-1}"
    # collapse all text cols into one and embed once
    df["all_text"] = (
        df[text_cols]
        .fillna("")                   # replace NaN with ""
        .agg(" [SEP] ".join, axis=1)  # join columns with a separator
    )
    texts = df["all_text"].astype(str).tolist()

    if load_embeddings:
        embeddings = torch.load("all_text_embeddings.pt")
    else: 
        embeddings = embed_texts(texts, tokenizer, model, device)  # (N, H)
    
    arr = embeddings.cpu().numpy()                              # (N, H)
    H = arr.shape[1]
    col_names = [f"all_text_emb{i}" for i in range(H)]
    # build a separate DataFrame of all H embedding columns
    emb_df = pd.DataFrame(
        arr,
        index=df.index,    # same row order
        columns=col_names  # e.g. ["all_text_emb0", …]
    )
    #  drop the raw text + helper from your main df
    df_reduced = df.drop(columns=text_cols + ["all_text"])
    # concat in one go (no fragmentation)
    df = pd.concat([df_reduced, emb_df], axis=1)
    # update feature list
    text_emb_cols = col_names
        
    # final feature list
    feature_cols = numeric_cols + dummy_cols + text_emb_cols
    # sorted list of days
    dates = sorted(df["created_date"].unique())

    # how many markets at most on any single day
    max_markets = int(df["created_date"].value_counts().max())

    # build one (max_markets × n_features) tensor per day
    day_tensors = [
        create_day_tensor(df, day, max_markets, feature_cols)
        for day in dates
    ]

    # stack into a single (n_days, max_markets, n_features) tensor
    tensor = torch.stack(day_tensors, dim=0)
    # print("tensor.shape:", tensor.shape)  # (n_days, max_markets, n_features)

    return tensor, dates

if __name__=="__main__":
    from transformers import AutoTokenizer, AutoModel
    from spark_utils import read_parquet, prep_spark

    data = read_parquet()

    categorical_cols = ["category"]
    text_cols = ["market_title","market_subtitle","market_desc","event_title","event_subtitle"]
    exclude = ["open_time", "close_time", "created_date", "market_type"]
    numeric_cols = [col for col in data.columns if col not in (categorical_cols + text_cols + exclude)]


    df = prep_spark(data, numeric_cols).toPandas()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

    market_tensor, dates = preprocess_markets(
        df,
        numeric_cols,
        categorical_cols,
        text_cols,
        tokenizer,
        model,
        device,
        "2025-01-01",
        "2025-04-25",
        load_embeddings=False
    )

    print("Tensor shape:", market_tensor.shape)  # (days, max_markets, D_total)
    print("Dates:", dates)