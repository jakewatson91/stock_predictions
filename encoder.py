import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

class TimeSeriesTextDataset(Dataset):
    def __init__(
        self,
        df,
        numeric_column_names: list[str],
        categorical_column_names: list[str],
        text_column_names: list[str],
        sequence_length: int = 7,
        text_model_name: str = "bert-base-uncased",
    ):
        """
        Prepares sliding windows of:
          1) numeric features  [seq_len × num_numeric]
          2) categorical codes [seq_len × num_categoricals]
          3) text embeddings   [seq_len × (num_text_cols × bert_dim)]
        paired with the *next* day’s target value (here: SPY close).
        """
        self.sequence_length = sequence_length

        # 1) Raw numeric matrix [#rows × #numeric_cols]
        self.numeric_matrix = (
            df[numeric_column_names].to_numpy(dtype="float32")
        )

        # 2) Convert each categorical column to integer codes
        self.categorical_matrix = (
            df[categorical_column_names]
            .astype("category")
            .apply(lambda col: col.cat.codes)
            .to_numpy(dtype="int64")
        )

        # 3) List-of-lists: each row → list of text fields
        self.text_lists = df[text_column_names].fillna("").values.tolist()

        # 4) Targets: next‐day SPY close price
        self.targets = df["SPY"].to_numpy(dtype="float32")

        # load HF tokenizer + model once
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

    def __len__(self):
        # can index from 0 .. (#rows − seq_len − 1)
        return len(self.targets) - self.sequence_length

    def __getitem__(self, idx: int):
        # slice out the window [idx : idx+seq_len]
        num_window = torch.from_numpy(
            self.numeric_matrix[idx : idx + self.sequence_length]
        )                                # shape: [T, num_numeric]

        cat_window = torch.from_numpy(
            self.categorical_matrix[idx : idx + self.sequence_length]
        )                                # shape: [T, num_categorical]

        # flatten all text fields across the T days
        flat_texts = sum(self.text_lists[idx : idx + self.sequence_length], [])
        token_batch = self.tokenizer(
            flat_texts, padding=True, truncation=True, return_tensors="pt"
        )
        encoded = self.text_encoder(**token_batch)
        # [T * num_text_cols, bert_hidden]
        pooled = encoded.pooler_output  
        # reshape → [T, num_text_cols * bert_hidden]
        text_window = pooled.view(self.sequence_length, -1)

        # the *next* day’s SPY is our regression target
        target_value = torch.tensor(self.targets[idx + self.sequence_length])

        return num_window, cat_window, text_window, target_value


class TimeSeriesWithTextLSTM(nn.Module):
    def __init__(
        self,
        numeric_column_names: list[str],
        text_column_names: list[str],
        categorical_cardinalities: list[int],
        categorical_embedding_dim: int,
        text_embedding_dim: int,
        lstm_hidden_dim: int,
        num_lstm_layers: int = 1,
    ):
        """
        - Learn a tiny embedding per numeric field (so the model “knows” which column is which).
        - Learn embeddings for truly categorical columns.
        - Project BERT’s pooled text outputs down if desired.
        - Feed everything into an LSTM, then regress to a scalar.
        """
        super().__init__()

        # -- 1) Field (header) embeddings for numeric cols --
        self.num_numeric_fields = len(numeric_column_names)
        self.field_embedding_dim = 16
        self.field_embeddings = nn.Embedding(
            num_embeddings=self.num_numeric_fields,
            embedding_dim=self.field_embedding_dim,
        )

        # -- 2) Embeddings for each categorical feature --
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_categories, categorical_embedding_dim)
            for num_categories in categorical_cardinalities
        ])

        # -- 3) (Optional) project concatenated BERT outputs → text_emb_dim --
        self.text_projection = nn.Linear(
            in_features=text_embedding_dim * len(text_column_names),
            out_features=text_embedding_dim,
        )

        # total dimension per time‐step:
        total_step_dim = (
            # raw numeric dims + header‐embedding dims
            self.num_numeric_fields + self.field_embedding_dim
            # + all categorical embeddings
            + len(categorical_cardinalities) * categorical_embedding_dim
            # + text embedding dim
            + text_embedding_dim
        )

        # -- 4) the LSTM encoder --
        self.lstm = nn.LSTM(
            input_size=total_step_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # -- 5) final regressor head from last time‐step hidden state to a scalar --
        self.regressor = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, numeric_inputs, categorical_inputs, text_inputs):
        """
        numeric_inputs : [batch_size, T, num_numeric]
        categorical_inputs : [batch_size, T, num_categorical]
        text_inputs : [batch_size, T, text_embedding_dim]
        """
        B, T, _ = numeric_inputs.shape

        # a) field‐wise header embeddings → repeat across batch & time
        field_ids = torch.arange(self.num_numeric_fields, device=numeric_inputs.device)
        # [num_numeric] → [1,1,num_numeric] → [B,T,num_numeric]
        field_ids = field_ids.view(1,1,-1).expand(B, T, -1)
        header_embs = self.field_embeddings(field_ids)  # [B,T,num_numeric × field_emb_dim?]

        # b) categorical embeddings: look up each feature and concat
        cat_emb_list = []
        for col_idx, emb in enumerate(self.categorical_embeddings):
            # pick out the j-th categorical column → embed
            cat_emb_list.append(emb(categorical_inputs[..., col_idx]))
        cat_embs = torch.cat(cat_emb_list, dim=-1)      # [B,T, C*cat_emb_dim]

        # c) project text if needed
        text_proj = self.text_projection(text_inputs)  # [B,T,text_embedding_dim]

        # d) assemble per-step features
        #   [numeric | header‐emb | cat‐emb | text‐emb]
        step_features = torch.cat(
            [numeric_inputs, header_embs, cat_embs, text_proj],
            dim=-1
        )  # shape: [B, T, total_step_dim]

        # e) LSTM encode
        lstm_out, _ = self.lstm(step_features)  # [B, T, lstm_hidden_dim]
        final_hidden = lstm_out[:, -1, :]       # last time‐step [B, hidden]

        # f) regress to a scalar
        return self.regressor(final_hidden).squeeze(-1)  # [B]


# ——— Example usage ———
df
numeric_cols = []  # e.g. ["yes_price_close", "no_price_close", ...]
categorical_cols = ["market_type", "category"]
text_cols        = ["market_title","market_subtitle","market_desc","event_title","event_subtitle"]

# 1) prepare dataset + loader
dataset = TimeSeriesTextDataset(
    df=df,
    numeric_column_names=numeric_cols,
    categorical_column_names=categorical_cols,
    text_column_names=text_cols,
    sequence_length=7,
)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# 2) build model
model = TimeSeriesWithTextLSTM(
    numeric_column_names=numeric_cols,
    categorical_cardinalities=[df[c].nunique() for c in categorical_cols],
    categorical_embedding_dim=8,
    text_embedding_dim=256,   # must match your BERT’s pooler_output dim
    lstm_hidden_dim=128,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# 3) in your train loop, each batch is:
#    num_batch, cat_batch, txt_batch, y_batch = batch
#    preds = model(num_batch, cat_batch, txt_batch)
#    loss = loss_fn(preds, y_batch); ...