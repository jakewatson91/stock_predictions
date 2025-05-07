
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# function for one-hot-encoding categorical columns
def encode_categorical(df, categorical_columns):
  df_encoded = pd.get_dummies(df, columns=categorical_columns)
  return df_encoded

# function which creates a matrix for each day containing each prediction market and its features
def create_day_tensor(df, day, max_markets, feature_columns):
    day_df = df[df['created_date'] == day]

    features = day_df[feature_columns].values

    if len(features) < max_markets:
        padding = np.zeros((max_markets - len(features), features.shape[1]))
        features = np.vstack([features, padding])

    return torch.tensor(features, dtype=torch.float32)

# function to preprocess and restructure data into torch tensor of shape (days, num_markets, market_features)
def preprocess_categorical(df, categorical_columns, start, end):
  df = df.copy()
  df["created_date"] = pd.to_datetime(df["created_date"])
  mask = df["created_date"].between(start, end)
  df = df.loc[mask]
  
  # one hot encode categorical features
  df = encode_categorical(df, categorical_columns)

  # fill NaN values with 0 and order by date
  df = df.fillna(0)
  df = df.replace({True: 1, False: 0})
  df['created_date'] = pd.to_datetime(df['created_date'])
  df = df.sort_values(by='created_date')

  # restructure in tensor of shape (days, num_markets, market_features)
  unique_days = df['created_date'].unique()
  feature_columns = [col for col in df.columns if col not in ['created_date']]

  max_markets = df.groupby('created_date').size().max()

  tensor_list = []

  for day in unique_days:
    matrix = create_day_tensor(df, day, max_markets, feature_columns)
    tensor_list.append(matrix)

  return torch.stack(tensor_list)

     