import pandas as pd

pd.set_option("display.max_columns", None)  # show all columns
pd.set_option("display.width", 0)           # auto-wrap for wide output

df = pd.read_parquet("trades_historical_trades_2025-04-07T14.parquet")
print(df.head())
print(df.columns)
print(df.shape)