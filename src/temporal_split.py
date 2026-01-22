def temporal_split(df, timestamp_col, train_ratio=0.5):
    df = df.sort_values(timestamp_col)
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]
