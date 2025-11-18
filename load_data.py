import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path = "data/1presidential_speeches_with_metadata.xlsx"):
    df = pd.read_excel(path)
    print(f"[INFO] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def split_data(df, test_size=0.10, val_size=0.10, random_state=42):

    df_temp, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    relative_val_size = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_temp,
        test_size=relative_val_size,
        random_state=random_state,
        shuffle=True
    )

    print(f"  Train: {df_train.shape[0]} rows")
    print(f"  Val:   {df_val.shape[0]} rows")
    print(f"  Test:  {df_test.shape[0]} rows")

    return df_train, df_val, df_test

def load_and_split_data(path):
    df = load_data(path)
    df_train, df_val, df_test = split_data(df)
    return df_train, df_val, df_test