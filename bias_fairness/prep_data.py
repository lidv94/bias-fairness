import pandas as pd
import numpy as np

def impute_nan(df: pd.DataFrame) -> pd.DataFrame:
    df_imputed = df.copy()
    
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df_imputed[col] = df[col].fillna(0)
            else:
                df_imputed[col] = df[col].fillna('Unknown')
    
    return df_imputed


def encode_categorical(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    df_encoded = df.copy()
    
    for col in cat_cols:
        unique_vals = df_encoded[col].dropna().unique()
        n_unique = len(unique_vals)

        if n_unique == 2:
            # Binary encoding: choose most frequent value
            val_counts = df_encoded[col].value_counts(normalize=True)
            top_val = val_counts.idxmax()
            new_col = f"is_{top_val}"
            df_encoded[new_col] = (df_encoded[col] == top_val).astype(int)
            df_encoded.drop(columns=col, inplace=True)

        elif n_unique > 2:
            # One-hot encoding
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False).astype(int)
            df_encoded = pd.concat([df_encoded.drop(columns=col), dummies], axis=1)

    return df_encoded
