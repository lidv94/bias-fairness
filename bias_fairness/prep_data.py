import pandas as pd
import numpy as np
from bias_fairness.utils import timer

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame, nominal_cols: list[str] = None, ordinal_cols: list[str] = None):
        """
        Initialize with DataFrame and optional column lists.
        """
        self.df = df.copy()
        self.nominal_cols = nominal_cols if nominal_cols else []
        self.ordinal_cols = ordinal_cols if ordinal_cols else []

    @timer
    def impute_nan(self) -> pd.DataFrame:
        """
        Fill NaNs: numeric with 0, categorical with 'Unknown'
        """
        for col in self.df.columns:
            if self.df[col].isna().any():
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(0)
                else:
                    self.df[col] = self.df[col].fillna('Unknown')
        return self.df

    @timer
    def encode_nominal_cat(self) -> pd.DataFrame:
        """
        Encode nominal columns: binary as is_X, multi-class as one-hot.
        """
        print("Nominal column value counts:")
        for col in self.nominal_cols:
            print(f"{col}: {self.df[col].value_counts(dropna=False).to_dict()}")

        for col in self.nominal_cols:
            unique_vals = self.df[col].dropna().unique()
            n_unique = len(unique_vals)

            if n_unique == 2:
                val_counts = self.df[col].value_counts(normalize=True)
                top_val = val_counts.idxmax()
                new_col = f"is_{top_val}"
                self.df[new_col] = (self.df[col] == top_val).astype(int)
                self.df.drop(columns=col, inplace=True)
            elif n_unique > 2:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=False).astype(int)
                self.df = pd.concat([self.df.drop(columns=col), dummies], axis=1)
        return self.df

    @timer
    def encode_ordinal_cat(self) -> pd.DataFrame:
        """
        Encode ordinal columns based on frequency (highest gets highest number).
        """
        print("Ordinal column value counts:")
        for col in self.ordinal_cols:
            print(f"{col}: {self.df[col].value_counts(dropna=False).to_dict()}")

        for col in self.ordinal_cols:
            value_counts = self.df[col].value_counts()
            n_unique = len(value_counts)
            mapping = {val: n_unique - 1 - i for i, val in enumerate(value_counts.index)}
            self.df[col + '_enc'] = self.df[col].map(mapping)
            self.df.drop(columns=col, inplace=True)
        return self.df
        
    @timer
    def run_all(self) -> pd.DataFrame:
        """
        Run imputation and both encoding steps.
        """
        self.impute_nan()
        self.encode_nominal_cat()
        self.encode_ordinal_cat()
        return self.df






# import pandas as pd
# import numpy as np

# def impute_nan(df: pd.DataFrame) -> pd.DataFrame:
#     df_imputed = df.copy()
    
#     for col in df.columns:
#         if df[col].isna().any():
#             if pd.api.types.is_numeric_dtype(df[col]):
#                 df_imputed[col] = df[col].fillna(0)
#             else:
#                 df_imputed[col] = df[col].fillna('Unknown')
    
#     return df_imputed


# def encode_nominal_cat(df: pd.DataFrame, nom_cols: list[str]) -> pd.DataFrame:
#     df_encoded = df.copy()
    
#     value_counts_dict = {
#         col: df[col].value_counts(dropna=False).to_dict() for col in nom_cols
#     }
#     print("Value counts per column:\n", value_counts_dict)
    
#     for col in nom_cols:
#         unique_vals = df_encoded[col].dropna().unique()
#         n_unique = len(unique_vals)

#         if n_unique == 2:
#             # Binary encoding: choose most frequent value
#             val_counts = df_encoded[col].value_counts(normalize=True)
#             top_val = val_counts.idxmax()
#             new_col = f"is_{top_val}"
#             df_encoded[new_col] = (df_encoded[col] == top_val).astype(int)
#             df_encoded.drop(columns=col, inplace=True)

#         elif n_unique > 2:
#             # One-hot encoding
#             dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False).astype(int)
#             df_encoded = pd.concat([df_encoded.drop(columns=col), dummies], axis=1)

#     return df_encoded

# def encode_ordinal_cat(df:pd.DataFrame,ord_cols:list[str]) -> pd.DataFrame:
#     df_encoded = df.copy()
    
#     value_counts_dict = {
#         col: df[col].value_counts(dropna=False).to_dict() for col in ord_cols
#     }
#     print("Value counts per column:\n", value_counts_dict)
    
#     for col in ord_cols:
#         value_counts = df[col].value_counts()
#         n_unique = len(value_counts)
#         mapping= {}
#         for i , val in enumerate(value_counts.index):
#             mapping[val] = n_unique - 1 - i
#         df_encoded[col + '_enc'] = df_encoded[col].map(mapping)  
#         df_encoded.drop(columns=[col],inplace=True)
#     return df_encoded

# def apply_encoding_from_dict_onecol(df: pd.DataFrame, 
#                                     col: str, 
#                                     mapping: dict) -> pd.DataFrame:
#     """
#     Apply custom encoding to a single categorical column using a provided mapping.

#     Args:
#         df (pd.DataFrame): Input DataFrame.
#         col (str): Column name to encode.
#         mapping (dict): Dictionary mapping original values to encoded values.

#     Returns:
#         pd.DataFrame: DataFrame with the encoded column added (as '<col>_enc') and the original column dropped.
#     """
#     df_encoded = df.copy()
    
#     if col not in df.columns:
#         raise ValueError(f"Column '{col}' not found in DataFrame.")
    
#     df_encoded[col + '_enc'] = df_encoded[col].map(mapping)
#     df_encoded.drop(columns=[col], inplace=True)
    
#     return df_encoded
