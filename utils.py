import hashlib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def hash_values(df, list_cols, labels):
    """
    Hash the values in a dataframe
    """
    other_cols = [col for col in df.columns if col not in list_cols + [labels]]
    # Hash the list columns
    for col in list_cols:
        new_col = f"{col}_hash"
        df[new_col] = df[col].apply(lambda x: hashlib.sha256(" ".join(x).encode()).hexdigest())
        df = hash2category(df, new_col)
    # Hash other columns
    df.reset_index(drop=True, inplace=True)
    df["hash"] = pd.Series(df[other_cols].fillna("NA").values.tolist()).str.join(" ")
    df["hash"] = df["hash"].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
    df = hash2category(df, "hash")
    return df

def hash2category(df, col):
    hash_map = {j: f"{col}_{str(i)}" for i,j in enumerate(df[col].unique())}
    df[col] = df[col].map(hash_map)
    return df

def col_melt(col, df):
    value_vars = [i for i in df.columns if i not in [col, "id"]]
    df = pd.melt(df, id_vars=[col, "id"], value_vars = value_vars)
    df.drop(["variable"], axis=1, inplace=True)
    df = df[[col, "value", "id"]]
    df.columns = ["var1", "var2", "count"]
    return df

def np_pad(x, list_thresh):
    """
    Pad zero values for the orthogonal matrix to make it uniform shape. This is essential
    to calculate Chordal Distance in batches.
    """
    x = np.array(x)
    x = np.pad(x, pad_width = ((0,0),(0,list_thresh-x.shape[1])), mode="constant", constant_values=0)
    return x

def calc_score(n, prob_vals, cat_cols, list_cols):
    """
    Function calculates the score of a seed set neighbor given its feature and weight set
    """
    score = 0
    for c in cat_cols:
        if n[c] is not None:
            score += prob_vals[n[c]]
    for c in list_cols:
        for d in n[c]:
            score += prob_vals[d]
    score *= n["Nc"]
    score *= n["Ec"]
    return score

