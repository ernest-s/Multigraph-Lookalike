import os
import warnings
import pandas as pd

from itertools import chain
from utils import hash_values
from collections import Counter
from yaml import CLoader as Loader, load

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    with open("config.yaml") as stream:
        config = load(stream, Loader=Loader)

    root_dir = config["dataset_params"]["root_dir"]
    cat_cols = config["dataset_params"]["cat_cols"]
    list_cols = config["dataset_params"]["list_cols"]
    freq_thresh = config["dataset_params"]["freq_thresh"]
    list_thresh = config["dataset_params"]["list_thresh"]
    card_thresh = config["dataset_params"]["card_thresh"]
    input_file_name = config["dataset_params"]["input_file_name"]
    output_file_name = config["dataset_params"]["output_file_name"]
    freq_thresh_list = config["dataset_params"]["freq_thresh_list"]
    feat_count_file_name = config["dataset_params"]["feat_count_file_name"]

    print("Reading the file...")
    df = pd.read_json(os.path.join(root_dir, input_file_name), lines=True)

    col_names = list(df.columns)[1:]

    print("Converting columns to numeric...")
    # Convert columns from list to numeric
    for col in col_names:
        if col not in list_cols:
            df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) else None)

    print("Removing columns with very high cardinality...")
    # Remove columns having cardinality more than the threshold
    del_cols = []
    for col in col_names:
        if col not in list_cols:
            if len(df[col].unique()) > card_thresh:
                del_cols.append(col)
    df.drop(del_cols, axis=1, inplace=True)

    print("Removing low frequency values...")
    # Remove rare categories from data (only non list columns)
    # Remove the value if the frequency is less than freq_thresh
    new_cols = list(set(df.columns[1:])-set(list_cols))
    df[new_cols] = df[new_cols].where(df[new_cols].apply(lambda x: x.map(x.value_counts())) >= freq_thresh, None)

    print("Replace missing values in list columns with empty list...")
    for col in list_cols:
        df[col] = df[col].apply(lambda d: d if isinstance(d, list) else [])

    print("Removing entries with low frequency in list columns...")
    # For the list columns remove entries with low frequency
    for col in list_cols:
        # calculate the frequency of every element in the list column
        freq_counts = pd.Series(Counter(chain.from_iterable(x for x in df[col])))
        # filter items which don't meet the criteria
        remove_elements = set(freq_counts[freq_counts <= freq_thresh_list].index)
        # Remove those elements from the set
        df[col] = df[col].apply(lambda x: list(set(x)-remove_elements))

    print("Converting hashed values to strings...")
    # Convert the hashed values to string values
    for col in df.columns[1:]:
        if col in list_cols:
            val_list = list(set(chain(*df[col])))
            map_dict = {i: f"{col}_{val_list.index(i)}" for i in val_list}
            df[col] = df[col].apply(lambda x:list(filter(None,map(map_dict.get,x))))
        else:
            val_list = list(df[col].unique())
            if None in val_list:
                val_list.remove(None)
            map_dict = {i: f"{col}_{val_list.index(i)}" for i in val_list}
            df[col] = df[col].map(map_dict)

    print("Applying threshold for list columns...")
    # If any row has list columns with elements more than the threshold, remove them
    for col in list_cols:
        df = df[df[col].apply(lambda x: len(x)) <= list_thresh]
        # calculate the frequency of every element in the list column after removing rows
        freq_counts = pd.Series(Counter(chain.from_iterable(x for x in df[col])))
        # filter items which don't meet the criteria
        remove_elements = set(freq_counts[freq_counts <= freq_thresh_list].index)
        # Remove those elements from the set
        df[col] = df[col].apply(lambda x: list(set(x) - remove_elements))
        # Sort the elements
        df[col] = df[col].apply(lambda x: sorted(x))


    print("Calculating feature frequencies...")
    feature_count = pd.DataFrame(columns=[ "value", "count", "feature"])
    for col in df.columns[1:]:
        if col in list_cols:
            freq_count = pd.Series(Counter(chain.from_iterable(x for x in df[col])))
        else:
            freq_count = df[col].value_counts()
        freq_count = pd.DataFrame(freq_count).reset_index()
        freq_count.columns = ["value", "count"]
        freq_count["feature"] = col
        feature_count = pd.concat([feature_count, freq_count])
    feature_count.reset_index(drop=True, inplace=True)
    feature_count["prob"] = feature_count["count"] / df.shape[0]
    feat_file = os.path.join(root_dir, feat_count_file_name)
    feature_count.to_csv(feat_file, index=False)

    # Create feature hashes for graph building
    df = hash_values(df, list_cols, "l")
    df.reset_index(inplace=True)
    df = df.rename({"l": "click", "index": "id"}, axis=1)
    print("Writing the processed data to disc...")
    output_file = os.path.join(root_dir, output_file_name)
    hash_file = df.groupby(["hash"])["id"].apply(list).reset_index()
    cat_values = df[cat_cols +["hash"]].drop_duplicates().reset_index(drop=True)
    l1 = cat_values[cat_cols].values.tolist()
    l2 = cat_values["hash"]
    cat_values = pd.DataFrame({"hash": l2, "values": l1})
    hash_file = hash_file.merge(cat_values, on="hash", how="left")
    hash_file.to_json(os.path.join(root_dir, "hash_id_map.json"))
    for col in list_cols:
        hash_col = f"{col}_hash"
        hash_file = df.groupby([hash_col])["id"].apply(list).reset_index()
        hash_vals = df[[hash_col, col]]
        hash_vals = hash_vals.drop_duplicates(hash_col)
        hash_file = hash_file.merge(hash_vals, on=hash_col, how="left")
        hash_file.to_json(os.path.join(root_dir, f"{col}_hash_id_map.json"))
    df.to_json(output_file)
