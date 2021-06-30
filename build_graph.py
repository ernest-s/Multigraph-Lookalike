import os
import warnings
import pandas as pd

from utils import np_pad
from argparse import ArgumentParser
from yaml import CLoader as Loader, load
from modules.multigraph import hellinger_distance, NNGraph, train_graph

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--h_dist", default=1, help="Calculate Hellinger Distance between attributes")

    opt = parser.parse_args()
    with open("config.yaml") as stream:
        config = load(stream, Loader=Loader)

    nn = config["graphs"]["neighbors"]
    n_iter = config["graphs"]["n_iter"]
    stop_thresh = config["graphs"]["stop_thresh"]
    cat_cols = config["dataset_params"]["cat_cols"]
    list_cols = config["dataset_params"]["list_cols"]
    data_root_dir = config["dataset_params"]["root_dir"]
    list_thresh = config["dataset_params"]["list_thresh"]
    data_file = config["dataset_params"]["output_file_name"]

    id_col = "id"

    print("Reading the data...")
    df = pd.read_json(os.path.join(data_root_dir, data_file))
    if int(opt.h_dist):
        print("Calculating Hellinger Distance for all attributes...")
        cols = cat_cols + [id_col]
        df = df[cols]
        df = hellinger_distance(df, cat_cols, id_col)
        df.to_json(os.path.join(data_root_dir, "hell_dist.json"))

    print("Builiding Graph for categorical variables...")
    hash_map = pd.read_json(os.path.join(data_root_dir, "hash_id_map.json"))
    ids = list(hash_map["hash"])
    graph_path = os.path.join(data_root_dir, "cat_graph")
    cat_graph = NNGraph(graph_path, ids, nn)
    weight_path = os.path.join(data_root_dir, "hell_dist.json")
    weights = pd.read_json(weight_path)
    hash_map = hash_map.set_index("hash")
    train_graph(cat_graph, weights, True, hash_map, n_iter, stop_thresh)

    for col in list_cols:
        print(f"Building Graph for variable {col}")
        col_name = f"{col}_hash"
        graph_name = f"{col}_graph"
        hash_map = pd.read_json(os.path.join(data_root_dir, f"{col}_hash_id_map.json"))
        weight_path = os.path.join(data_root_dir, f"{col}_orth.json")
        weights = pd.read_json(weight_path)
        weights["l"] = weights["orth"].apply(lambda x: len(x))
        empty_hash = weights.loc[weights["l"]==0, col_name].iloc[0]
        ids = list(hash_map[col_name])
        ids.remove(empty_hash)
        weights = weights[weights["l"] != 0]
        weights.drop(["l"], axis=1, inplace=True)
        weights = weights.set_index(col_name)
        weights["len"] = weights["orth"].apply(lambda x: len(x))
        weights["orth"] = weights["orth"].apply(lambda x: np_pad(x, list_thresh))
        graph_path = os.path.join(data_root_dir, graph_name)
        graph = NNGraph(graph_path, ids, nn)
        train_graph(graph, weights, False, None, n_iter, stop_thresh)
        graph.save_graph()
