import os
import yaml
import warnings
import pandas as pd

from argparse import ArgumentParser
from modules.scoring import ScoreSeed
from modules.multigraph import NNGraph

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--seed_path", default="data/adform/seed.csv", help="Path to seed ids")
    parser.add_argument("--out_path", default="data/adform/extn.csv", help="Path to save extension ids")

    opt = parser.parse_args()
    with open ("config.yaml") as f:
        config = yaml.load(f)

    nn = config["score"]["nn"]
    eps = float(config["score"]["eps"])
    cat_cols = config["dataset_params"]["cat_cols"]
    list_cols = config["dataset_params"]["list_cols"]
    data_root_dir = config["dataset_params"]["root_dir"]
    data_file = config["dataset_params"]["output_file_name"]
    prob_file = config["dataset_params"]["feat_count_file_name"]

    seed_file = pd.read_csv(opt.seed_path)
    seed_ids = list(seed_file["id"].unique())
    prob_vals = pd.read_csv(os.path.join(data_root_dir, prob_file))
    data_path = os.path.join(data_root_dir, data_file)
    cat_graph = NNGraph(os.path.join(data_root_dir, "cat_graph"))
    list_graphs = []
    for i in list_cols:
        g = NNGraph(os.path.join(data_root_dir, f"{i}_graph"))
        list_graphs.append(g)

    scorer = ScoreSeed(seed_ids, data_path, prob_vals, cat_graph, list_graphs, cat_cols,
                       list_cols, opt.out_path, nn, eps)

    scorer.score()

