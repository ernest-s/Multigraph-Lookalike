import warnings
import numpy as np
import pandas as pd

from itertools import chain
from collections import Counter

from utils import calc_score


warnings.filterwarnings("ignore")


class ScoreSeed:

    """
    Scorer that given seed set and a multigraph, extracts the neighbors of the seed set
    and scores them
    """

    def __init__(self, seed_ids, data_path, prob_vals, cat_graph, list_graphs, cat_cols, list_cols,
                 out_path = "data/adform/extn.csv", nn=2, eps=1e-6):
        self.seed_ids = seed_ids
        self.data_path = data_path
        self.prob_vals = prob_vals
        self.cat_graph = cat_graph
        self.list_graphs = list_graphs
        self.cat_cols = cat_cols
        self.list_cols = list_cols
        self.cols = self.cat_cols + self.list_cols
        self.weighted_iv = None
        self.eps = eps
        self.nn = nn
        self.out_path = out_path
        self.scored = False
        self.extension = []

    def score(self):
        print("Reading data...")
        df = pd.read_json(self.data_path)
        print("Calculating Information Value...")
        self.__calc_seed_prob__(df)
        self.__calculate_iv__()
        print("Extracting Neighbors...")
        neighbors = self.__retrieve_neighbors__(df)
        neighbors = neighbors.merge(df, on=["id"], how="left")
        print("Scoring Neighbors...")
        extn = self.__rank_neighbors__(neighbors)
        self.extension.extend(extn)
        extn = pd.DataFrame({"id": self.extension})
        print("Saving output file...")
        extn.to_csv(self.out_path, index=False)
        self.scored = True

    def __calc_seed_prob__(self, df):
        """
        Calculate feature probabilities for scoring
        """
        print("Calculating seed set distribution...")
        df = df[df["id"].isin(self.seed_ids)]
        feature_count = pd.DataFrame(columns=["value", "count", "feature"])
        for col in self.cols:
            if col in self.list_cols:
                freq_count = pd.Series(Counter(chain.from_iterable(x for x in df[col])))
            else:
                freq_count = df[col].value_counts()
            freq_count = pd.DataFrame(freq_count).reset_index()
            freq_count.columns = ["value", "count"]
            freq_count["feature"] = col
            feature_count = pd.concat([feature_count, freq_count])
        feature_count.reset_index(drop=True, inplace=True)
        feature_count["s_prob"] = feature_count["count"] / df.shape[0]
        feature_count.drop("count", axis=1, inplace=True)
        prob_vals = self.prob_vals
        prob_vals = prob_vals.merge(feature_count, on=["value", "feature"], how="outer")
        prob_vals.fillna(0, inplace=True)
        self.prob_vals = prob_vals

    def __calculate_iv__(self):
        """
        Calculate the information value for all the features
        """
        prob_vals = self.prob_vals
        prob_vals["iv"] = prob_vals.apply(lambda x: iv_(x, self.eps), axis=1)
        iv = prob_vals[["feature", "iv"]]
        iv = iv.groupby(["feature"])["iv"].sum().reset_index()
        prob_vals.drop("iv", axis=1, inplace=True)
        prob_vals = prob_vals.merge(iv, on=["feature"], how="outer")
        prob_vals["weighted_iv"] = prob_vals["s_prob"] * prob_vals["iv"]
        prob_vals.fillna(0, inplace=True)
        self.prob_vals = dict(zip(prob_vals["value"], prob_vals["weighted_iv"]))

    def __rank_neighbors__(self, neighbors):
        """
        Given a set of neighbors, the method ranks them based on their closeness to seed set
        """
        neighbors["score"] = neighbors.apply(lambda x: calc_score(x, self.prob_vals, self.cat_cols, self.list_cols),
                                             axis=1)
        neighbors.sort_values("score", ascending=False, inplace=True)
        self.neighbors = neighbors
        return list(neighbors["id"])

    def __retrieve_neighbors__(self, df):
        """
        Retrieves the neighbors of seed set from the multigraph
        """
        df1 = df[df["id"].isin(self.seed_ids)]
        cols = ["hash"] + [f"{c}_hash" for c in self.list_cols]
        col_hash = df1[["id", "hash"]]
        neighbors = []
        for h in col_hash["hash"]:
            neighbors.extend(self.cat_graph.extract_neighbors(h, self.nn)[0])
        hash_vals = pd.DataFrame({"neighbors": neighbors})
        hash_vals["hash_type"] = "hash"
        hash_vals = hash_vals.groupby("neighbors")["hash_type"].count().reset_index()
        hash_vals.columns = ["neighbors", "Nc"]
        hash_vals["hash_type"] = "hash"
        hash_vals = hash_vals.merge(df[["id", "hash"]], left_on="neighbors", right_on="hash", how="left")
        hash_vals.drop("hash", axis=1, inplace=True)
        for i, c in enumerate(cols[1:]):
            col_hash = df1[["id", c]]
            neighbors = []
            for h in col_hash[c]:
                neighbors.extend(self.list_graphs[i].extract_neighbors(h, self.nn)[0])
            hv = pd.DataFrame({"neighbors": neighbors})
            hv["hash_type"] = c
            hv = hv.groupby("neighbors")["hash_type"].count().reset_index()
            hv.columns = ["neighbors", "Nc"]
            hv["hash_type"] = c
            hv = hv.merge(df[["id", c]], left_on="neighbors", right_on=c, how="left")
            hv.drop(c, axis=1, inplace=True)
            hash_vals = pd.concat([hash_vals, hv])
        hash_vals.groupby("id")["Nc"].max().reset_index()
        edge_count = hash_vals[["id", "hash_type"]].drop_duplicates()
        edge_count = edge_count.groupby("id")["hash_type"].count().reset_index()
        edge_count.columns = ["id", "Ec"]
        neighbors = hash_vals.merge(edge_count, on=["id"], how="outer")
        neighbors.fillna(0, inplace=True)
        neighbors = neighbors[["id", "Nc", "Ec"]]
        neighbors["Nc"] = neighbors.groupby("id")["Nc"].transform(max)
        neighbors = neighbors.drop_duplicates()
        return neighbors

    def similar_users(self, k=30):
        """
        Given a seed set of size m, the method gets the top kXm neighbors
        """
        if not self.scored:
            print("score method has to be called before retrieving neighbors")
        else:
            return self.extension[:len(self.seed_ids)*k]


def iv_(x, eps):
    """
    Function to calculate the information value
    """
    pu = x["prob"]
    ps = x["s_prob"]
    iv = (pu-ps) * np.log((pu+eps)/(ps+eps))
    return iv
