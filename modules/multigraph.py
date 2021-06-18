import torch
import pickle
import random
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import col_melt

warnings.filterwarnings("ignore")


class NNGraph:
    """
    Nearest Neighbor Graph Implementation
    """
    def __init__(self, path, ids=None, n=10):
        self.path = path
        if ids is None:
            self.load_graph()
        else:
            self.n = n
            self.ids = ids
            self.path = path
            self.graph = {}
            self.__build_initial_graph__()
            self.trained = False

    def __build_initial_graph__(self):
        for i in self.ids:
            self.graph[i] = random.sample(self.ids, self.n)

    def graph_update(self, ids, neighbors):
        for i, vertex in enumerate(ids):
            self.graph[vertex] = neighbors[i]

    def save_graph(self):
        graph_file = {"n":self.n, "ids":self.ids, "graph":self.graph, "trained":self.trained}
        with open(self.path, "wb") as f:
            pickle.dump(graph_file, f)

    def load_graph(self):
        graph_file = pickle.load(open(self.path, "rb"))
        self.n = graph_file["n"]
        self.ids = graph_file["ids"]
        self.graph = graph_file["graph"]
        self.trained = graph_file["trained"]

    def extract_neighbors(self, v, depth=1):
        neighbors = []
        if v not in self.graph:
            return [], None
        for i in range(depth):
            if i == 0:
                neighbors.extend(self.graph[v])
                last = self.graph[v][-1]
            else:
                n_d =[]
                for n in neighbors:
                    n_d.extend(self.graph[n])
                neighbors.extend(n_d)
        return list(set(neighbors)), last


def train_graph(graph, weights, categorical=False, df=None, n_iter=20, stop_thresh=0.5):

    for i in range(n_iter):
        n_changes = 0
        vertex_len = len(graph.graph)
        print(f"Iteration {i+1} of {n_iter}")
        for v in tqdm(graph.graph):
            neighbors, last = graph.extract_neighbors(v, depth=2)
            if categorical:
                v_data, n_data = get_data(v, neighbors, df)
                new_neighbors = find_nearest_cat(v_data, n_data, neighbors, weights, last, graph.n, graph.trained)
            else:
                new_neighbors = find_nearest_list(v, neighbors, weights, last, graph.n, graph.trained)
            if new_neighbors != "No Change":
                n_changes += len(set(neighbors)-set(new_neighbors))
                graph.graph[v] = new_neighbors
            else:
                n_changes += 0
        avg_change = n_changes/(vertex_len * graph.n)
        graph.trained = True
        graph.save_graph()
        print(f"Average neighbor change per vertex: {round(avg_change, 3)}")
        if avg_change < stop_thresh:
            print("Early stopping: Average change per vertex is less than threshold.")
            break


def find_nearest_cat(v_data, n_data, neighbors, weights, last, nn=10, check=False):
    dist = []

    for n in n_data:
        sq_dist = 0.0
        for i,j in enumerate(n):
            if (j is None) | (v_data[i] is None):
                sq_dist += 1.0
            else:
                sq_dist += weights.loc[j,v_data[i]]
        dist.append(np.sqrt(sq_dist))
    if check:
        idx = neighbors.index(last)
        prev_best = dist[idx]
    dist = [(neighbors[i],j) for i,j in enumerate(dist)]
    dist.sort(key=lambda x: x[1])
    if check:
        if dist[nn-1][1] >= prev_best:
            return "No Change"
    dist = [i[0] for i in dist][:nn]
    return dist

def find_nearest_list(v, neighbors, weights, last, nn=10, check=False):
    m1 = weights.loc[v,"orth"]
    m2 = np.array(list(weights.loc[neighbors, "orth"]))

    v_length = weights.loc[v, "len"]
    n_length = np.array(list(weights.loc[neighbors, "len"]))
    n_length[n_length < v_length] = v_length
    dist = modified_chordal_distance(m1, m2, n_length)
    if check:
        idx = neighbors.index(last)
        prev_best = dist[idx]
    dist = [(neighbors[i],j) for i,j in enumerate(dist)]
    dist.sort(key=lambda x: x[1])
    if check:
        if dist[nn-1][1] >= prev_best:
            return "No Change"
    dist = [i[0] for i in dist][:nn]
    return dist


def modified_chordal_distance(m1, m2, n_length):
    m1 = torch.from_numpy(m1).to("cuda").float()
    m2 = torch.from_numpy(m2).to("cuda").float()
    r = torch.einsum('ik,lkj->lij', [m1.T, m2])
    r = r.sum(axis=[1, 2])
    return list(np.sqrt(np.abs(n_length - r.cpu().detach().numpy())))


def get_data(v, neighbors, df):
    x1 = df.loc[v,"values"]
    x2 = list(df.loc[neighbors, "values"])
    return x1, x2


def hellinger_distance(df, cols, id_col):
    """
    Calculates the square of Hellinger Distance between all features in a dataframe
    :param df:
    :param cols:
    :param id_col:
    :return:
    """
    df = df.fillna("M")
    counts = df.groupby(cols)[id_col].count().reset_index()
    counts_m = pd.DataFrame()
    for col in cols:
        counts_m = pd.concat([counts_m, col_melt(col, counts)])
    counts_m = counts_m[counts_m["var1"] != "M"]
    counts_m = counts_m[counts_m["var2"] != "M"]
    counts_m = counts_m.groupby(["var1", "var2"])["count"].sum().reset_index()
    counts_m = counts_m.pivot(index='var1', columns='var2')['count']
    cols = list(counts_m.columns)
    idx = list(counts_m.index)
    counts_m.fillna(0, inplace=True)
    counts_arr = np.array(counts_m)
    counts_arr = np.sqrt(counts_arr / counts_arr.sum(axis=0))
    dist = np.matmul(counts_arr.T, counts_arr)
    np.fill_diagonal(dist, 0)
    dist[dist > 1] = 1.0
    dist[dist < 0] = 0.0
    dist = 1.0 - dist
    np.fill_diagonal(dist, 0)
    dist_df = pd.DataFrame(dist, columns=cols, index=idx)
    return dist_df
