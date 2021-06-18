import os
import warnings
import numpy as np
import pandas as pd
import scipy.linalg

import torch
from torch.autograd import Variable

from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

class CoOccurrenceOptimization():
    """
    Class contains all the necessary information for operating on the co-occurrence matrix
    like weights, left and right matrices to be used, and biases.
    """

    def __len__(self):
        """
        Returns number of items.
        """
        return self.n_obs

    def __init__(self, base_std, x_max, alpha, n_embedding, file_path, random_state=0):
        """
        Initializes variables and sets parameters.
        """

        torch.manual_seed(random_state)

        # Read the co-occurrence matrix
        df = pd.read_csv(file_path)

        # Create indices for the items
        unique_items = list(set(df["item1"]).union(set(df["item2"])))
        unique_items.sort()
        self.item_dict = {i: unique_items.index(i) for i in unique_items}
        self.inv_item_dict = {v: k for k, v in self.item_dict.items()}
        self.n_items = len(self.item_dict)

        # Convert item columns to integer using item map
        df["item1"] = df["item1"].map(self.item_dict)
        df["item2"] = df["item2"].map(self.item_dict)
        df["count"] = df["count"].astype(float)

        n_occurrences = df["count"].tolist()
        n_occurrences = np.array(n_occurrences)

        left = df["item1"].tolist()
        right = df["item2"].tolist()

        self.n_obs = len(left)

        # creating the variables
        self.L_words = cuda(torch.LongTensor(left))
        self.R_words = cuda(torch.LongTensor(right))

        self.weights = np.minimum((n_occurrences / x_max) ** alpha, 1)
        self.weights = Variable(cuda(torch.FloatTensor(self.weights)))

        self.y = Variable(cuda(torch.FloatTensor(np.log(n_occurrences))))

        # Creating embeddings and biases
        l_vecs = cuda(torch.randn((self.n_items, n_embedding)) * base_std)
        r_vecs = cuda(torch.randn((self.n_items, n_embedding)) * base_std)
        l_biases = cuda(torch.randn((self.n_items,)) * base_std)
        r_biases = cuda(torch.randn((self.n_items,)) * base_std)

        self.all_params = [Variable(e, requires_grad=True)
                           for e in (l_vecs, r_vecs, l_biases, r_biases)]
        self.l_vecs, self.r_vecs, self.l_biases, self.r_biases = self.all_params

    def save_embedding(self, file_path):
        avg_embedding = (self.all_params[0] + self.all_params[1])/2
        embed_df = pd.DataFrame(avg_embedding.cpu().detach().numpy()).reset_index()
        embed_df.rename({"index": "item"}, axis=1, inplace=True)
        embed_df["item"] = embed_df["item"].map(self.inv_item_dict)
        embed_df.to_csv(file_path, index=False)


def gen_batches(data, batch_size=2048):

    """generates batches for training"""
    indices = torch.randperm(len(data))
    indices = indices.cuda()

    for idx in range(0, len(data) - batch_size + 1, batch_size):
        sample = indices[idx:idx + batch_size]
        l_words, r_words = data.L_words[sample], data.R_words[sample]
        l_vecs = data.l_vecs[l_words]
        r_vecs = data.r_vecs[r_words]
        l_bias = data.l_biases[l_words]
        r_bias = data.r_biases[r_words]
        weight = data.weights[sample]
        y = data.y[sample]
        yield weight, l_vecs, r_vecs, y, l_bias, r_bias


def embed_loss(weight, l_vecs, r_vecs, log_covals, l_bias, r_bias):
    """calculates the mean squared error"""
    sim = (l_vecs * r_vecs).sum(1).view(-1)
    x = (sim + l_bias + r_bias - log_covals) ** 2
    loss = torch.mul(x, weight)
    return loss.mean()

def cuda(x):
    if True:
        return x.cuda()
    return x

def create_cooc_matrix(df, col, folder):
    """
    :param df: dataframe
    :param col: column name
    :param folder: folder to which the co-occurrence matrix to be written
    """

    # Extract the column as a list of lists
    arr = list(df[col])

    # Join items for every user using space as separator
    arr = [" ".join(i) for i in arr]

    # Vectorize the data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(arr)

    # Calculate co-occurrence
    Xc = X.T * X

    # Set all diagonal elements to zero
    Xc.setdiag(0)

    # Convert sparse matrix to dense
    Xc = Xc.todense()

    # Convert dense matrix to dataframe
    Xd = pd.DataFrame(Xc)

    # Give column names and index
    Xd.columns = vectorizer.get_feature_names()
    Xd.index = vectorizer.get_feature_names()

    # melt the dataframe
    Xd = melt_df(Xd)

    # Save the file to disc
    Xd.to_csv(os.path.join(folder, f"{col}_cooc.csv"), index=False)


def melt_df(df):

    # create an upper-triangular matrix out of the co-occurrence matrix to avoid duplicates
    df = df.where(np.triu(np.ones(df.shape)).astype(np.bool))

    # melt the dataframe
    df = df.stack().reset_index()
    df.columns = ['item1', 'item2', 'count']

    # Remove zero count entries
    df = df[df["count"] != 0]

    return df

def orthonorm_basis(emb, items):
    if len(items) ==0:
        return []
    mat = np.array(emb.loc[emb["item"].isin(items),:].iloc[:,1:]).T
    return list(scipy.linalg.orth(mat))
