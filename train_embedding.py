import os
import yaml
import torch
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm

from argparse import ArgumentParser

from modules.embedding import gen_batches, embed_loss, cuda, orthonorm_basis
from modules.embedding import create_cooc_matrix, melt_df, CoOccurrenceOptimization

warnings.filterwarnings("ignore")

def train_model(data: CoOccurrenceOptimization, epochs, bs=2048):
    """
    Function to train the model
    """
    # using adam optimizer
    optimizer = torch.optim.Adam(data.all_params, weight_decay=1e-8)
    optimizer.zero_grad()

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        pbar.set_description(f"Epoch: {epoch}")
        loss_ = []
        for batch in gen_batches(data, bs):
            optimizer.zero_grad()
            loss = embed_loss(*batch)
            loss_.append(loss.item())
            loss.backward()
            optimizer.step()
        pbar.set_postfix(loss=np.mean(loss_))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--co_occurrence", default=1, help="Calculate co-occurrence matrix")

    opt = parser.parse_args()
    with open ("config.yaml") as f:
        config = yaml.load(f)

    x_max = config["embeddings"]["x_max"]
    alpha = config["embeddings"]["alpha"]
    base_std = config["embeddings"]["base_std"]
    num_epochs = config["embeddings"]["num_epochs"]
    batch_size = config["embeddings"]["batch_size"]
    n_embedding = config["embeddings"]["n_embedding"]
    list_cols = config["dataset_params"]["list_cols"]
    data_root_dir = config["dataset_params"]["root_dir"]
    data_file = config["dataset_params"]["output_file_name"]

    file_names = [os.path.join(data_root_dir, f"{i}_cooc.csv") for i in list_cols]
    input_file = os.path.join(data_root_dir, data_file)

    print("Reading the data...")
    df = pd.read_json(input_file)

    if int(opt.co_occurrence):
        print("Calculating co-occurrence matrices...")
        for col in list_cols:
            print(f"Processing column {col}...")
            create_cooc_matrix(df, col, data_root_dir)

    for i,j in enumerate(list_cols):
        print(f"Learning embeddings for column {j}...")
        coOccurrence_data = CoOccurrenceOptimization(base_std, x_max, alpha, n_embedding, file_names[i])
        train_model(coOccurrence_data, num_epochs[i], batch_size)
        coOccurrence_data.save_embedding(os.path.join(data_root_dir, f"{j}_emb.csv"))

    for j in list_cols:
        print(f"Calculating OrthoNormal Vectors for users in column {j}...")
        hash_col = f"{j}_hash"
        out_file = f"{j}_orth.json"
        emb_file = os.path.join(data_root_dir, f"{j}_emb.csv")
        emb = pd.read_csv(emb_file)
        df1 = df[[j, hash_col]].drop_duplicates([hash_col])
        df1.drop_duplicates([hash_col])
        df1["orth"] = df1[j].apply(lambda x: orthonorm_basis(emb, x))
        df1.drop(j, axis=1, inplace=True)
        df1.to_json(os.path.join(data_root_dir, out_file))
