# Multigraph Approach Towards a Scalable, Robust Look-alike Audience Extension System

This repository contains the source code for the paper Multigraph Approach Towards a Scalable, Robust Look-alike Audience Extension System.

### Installation

To install the dependencies run:
```
pip install -r requirements.txt
```

### Adform Dataset
We have used the [Adform Click Prediction Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TADBY7) to benchmark our model performance. 
To train the model on Adform data, download and unzip the data and place it inside the folder ```data/adform``` folder.
The data contains a set of 10 features. The features are hashed into 32-bit integers to preserve privacy.
Some of the features can have multiple values. Apart from these features, there is also a 
binary column indicating whether the ad was clicked by the user or not.

### Data Processing

To process the data run:
```python data_processing.py```

The data contains 5 large json files and all those files have to be combined before running the script. 
The data processing script does the following:
1) Removes certain columns with very high cardinality and are not useful for modeling.
2) Removes low-frequency categories from the dataset.
3) Converts hashed values to strings.
4) For columns that can have multiple values, removes rows where the number of items exceeds a defined threshold.
5) Calculates feature frequencies for scoring.
6) Saves the processed data to disc.

To learn embeddings for columns with multiple values run:
```
python train_embedding.py
```

### Building the graphs

To build the graphs run:
```
python build_graph.py
```

### Extending a Seed Set
To extend a seed set run:
```
python score_seed.py --seed_set path_to_seed_data.csv
```

The seed set data should have one column with the name ```id``` and that column should have all the ids in the seed set.


### Recall experiments on Adform data

The steps to reproduce the recall experiments on Adform data is in the notebook ```demo.ipynb```. 
Note that the model used in the demo was trained on one file ```adform.click.2017.05``` of the  Adform data.
It took roughly 6 hours for building the graph using a machine with Tesla RTX and 64GB memory.

