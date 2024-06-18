import json

import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer

from data_reading import read_df


def cluster(config):
    # Apply UMAP for dimensionality reduction
    reducer = umap.UMAP(n_neighbors=config['umap']['n_neighbors'],
                        n_components=config['umap']['n_components'],
                        metric=config['umap']['metric'])
    if config['feature_types'] == 'embedding':
        df, embedding_feature_names = read_df(config, True)
        umap_data = reducer.fit_transform(df[embedding_feature_names])
    else:
        df = read_df(config, False)
        idf_vect = TfidfVectorizer(ngram_range=tuple(config['tf_idf_params']['ngram_range']))
        umap_data = reducer.fit_transform(idf_vect.fit_transform(df['clean_abstract']))

    # Initialize DBSCAN
    dbscan = hdbscan.HDBSCAN(
        min_cluster_size=config['clustering']['min_cluster_size'],
        metric=config['clustering']['metric'],
        cluster_selection_method=config['clustering']['cluster_selection_method'])
    dbscan.fit(umap_data)

    # Obtain cluster labels
    labels = dbscan.labels_
    # Add cluster labels to the DataFrame

    df = read_df(config, False)
    df['Cluster'] = labels
    # Save DataFrame to a CSV file
    feature_types = config['feature_types']
    path = config['paths'][f'data_with_clusters_{feature_types}_path']
    df.to_csv(path, index=False)
    print(f'{path} was saved')

    # Count the number of unique clusters
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f'number of clusters: {num_clusters}')
