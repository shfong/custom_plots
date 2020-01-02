"""Contains miscellaneous utility functions"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

def enumerate_clusters(Z, starting_threshold=-1, step=0.1, criterion='distance', min_size=0, max_size=np.inf):
    """Sweep through linkage matrix and enumerate all clusters
    
    The sweep will end when there is only a single cluster containing all nodes

    Parameters
    ----------
    Z : np.array
        Scipy output from hierarchy.linkage
    starting_threshold : float
        Starting value for hierarchy.fcluster to cut the dendrogram
    step : float
        Step size for the sweep. Smaller value reduces the chance of missing
        clusters, but it will take longer
    min_size : int
        Minimum size of the cluster to return
    max_size : int 
        maximum size of the cluster to return

    Returns
    -------
    number_of_clusters : list of list of ints
        List of list where the inner list is a list of indices of the nodes that
        form a cluster
    """

    number_of_clusters = []
    current_n = -1
    threshold = starting_threshold
    
    while True: 
        threshold += step
        out = fcluster(Z, threshold, criterion=criterion)

        n_unique_clusters = len(np.unique(out))
        if n_unique_clusters == current_n: 
            continue

        clusters = define_cluster(out, min_size=min_size, max_size=max_size)
        
        if clusters: 
            number_of_clusters += clusters
            
        current_n = n_unique_clusters


        if current_n == 1: 
            break
            
    return number_of_clusters

def define_cluster(cluster_assignments, name_map=None, min_size=0, max_size=np.inf):
    """Assumes cluster_assignments 
    
    Given a list of indices, where each index corresponds to the cluster 
    assignment, this function enumerates each cluster's members

    Paramters
    ---------
    cluster_assignments : list of ints
        The cluster assignment for each element
    name_map : list or dictionary
        Maps position in `cluster_assignments` to name. Can be any object that 
        can be indexed using the square brackets
    min_size : int
        Minimum size of the cluster to include
    max_size : int
        Maximum size of the cluster to include

    Returns
    -------
    kept_clusters : list of list of ints
        List of list where the inner list is a list of indices of the nodes that
        form a cluster
    """
    if name_map is None: 
        name_map = list(range(len(cluster_assignments)))

    n_clusters = len(np.unique(cluster_assignments))
    clusters = [[] for _ in range(n_clusters)]
    
    for ind, i in enumerate(cluster_assignments):
        clusters[i-1].append(name_map[ind])
        
    kept_clusters = []
    for i in clusters: 
        if len(i) >= min_size and len(i) <= max_size:
            kept_clusters.append(i)
            
    return kept_clusters


def melt_upper_triu(df): 
    keep = np.triu(np.ones(df.shape))
    keep = keep.astype('bool').reshape(df.size)
    return df.stack(dropna=False)[keep]


def shuffle_along_axis(df, shuffle_names, axis=1): 
    """Shuffle a subset of a dataframe along an axis"""

    shuffled = df.copy()
    for name in shuffle_names:
        if axis == 0: 
            x = shuffled.loc[name].values.ravel()
            np.random.shuffle(x)
            shuffled.loc[name] = x

        else: 
            x = shuffled[name].values.ravel()
            np.random.shuffle(x)
            shuffled.loc[:, name] = x

    return shuffled


def flatten_3d_array_to_dataframe(array, index, columns, names): 
    """Flatten a 3 dimensional array to a flattend datafarme
    
    Parameters
    ----------
    array : numpy array
        3 dimensional numpy array
    index : iterable
        row names of the 2-D array after it is sliced from the 3d
    columns : iterable
        column names of the 2-D array after it is sliced
    names : iterable
        column names of the flattened array

    Returns
    -------
    dfs : pandas DataFrame
        Flatten dataframe
    """

    dfs = []
    for i in range(array.shape[-1]): 
        mat = array[:, :, i]
        tmp = pd.DataFrame(mat, index=index, columns=columns)
        tmp = tmp.unstack().reset_index()
        
        tmp.columns = names
        
        dfs.append(tmp)
        
    return pd.concat(dfs, axis=0)


def flat_adjacency_correlations(data, index, columns, values, corr_name=None, axis=0):
    """Calculates correlation for a flattened dataframe
    
    Note: axis matches the number of unique elements in the respective axis. 0
    refers to the index, 1 refers to the columns
    """

    if axis == 0: 
        index, columns = columns, index
    elif axis == 1: 
        index, columns = index, columns
    else: 
        raise ValueError(f"Expected axis to be 0 or 1. Received {axis} instead")

    mat = data.pivot(index, columns, values)

    corr = mat.corr()
    corr.index.name = f"{columns}-1"
    corr.columns.name = f"{columns}-2"

    corr = melt_upper_triu(corr).reset_index()
    column_names = list(corr.columns)
    column_names[-1] = corr_name if corr_name is not None else "Corr"
    corr.columns = column_names

    return corr

def subset_dataframe(dataframe, columns, isin):
    """Subset dataframe based on whether all columns contain isin"""

    index = dataframe[columns].isin(isin).all(axis=1)

    return dataframe.loc[index]


def expand_string_arguments(arg, length):
    if isinstance(arg, str):
        return [arg]*length

    return arg


def normalize_counts(counts): 
    total_counts = counts.sum(axis=0)

    return counts/total_counts


def get_square_interactions(df, index='geneA', columns='geneB', values='pi_mean'):
    all_index = sorted(list(set(df[[index, columns]].values.ravel())))
    df_sq = df.pivot(index=index, columns=columns, values=values)
    df_sq = df_sq.reindex(index=all_index, columns=all_index)

    mask = pd.notnull(df_sq).values.astype(int)
    mask += mask.T
    
    df_sq = df_sq.fillna(0)
    df_sq += df_sq.T
    df_sq.values[mask > 1] = df_sq.values[mask > 1]/2 #average the constructs where it's in both locations

    return df_sq

def get_common_index(df): 
    return sorted(list(set(list(np.hstack([df.index, df.columns])))))


def expand_collections(cols1, cols2):
    """Expand a collection by duplicating an object if needed

    Expect inputs to be either strings or iterables that contain __len__ method.
    If both inputs are not strings, check the inputs have the same lengths and
    return. If only one input is string, then force that input to have the 
    same length as the other by duplicating the string in a list. If both
    inputs are strings, return both strings. 
    """

    if isinstance(cols1, str) and isinstance(cols2, str): 
        return cols1, cols2

    if isinstance(cols1, str) and not isinstance(cols2, str):
        cols1 = [cols1]*len(cols2)
    
    elif isinstance(cols2, str) and not isinstance(cols1, str): 
        cols2 = [cols2]*len(cols1)

    if len(cols1) != len(cols2):
        raise ValueError("cols1 and col2 do not have the same columns!")

    return cols1, cols2

def combine_pi(df1, df2, geneA=None, geneB=None, scores=None):
    """Combines two dataframe side by side
    
    If no indices are provided, the first two columns will be used as the index,
    and the last column will be used to combine.
    """

    if all([g is None for g in [geneA, geneB, scores]]):
        indices1 = list(df1.columns[:2])
        indices2 = list(df2.columns[:2])
        col = df2.columns[2]

    else: 
        indices1 = [geneA, geneB]
        indices2 = [geneA, geneB]
        col = scores

    return pd.concat([
        df1.set_index(indices1)[col], 
        df2.set_index(indices2)[col]
    ], axis=1)

def subset_correlation(df1, df2, x, return_size=False, geneA='geneA', geneB='geneB', scores='pi_mean'):
    """combine the two dataframes and slice to calculate correlations on the 
    extremes"""

    reps = combine_pi(df1, df2, geneA=geneA, geneB=geneB, scores=scores)
    reps.columns = ['Rep1', 'Rep2']
    
    zs = (reps - reps.mean()) / reps.std()
    
    y = []
    size = []
    for i in x: 
        subset = reps[(abs(zs) >= i).any(axis=1)]
        y.append(subset.corr().values[0,1])
        size.append(subset.shape[0])
        
    if return_size: 
        return x, y, size
    
    return x, y 

def set_n_columns_as_index(df, n):
    """Set the first n columns as index"""
    
    columns = list(df.columns)[:n]
    
    return df.set_index(columns)

def drop_controls_indices(df, drop_str): 
    columns = [i for i in df.columns if drop_str not in i]
    index = [i for i in df.index if drop_str not in i]
    
    return df.loc[index, columns]


def combine_counts_files(files, index_cols=5, **read_kwargs):
    dfs = [set_n_columns_as_index(pd.read_csv(f, **read_kwargs), index_cols) for f in files]
    df = pd.concat(dfs, axis=1)
    
    return df

def group_arg(iterable, key=None):
    if key is not None: 
        iterable = [key(i) for i in iterable]
        
    uniques = np.unique(iterable)
    iterable = np.array(iterable)
    
    return [np.argwhere(iterable == i).ravel() for i in uniques]