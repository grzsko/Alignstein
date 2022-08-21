from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from itertools import repeat
from typing import List, Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

from .chromatogram import Chromatogram
from .align import gwd_distance_matrix, gwd_distance_matrix_parallel
from .mfmc import match_chromatograms_gathered_by_clusters

_BIG_CLUSTER_COUNT = 16


def gather_mids(feature_sets_list):
    """
    Gather M/Zs and RTs centroids from all chromatograms and chromatogram sets.

    Parameters
    ----------
    feature_sets_list : iterable of iterable of Chromatogram
        List of every chromatogram features

    Returns
    -------
    numpy.array
        Array of all features centroids.
    """

    mzs = []
    rts = []
    # it turns out that the best solution is to do it on python ordinary lists
    feature_indices: list[tuple[int, int]] = []
    for i, feature_set in enumerate(feature_sets_list):
        for j, feature in enumerate(feature_set):
            mzs.append(np.mean(feature.mzs))
            rts.append(np.mean(feature.rts))
            feature_indices.append((i, j))
    return np.array(list(zip(rts, mzs))).reshape((-1, 2))


def flatten_chromatograms(chromatograms_sets_list, clusters,
                          exclude_chromatogram_sets=[]):
    _, count = np.unique(clusters,  # [idx_sort],
                         return_counts=True)
    print("Average cluster size:", np.mean(count))
    flat_chromatograms = []
    filtered_clusters = []

    first_chromatogram_index = 0
    # TODO rename every chromatogram set to openms_feature
    for i, one_sample_chromatogram_set in enumerate(chromatograms_sets_list):
        if i not in exclude_chromatogram_sets:
            for chromatogram in one_sample_chromatogram_set:
                flat_chromatograms.append(chromatogram)
            filtered_clusters.extend(
                clusters[first_chromatogram_index:
                         first_chromatogram_index + len(
                             one_sample_chromatogram_set)
                ])
        first_chromatogram_index += len(one_sample_chromatogram_set)
    assert first_chromatogram_index == len(clusters)
    return flat_chromatograms, filtered_clusters


def find_consensus_features(clusters, feature_sets_list,
                            centroid_upper_bound=10, gwd_upper_bound=10,
                            matching_penalty=5, turns=10,
                            mz_mid_upper_bound=float("inf"), eps=0.1):
    """
    Find consensus feature in preclustered dataset.

    Parameters
    ----------
    clusters
    feature_sets_list
    centroid_upper_bound : float
        Maximum cetroid distance between which GWD will computed. For efficiency
        reasons should be reasonably small.
    gwd_upper_bound : float
        Penalty for not transporting a part of signal, aka the lambda parameter.
        Can be interpreted as maximal distance over which signal is
        transported while computing GWD.
    matching_penalty : penalty for feature not matching
    turns : number of one feature set matching repeats
    mz_mid_upper_bound : float
        Additional parameter if GDW should computed only for features with
        centroid M/Z difference lower than this parameter. Usually not used.
    eps : float
        GWD entropic penalization coefficient, aka the epsilon parameter.
        Default value is chosen reasonably. Change it only if you understand how
        it works.

    Returns
    -------
    Complicated consensus features.
    """
    # TODO Refactor this function
    consensus_features = [[[] for _ in range(len(np.unique(clusters)))]
                          for _ in range(turns)]
    import time
    for sample_i, one_sample_features in enumerate(feature_sets_list):
        rest_of_features, clusters_filtered = flatten_chromatograms(
            feature_sets_list, clusters,
            exclude_chromatogram_sets=[sample_i])
        start = time.time()
        dists = gwd_distance_matrix_parallel(
            one_sample_features, rest_of_features,
            centroid_upper_bound=centroid_upper_bound,
            gwd_upper_bound=gwd_upper_bound,
            mz_mid_upper_bound=mz_mid_upper_bound,
            eps=eps)
        print("Dists calculated in:", time.time() - start)

        for turn in range(turns):
            start = time.time()
            matchings, matched_left, matched_right = \
                match_chromatograms_gathered_by_clusters(
                    dists, clusters_filtered, matching_penalty)
            print("Matching done in:", time.time() - start)
            if len(matchings) == 0:
                print("Breaking at turn ", turn)
                break  # there is nothing more to be matched in next turns
            for feature_j, cluster in matchings:
                consensus_features[turn][int(cluster)].append((sample_i, feature_j))
            dists[list(matched_left)] = np.inf  # simply stupid way to omit
            # already used chromatograms (indeed, maybe confusing, but previous
            # line changes whole rows)

    succeeded_consensus_features = []
    for one_turn_c_features in consensus_features:
        for c_feature in one_turn_c_features:
            if len(c_feature) > 1:
                succeeded_consensus_features.append(c_feature)
    return succeeded_consensus_features


def find_consensus_features_paralel(clusters, feature_sets_list,
                                    centroid_upper_bound=10, gwd_upper_bound=10,
                                    matching_penalty=5, turns=10,
                                    mz_mid_upper_bound=float("inf"), eps=0.1,
                                    big_clusters=None):
    """
    Find consensus feature in preclustered dataset paralelized version.

    This function in comparison with find_consensus_features additionally split
    input sets by big clusters and runs it paralely. It does not fall much in
    precision but reduces polynomial problems to smallers and paralelizes
    execution.

    Parameters
    ----------
    clusters
    feature_sets_list
    centroid_upper_bound : float
        Maximum cetroid distance between which GWD will computed. For efficiency
        reasons should be reasonably small.
    gwd_upper_bound : float
        Penalty for not transporting a part of signal, aka the lambda parameter.
        Can be interpreted as maximal distance over which signal is
        transported while computing GWD.
    matching_penalty : penalty for feature not matching
    turns : number of one feature set matching repeats
    mz_mid_upper_bound : float
        Additional parameter if GDW should be computed only for features with
        centroid M/Z difference lower than this parameter. Usually not used.
    eps : float
        GWD entropic penalization coefficient, aka the epsilon parameter.
        Default value is chosen reasonably. Change it only if you understand how
        it works.
    big_clusters : iterable of int
        Feature matching is done within these clusters.

    Returns
    -------
    Complicated consensus features.
    """
    # 1. Split inputs by big clusters
    big_clusters = big_clusters.astype(int)
    assert len(np.unique(big_clusters)) == np.max(big_clusters) + 1
    big_clusters_number = len(np.unique(big_clusters))
    feature_id_mapping: list[list[list[int]]] = [[[] for _2 in range(len(feature_sets_list))] for _ in range(big_clusters_number)]
    features_separated: list[list[list[Chromatogram]]] = [ # features separated by big clusters
        [
            [] for _2 in range(len(feature_sets_list))
        ] for _ in range(big_clusters_number)]
    i = 0
    for fset_id, fset in enumerate(feature_sets_list):
        for f_id, feature in enumerate(fset):
            features_separated[big_clusters[i]][fset_id].append(feature)
            feature_id_mapping[big_clusters[i]][fset_id].append(f_id)
            i += 1
    clusters_separated = clusters
    with ProcessPoolExecutor(max_workers=_BIG_CLUSTER_COUNT) as outer_pool:
        separated_cfeatures = list(
            outer_pool.map(find_consensus_features,
                           clusters_separated, features_separated,
                           repeat(centroid_upper_bound), repeat(gwd_upper_bound),
                           repeat(matching_penalty), repeat(turns),
                           repeat(mz_mid_upper_bound), repeat(eps)))

    # 3. Merge results
    consensus_features = []
    for big_cluster_id, big_cluster_cfeatures in enumerate(separated_cfeatures):
        for cfeature in big_cluster_cfeatures:
            mapped_cfeature = []
            for set_id, f_id in cfeature:
                mapped_cfeature.append((set_id, feature_id_mapping[big_cluster_id][set_id][f_id]))
            consensus_features.append(mapped_cfeature)
    return consensus_features

def precluster_mids(mids):
    return np.array(
        MiniBatchKMeans(n_clusters=_BIG_CLUSTER_COUNT, init='k-means++',
                        max_iter=100, batch_size=100, verbose=0,
                        compute_labels=True, random_state=None, tol=0.0,
                        max_no_improvement=10, init_size=None, n_init=3,
                        reassignment_ratio=0.01).fit_predict(mids))


def cluster_mids_subsets(mids, distance_threshold=10):
    mids_log = np.copy(mids)
    RT_COL = 0
    # For clustering we want to smash RT to obtain clusters 'flat' over RT axis,
    # the simplest solution is to take log form RT
    mids_log[:, RT_COL] = np.log(1 + mids_log[:, RT_COL])
    return AgglomerativeClustering(n_clusters=None, affinity="l1",
                                   linkage='average',
                                   distance_threshold=distance_threshold,
                                   ).fit_predict(mids_log)


def big_clusters_to_clusters(mids, big_clusters, distance_threshold=5,
                             clusters_flat=False):
    """
    Do clustering over rough preclusters.
    Parameters
    ----------
    mids
        Feature centroids
    big_clusters
        Indices of preclusters for every feature.
    distance_threshold
        Maximum distance between centroid in one cluster.
    Returns
    -------
    list of ints
        Clusters
    """
    # TODO make it better
    if clusters_flat:
        clusters = -1 * np.ones(len(mids))
    else:
        clusters = []
    for i in range(np.max(big_clusters) + 1):
        inds = np.where(big_clusters == i)
        mids_subset = mids[inds]
        clusters_subsets = cluster_mids_subsets(
            mids_subset, distance_threshold=distance_threshold)
        if clusters_flat:
            clusters[inds] = clusters_subsets + np.max(clusters) + 1
        else:
            clusters.append(clusters_subsets)
    return clusters


def cluster_mids(feature_sets_list, distance_threshold=5, clusters_flat=False):
    mids = gather_mids(feature_sets_list)
    big_clusters = precluster_mids(mids)
    clusters = big_clusters_to_clusters(mids, big_clusters,
                                        distance_threshold=distance_threshold,
                                        clusters_flat=clusters_flat)
    return mids, big_clusters, clusters
    # TODO Do it better with this clusters_flat
