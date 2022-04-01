import numpy as np
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

from .align import calc_two_ch_sets_dists  # , align_chromatogram_sets
from .mfmc import match_chromatograms_gathered_by_clusters


def gather_mids(feature_sets_list):
    """
    Gather M/Zs and RTs centroids from all chromatograms and chromatogram sets.

    Parameters
    ----------
    feature_sets_list : iterable of iterable of Chromatogram
        openms_featues gathered by input chromatograms
    """

    mzs = []
    rts = []
    # it turns out that the best solution is to do it on python ordinary lists
    chromatogram_indices: list[tuple[int, int]] = []
    for i, chromatogram_set in enumerate(feature_sets_list):
        for j, chromatogram in enumerate(chromatogram_set):
            mzs.append(np.mean(chromatogram.mzs))
            rts.append(np.mean(chromatogram.rts))
            chromatogram_indices.append((i, j))
    return np.array(list(zip(rts, mzs))).reshape((-1, 2)) #,
           # np.array(chromatogram_indices))


def cluster_mids_subsets(mids, distance_threshold=20):
    mids_exp = np.copy(mids)
    RT_COL = 0
    mids_exp[:, RT_COL] = np.log(mids_exp[:, RT_COL])
    return AgglomerativeClustering(n_clusters=None, affinity="l1",
                                   linkage='average',
                                   # linkage='complete',
                                   distance_threshold=distance_threshold,
                                   ).fit_predict(mids_exp)


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


def find_consensus_features(clusters,
                            features_per_samples,
                            sinkhorn_upper_bound=40, flow_trash_penalty=5,
                            turns=1):
    consensus_features = [[[] for _ in range(len(np.unique(clusters)))]
                          for _ in range(turns)]
    clusters = clusters.astype(int)
    for sample_i, one_sample_features in enumerate(features_per_samples):
        rest_of_features, clusters_filtered = flatten_chromatograms(
            features_per_samples, clusters,  # chromatogram_indices,
            exclude_chromatogram_sets=[sample_i])
        c_dists = calc_two_ch_sets_dists(one_sample_features, rest_of_features,
                                         sinkhorn_upper_bound=sinkhorn_upper_bound)

        for turn in range(turns):
            matchings, matched_left, matched_right = \
                match_chromatograms_gathered_by_clusters(
                    c_dists, clusters_filtered, flow_trash_penalty)
            if len(matchings) == 0:
                print("Breaking at turn ", turn)
                break  # there is nothing more to be matched in next turns
            for feature_j, cluster in matchings:
                consensus_features[turn][cluster].append((sample_i, feature_j))
            c_dists[list(matched_left)] = np.inf  # simply stupid way to omit
            # already used chromatograms (indeed, maybe confusing, but previous
            # line changes whole rows)

    succeeded_consensus_features = []
    for one_turn_c_features in consensus_features:
        for c_feature in one_turn_c_features:
            if len(c_feature) > 1:
                succeeded_consensus_features.append(c_feature)
    return succeeded_consensus_features


def precluster_mids(mids, distance_threshold=20):
    return np.array(
        MiniBatchKMeans(n_clusters=16, init='k-means++', max_iter=100,
                        batch_size=100, verbose=0, compute_labels=True,
                        random_state=None, tol=0.0, max_no_improvement=10,
                        init_size=None, n_init=3,
                        reassignment_ratio=0.01).fit_predict(mids))


def big_clusters_to_clusters(mids, big_clusters, distance_threshold=5):
    clusters = -1 * np.ones(len(mids))
    for i in range(np.max(big_clusters) + 1):
        inds = np.where(big_clusters == i)
        mids_subset = mids[inds]
        clusters_subsets = cluster_mids_subsets(mids_subset,
                                                distance_threshold=distance_threshold)
        clusters[inds] = clusters_subsets + np.max(clusters) + 1
    return clusters
