import numpy as np
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

from Alignstein.align import calc_two_ch_sets_dists, align_chromatogram_sets


def gather_mids(chromatograms_sets_list):
    # it turns out that the best solution is to do it on python ordinary lists
    mzs = []
    rts = []
    chromatogram_indices: list[tuple[int, int]] = []
    for i, chromatogram_set in enumerate(chromatograms_sets_list):
        for j, chromatogram in enumerate(chromatogram_set):
            mzs.append(np.mean(chromatogram.mzs))
            rts.append(np.mean(chromatogram.rts))
            chromatogram_indices.append((i, j))
    return (np.array(list(zip(rts, mzs))).reshape((-1, 2)),
            np.array(chromatogram_indices))


def cluster_mids_subsets(mids, distance_threshold=20):
    return AgglomerativeClustering(n_clusters=None, affinity="l1",
                                   linkage='complete',
                                   distance_threshold=distance_threshold,
                                   ).fit_predict(mids)


# def create_chrom_sums(chromatograms_sets_list, clusters, chromatogram_indices):
#     idx_sort = np.argsort(clusters)
#     vals, idx_start, count = np.unique(clusters[idx_sort],
#                                        return_counts=True, return_index=True)
#     chromatogram_indices_by_clusters = np.split(idx_sort, idx_start[1:])
#     print("Average cluster size:",
#           np.mean(list(map(len, chromatogram_indices_by_clusters))))
#     result_ch_set = []
#     for i, one_cluster_indices in enumerate(chromatogram_indices_by_clusters):
#         cluster_chroms = []
#         for ch_set_id, ch_id in chromatogram_indices[one_cluster_indices]:
#             cluster_chroms.append(chromatograms_sets_list[ch_set_id][ch_id])
#         new_chromatogram = Chromatogram.sum_chromatograms(cluster_chroms)
#         #         new_chromatogram.normalize()
#         #         l = len(new_chromatogram)
# 
#         new_chromatogram.cut_smallest_peaks(0.005)
#         #         if (l > 7000):
#         #             print(l, "->", len(new_chromatogram))
#         result_ch_set.append(new_chromatogram)
#     return result_ch_set


def find_consensus_features(clustered_chromatogram_set, chromatograms_sets_list,
                            sinkhorn_upper_bound=40, flow_trash_penalty=5,
                            turns=1):
    all_consensus_features = []
    consensus_features = [[[] for _ in range(len(clustered_chromatogram_set))]
                          for _ in range(turns)]

    for i, ch_set in enumerate(chromatograms_sets_list):
        c_dists = calc_two_ch_sets_dists(ch_set, clustered_chromatogram_set,
                                         sinkhorn_upper_bound=sinkhorn_upper_bound)

        for turn in range(turns):
            matchings, matched_left, matched_right = align_chromatogram_sets(
                c_dists, flow_trash_penalty=flow_trash_penalty)
            for chromatogram_j, feature_ind in matchings:
                consensus_features[turn][feature_ind].append(
                    (i, chromatogram_j))
            c_dists[list(
                matched_left)] = np.inf  # simply stupid way to omit already used chromatograms

    for one_turn_cfeatures in consensus_features:
        for c_feature in one_turn_cfeatures:
            if len(c_feature) > 1:
                all_consensus_features.append(c_feature)
    return all_consensus_features


def cluster_mids(mids, distance_threshold=20):
    return MiniBatchKMeans(n_clusters=16, init='k-means++', max_iter=100,
                           batch_size=100, verbose=0, compute_labels=True,
                           random_state=None, tol=0.0, max_no_improvement=10,
                           init_size=None, n_init=3,
                           reassignment_ratio=0.01).fit_predict(mids)


def big_clusters_to_clusters(mids, big_clusters, distance_threshold=5):
    clusters = -1 * np.ones(len(mids))
    for i in range(np.max(big_clusters) + 1):
        inds = np.where(big_clusters == i)
        mids_subset = mids[inds]
        clusters_subsets = cluster_mids_subsets(mids_subset,
                                                distance_threshold=distance_threshold)
        clusters[inds] = clusters_subsets + max(clusters) + 1
    return clusters
