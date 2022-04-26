"""Parse mzml files and perform alignment

Usage: align.py -h
       align.py MZML_FILE MZML_FILE...

Arguments:
    MZML_FILE       names of files with chromatograms to be aligned

"""

import numpy as np
from MassSinkhornmetry import distance_dense
from scipy.spatial.distance import cdist
from tqdm import tqdm

from .chromatogram import Chromatogram
from .mfmc import match_chromatograms


# def gather_ch_mzs_rts(features):
#     rts = []
#     mzs = []
#     features_nb = []
#     for f_i, f in enumerate(features):
#         for ch in f.getConvexHulls():
#             for rt, mz in ch.getHullPoints():
#                 rts.append(rt)
#                 mzs.append(mz)
#                 features_nb.append(f_i)
#     #     rts.append(f.getRT())
#     #     mzs.append(f.getMZ())
#     return rts, mzs, features_nb


# def gather_ch_stats_over_exps(dirname):
#     fnames = []
#     means = []
#     variances = []
#     for i, filename in enumerate(os.listdir(dirname)):
#         if filename.endswith(".mzXML") and i % 5 == 0:
#             print("Now working on:", filename)
#             input_map = parse_ms1_xml(os.path.join(dirname, filename))
#             gc.collect()
#             openms_featues = find_features(input_map)
#             lengths, widths = gather_widths_lengths(openms_featues)
#             means.append((np.mean(widths), np.mean(lengths),
#                          np.mean(lengths) / np.mean(widths)))
#             variances.append((np.std(widths), np.std(lengths),
#                              np.std(lengths) / np.std(widths)))
#     return means, variances


def feature_dist(f1, f2, penalty=40, eps=0.1):
    """
    Compute GWD between two features.

    Parameters
    ----------
    f1 : Chromatogram
        One feature
    f2 : Chromatogram
        Other feature
    penalty : float
        Penalty for not transporting a part of signal, aka the lambda parameter.
        Can cen interpreted as maximal distance over which signal is
        transported.
    eps : float
        Entropic penalization coefficient, aka the epsilon parameter. Default
        value is chosen reasonably. Change it only if you understand how it
        works.
    Returns
    -------
    float
        A GWD between f1 and f2.
    """
    dists = cdist(np.column_stack((f1.rts, f1.mzs)),
                  np.column_stack((f2.rts, f2.mzs)), 'cityblock')
    return distance_dense(f1.ints, f2.ints, dists=dists, eps=eps, lam=penalty,
                          method="TV")


def mid_dist(ch1, ch2):
    return np.sum(np.abs(ch1.mid - ch2.mid))


def mid_mz_dist(ch1, ch2):
    return abs(ch1.mid[1] - ch2.mid[1])


def gwd_distance_matrix(chromatograms1: list[Chromatogram],
                        chromatograms2: list[Chromatogram],
                        centroid_upper_bound=10,
                        gwd_upper_bound=40,
                        mz_mid_upper_bound=float("inf"), eps=0.1):
    """
    Compute GWD distance matrix between two feature sets.

    Parameters
    ----------
    chromatograms1 : list of Chromatogram
        First list of features.
    chromatograms2 : list of Chromatogram
        First list of features.
    centroid_upper_bound : float
        Maximum cetroid distance between which GWD. For efficiency reasons
        should be reasonably small. If centroid distance is larger than
        distance_upper_bound, infinity is computed.
    mz_mid_upper_bound :
        Additional parameter if GDW should computed only for features with
        centroid M/Z difference lower than this parameter. Usually not used.
    gwd_upper_bound : float
        Penalty for not transporting a part of signal, aka the lambda parameter.
        Can cen interpreted as maximal distance over which signal is
        transported.
    eps : float
        GWD entropic penalization coefficient, aka the epsilon parameter.
        Default value is chosen reasonably. Change it only if you understand how
        it works.
    Returns
    -------
    numpy.array
        GDW distance (or infinity if centroids to distant) matrix between two
        feature sets.
    """
    total = len(chromatograms1) * len(chromatograms2)
    dists = np.full((len(chromatograms1), len(chromatograms2)), np.inf)

    tick = 0
    pbar = tqdm(total=total)
    print("Calculating distances")
    # TODO Think is parallelization is possible
    # TODO Make dists sparse
    for i, chi in enumerate(chromatograms1):
        for j, chj in enumerate(chromatograms2):
            if not chi.empty and not chj.empty:
                if (mid_dist(chi, chj) <= centroid_upper_bound and
                        mid_mz_dist(chi, chj) < mz_mid_upper_bound):
                    dists[i, j] = feature_dist(
                        chi, chj, penalty=gwd_upper_bound, eps=eps)
                tick += 1
                if tick % 300 == 0:
                    pbar.update(300)
    pbar.close()
    print("Calculated dists, number of nans:", np.sum(np.isnan(dists)))
    print("All columns have any row non zero:",
          np.all(np.any(dists < np.inf, axis=0)))
    print("All rows have any column non zero:",
          np.all(np.any(dists < np.inf, axis=1)))
    return dists


# def align_chromatogram_sets(ch_dists, matching_penalty=40):
#     return match_chromatograms(ch_dists, matching_penalty)


# def find_consensus_features(clustered_chromatogram_set, features_per_samples,
#                             gwd_upper_bound=40, matching_penalty=5):
#     consensus_features = [[] for _ in range(len(clustered_chromatogram_set))]
#     all_matched_left = []
#
#     for chrom_set_i, chrom_set in enumerate(features_per_samples):
#         #         print(i)
#         chromatogram_dists = calc_two_ch_sets_dists(
#             chrom_set, clustered_chromatogram_set,
#             gwd_upper_bound=gwd_upper_bound)
#         matchings, matched_left, matched_right = match_chromatograms(
#             chromatogram_dists, matching_penalty)
#         for chromatogram_j, feature_ind in matchings:
#             consensus_features[feature_ind].append(
#                 (chrom_set_i, chromatogram_j))
#
#         all_matched_left.append(matched_left)
#
#     return consensus_features, all_matched_left


def dump_consensus_features(consensus_features, filename,
                            chromatograms_sets_list):
    rows = []
    with open(filename, "weight") as outfile:
        for consensus_feature in consensus_features:
            row = []
            for set_i, chromatogram_j in consensus_feature:
                f_id = chromatograms_sets_list[set_i][chromatogram_j].feature_id
                row.extend(f_id)
            rows.append(" ".join(map(str, row)))
        outfile.write("\n".join(rows))


def find_pairwise_consensus_features(feature_set1, feature_set2,
                                     centroid_upper_bound=10,
                                     gwd_upper_bound=10,
                                     matching_penalty=5,
                                     mz_mid_upper_bound=2,
                                     eps=0.1):
    """
    Find consensus features between two feature sets.

    Parameters
    ----------
    feature_set1 : iterable of Chromatogram
        List with one chromatogram features
    feature_set2 : iterable of Chromatogram
        List with other chromatogram features
    centroid_upper_bound : float
        Maximum cetroid distance between which GWD will computed. For efficiency
        reasons should be reasonably small.
    gwd_upper_bound : float
        Penalty for not transporting a part of signal, aka the lambda parameter.
        Can be interpreted as maximal distance over which signal is
        transported while computing GWD.
    matching_penalty : penalty for feature not matching
    mz_mid_upper_bound : float
        Additional parameter if GWD should computed only for features with
        centroid M/Z difference lower than this parameter.
    eps : float
        GWD entropic penalization coefficient, aka the epsilon parameter.
        Default value is chosen reasonably. Change it only if you understand how
        it works.

    Returns
    -------
    Consensus features
    """
    consensus_features = []

    dists = gwd_distance_matrix(feature_set1, feature_set2,
                                centroid_upper_bound=centroid_upper_bound,
                                gwd_upper_bound=gwd_upper_bound,
                                mz_mid_upper_bound=mz_mid_upper_bound,
                                eps=eps)

    matchings, matched_left, matched_right = match_chromatograms(
        dists, penalty=matching_penalty)

    for left_f_ind, right_f_ind in matchings:
        consensus_features.append([(0, left_f_ind), (1, right_f_ind)])

    return consensus_features, [matched_left]
