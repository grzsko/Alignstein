from multiprocessing import Pool

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from MassSinkhornmetry import distance_dense

from .chromatogram import Chromatogram
from .mfmc import match_chromatograms


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


def monoisotopic_dist(ch1, ch2):
    return abs(np.min(ch1.mzs) - np.min(ch2.mzs))


def gwd_distance_matrix(chromatograms1: list[Chromatogram],
                        chromatograms2: list[Chromatogram],
                        centroid_upper_bound=10,
                        gwd_upper_bound=40,
                        mz_mid_upper_bound=float("inf"),
                        monoistopic_max_dist=float("inf"),
                        eps=0.1):
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
    monoistopic_max_dist : float
        Additional parameter if GDW should computed only for features with
        monoisotopic mass difference lower than this parameter. May stabilise
        the results
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
                        mid_mz_dist(chi, chj) < mz_mid_upper_bound and
                        monoisotopic_dist(chi, chj) < monoistopic_max_dist):
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


def pool_init(chromatograms1, params):
    global chromatograms
    chromatograms = chromatograms1
    global parameters
    parameters = params


def async_col_distances(chromatogram):
    column = np.full(len(chromatograms), np.inf)
    if not chromatogram.empty:
        for i, chi in enumerate(chromatograms):
            if not chi.empty:
                if (mid_dist(chi, chromatogram) <= parameters[
                        "centroid_upper_bound"] and
                    mid_mz_dist(chi, chromatogram) < parameters[
                        "mz_mid_upper_bound"] and
                    monoisotopic_dist(chi, chromatogram) < parameters[
                        "monoisotopic_max_dist"]
                ):
                    column[i] = feature_dist(
                        chi, chromatogram, penalty=parameters["gwd_upper_bound"],
                        eps=parameters["eps"])
    return column


def gwd_distance_matrix_parallel(
        chromatograms1: list[Chromatogram], chromatograms2: list[Chromatogram],
        centroid_upper_bound=10, gwd_upper_bound=40,
        mz_mid_upper_bound=float("inf"), monoistopic_max_dist=float("inf"),
        eps=0.1):
    """
    Compute GWD distance matrix between two feature sets, paralelized version.

    Parameters
    ----------
    chromatograms1 : list of Chromatogram
        First list of features.
    chromatograms2 : list of Chromatogram
        First list of features.
    centroid_upper_bound : float
        Maximum centroid distance between which GWD. For efficiency reasons
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
        GDW distance (or infinity if centroids are too distant) matrix between
        two feature sets.
    """
    # total = len(chromatograms1) * len(chromatograms2)
    # dists = np.full((len(chromatograms1), len(chromatograms2)), np.inf)

    with Pool(
            initializer=pool_init,
            initargs=(chromatograms1, {
                "centroid_upper_bound": centroid_upper_bound,
                "gwd_upper_bound": gwd_upper_bound,
                "mz_mid_upper_bound": mz_mid_upper_bound,
                "monoistopic_max_dist": monoistopic_max_dist,
                "eps": eps})) as pool:
        # TODO Make dists sparse
        columns = [pool.apply_async(async_col_distances, [ch]) \
                   for ch in chromatograms2]
        # There is probably more chromatograms2 than chromatograms1 so we have
        # more flexibility in spawning data over processes and less data is
        # copied to all processes
        dists = np.column_stack([column.get() for column in columns])
    print("Calculated dists, number of nans:", np.sum(np.isnan(dists)))
    print("All columns have any row non zero:",
          np.all(np.any(dists < np.inf, axis=0)))
    print("All rows have any column non zero:",
          np.all(np.any(dists < np.inf, axis=1)))
    return dists


def find_pairwise_consensus_features(feature_set1, feature_set2,
                                     centroid_upper_bound=10,
                                     gwd_upper_bound=10,
                                     matching_penalty=5,
                                     mz_mid_upper_bound=2,
                                     monoistopic_max_dist=float("inf"),
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

    dists = gwd_distance_matrix_parallel(
        feature_set1, feature_set2,
        centroid_upper_bound=centroid_upper_bound,
        gwd_upper_bound=gwd_upper_bound,
        mz_mid_upper_bound=mz_mid_upper_bound,
        monoistopic_max_dist=monoistopic_max_dist,
        eps=eps)

    matchings, matched_left, matched_right = match_chromatograms(
        dists, penalty=matching_penalty)

    for left_f_ind, right_f_ind in matchings:
        consensus_features.append([(0, left_f_ind), (1, right_f_ind)])

    # TODO remove this matched left
    return consensus_features, [matched_left]
