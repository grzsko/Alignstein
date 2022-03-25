"""Parse mzml files and perform alignment

Usage: align.py -h
       align.py MZML_FILE MZML_FILE...

Arguments:
    MZML_FILE       names of files with chromatograms to be aligned

"""
import xml.etree.ElementTree as ET

import numpy as np
from Alignstein.parse import (openms_feature_to_feature,
                              parse_chromatogram_file, features_to_weight)
from MassSinkhornmetry import distance_dense
from scipy.spatial.distance import cdist
from tqdm import tqdm
from typing import List

from .OpenMSMimicry import MyCollection, OpenMSFeatureMimicry
from .chromatogram import Chromatogram
from .mfmc import match_chromatograms


# TODO read about better practices in importing

def feature_from_file_features_to_chromatograms(input_map, openms_features, w):
    chromatograms = []
    for oms_f in openms_features:
        ch = openms_feature_to_feature(oms_f, input_map, w)
        if not ch.empty:
            ch.feature_id = [oms_f.intesity, oms_f.rt, oms_f.mz]
            ch.cut_smallest_peaks(0.001)
            chromatograms.append(ch)
    return chromatograms


def parse_feature_with_model_xml(filename):
    features = MyCollection()
    tree = ET.parse(filename)
    root = tree.getroot()
    feature_list = root.find("./featureList")
    for i, feature in enumerate(feature_list.findall("./openms_feature")):
        convex_hull = feature.find("./convexhull")
        points = []
        for hullpoint in convex_hull.findall("./hullpoint"):
            positions = {int(position.attrib["dim"]): float(position.text)
                         for position in hullpoint.findall("./hposition")}
            points.append((positions[0], positions[1]))
        if len(points) > 2:
            mimicry_feature = OpenMSFeatureMimicry(points)
            # mimicry_feature.caap_id = i
            features.append(mimicry_feature)
        else:
            print("Skipping small openms_feature, id:", feature.attrib["id"])
            continue
    return features


def feature_from_file_experiment_chromatogram_set(filename, features_filename):
    input_map = parse_chromatogram_file(filename)
    features = parse_feature_with_model_xml(features_filename)
    weight = features_to_weight(features)
    print("Parsed file", filename, "\n", features.size(),
          "openms_featues found,\nAverage lenght to width:", weight)
    return feature_from_file_features_to_chromatograms(input_map, features,
                                                       weight)


def parse_feature_from_file_alignment_experiment_chromatogram_sets(
        chromatogram_filenames, features_filenames):
    ch_sets_list = []
    for ch_fname, f_fname in zip(chromatogram_filenames, features_filenames):
        ch_sets_list.append(
            feature_from_file_experiment_chromatogram_set(ch_fname, f_fname))

    return ch_sets_list


def gather_ch_mzs_rts(features):
    rts = []
    mzs = []
    features_nb = []
    for f_i, f in enumerate(features):
        for ch in f.getConvexHulls():
            for rt, mz in ch.getHullPoints():
                rts.append(rt)
                mzs.append(mz)
                features_nb.append(f_i)
    #     rts.append(f.getRT())
    #     mzs.append(f.getMZ())
    return rts, mzs, features_nb


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


def chromatogram_dist(ch1, ch2, penalty=40):
    dists = cdist(np.column_stack((ch1.rts, ch1.mzs)),
                  np.column_stack((ch2.rts, ch2.mzs)), 'cityblock')
    return distance_dense(ch1.ints, ch2.ints, dists=dists, eps=1, lam=penalty,
                          method="TV")


def mid_dist(ch1, ch2):
    return np.sum(np.abs(ch1.mid - ch2.mid))


def mid_mz_dist(ch1, ch2):
    return abs(ch1.mid[1] - ch2.mid[1])


def calc_two_ch_sets_dists(chromatograms1: List[Chromatogram],
                           chromatograms2: List[Chromatogram],
                           sinkhorn_upper_bound=40,
                           mz_mid_upper_bound=float("inf")):
    # dist to empty chromatogram is inf
    total = len(chromatograms1) * len(chromatograms2)

    ch_dists = np.full((len(chromatograms1), len(chromatograms2)), np.inf)

    tick = 0
    pbar = tqdm(total=total)
    print("Calculating distances")
    for i, chi in enumerate(chromatograms1):
        for j, chj in enumerate(chromatograms2):
            if not chi.empty and not chj.empty:
                if (mid_dist(chi, chj) <= sinkhorn_upper_bound and
                        mid_mz_dist(chi, chj) < mz_mid_upper_bound):
                    ch_dists[i, j] = chromatogram_dist(
                        chi, chj, sinkhorn_upper_bound)
                tick += 1
                if tick % 300 == 0:
                    pbar.update(300)
    pbar.close()
    print("Calculated dists, number of nans:", np.sum(np.isnan(ch_dists)))
    print("All columns have any row non zero:",
          np.all(np.any(ch_dists < np.inf, axis=0)))
    print("All rows have any column non zero:",
          np.all(np.any(ch_dists < np.inf, axis=1)))
    return ch_dists


# def align_chromatogram_sets(ch_dists, flow_trash_penalty=40):
#     return match_chromatograms(ch_dists, flow_trash_penalty)


# def find_consensus_features(clustered_chromatogram_set, features_per_samples,
#                             sinkhorn_upper_bound=40, flow_trash_penalty=5):
#     consensus_features = [[] for _ in range(len(clustered_chromatogram_set))]
#     all_matched_left = []
#
#     for chrom_set_i, chrom_set in enumerate(features_per_samples):
#         #         print(i)
#         chromatogram_dists = calc_two_ch_sets_dists(
#             chrom_set, clustered_chromatogram_set,
#             sinkhorn_upper_bound=sinkhorn_upper_bound)
#         matchings, matched_left, matched_right = match_chromatograms(
#             chromatogram_dists, flow_trash_penalty)
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


def find_pairwise_consensus_features(chromatogram_set1, chromatogram_set2,
                                     sinkhorn_upper_bound=40,
                                     flow_trash_penalty=5):
    consensus_features = []

    c_dists = calc_two_ch_sets_dists(chromatogram_set1, chromatogram_set2,
                                     sinkhorn_upper_bound=sinkhorn_upper_bound,
                                     mz_mid_upper_bound=2)
    matchings, matched_left, matched_right = match_chromatograms(
        c_dists, penalty=flow_trash_penalty)

    for left_f_ind, right_f_ind in matchings:
        consensus_features.append([(0, left_f_ind), (1, right_f_ind)])

    return consensus_features, [matched_left]
