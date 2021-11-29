"""Parse mzml files and perform alignment

Usage: align.py -h
       align.py MZML_FILE MZML_FILE...

Arguments:
    MZML_FILE       names of files with chromatograms to be aligned

"""
import xml.etree.ElementTree as ET
import pyopenms
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import networkx as nx

from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.sparse import csr_matrix
from MassSinkhornmetry import distance_dense

from Alignstein.chromatogram import Chromatogram
from Alignstein.OpenMSMimicry import MyCollection, OpenMSFeatureMimicry


def parse_ms1_xml(filename):
    options = pyopenms.PeakFileOptions()
    options.setMSLevels([1])
    fh = pyopenms.MzXMLFile()
    fh.setOptions(options)

    input_map = pyopenms.MSExperiment()
    fh.load(filename, input_map)
    input_map.updateRanges()
    return input_map


def parse_ms1_mzdata(filename):
    options = pyopenms.PeakFileOptions()
    options.setMSLevels([1])
    fh = pyopenms.MzDataFile()
    fh.setOptions(options)

    input_map = pyopenms.MSExperiment()
    fh.load(filename, input_map)
    input_map.updateRanges()
    return input_map

def feature_from_file_features_to_chromatograms(input_map, features, w):
    chromatograms = []
    for f in features:
        ch = feature_to_chromatogram(f, input_map, w)
        if not ch.empty:
            ch.feature_id = [f.intesity, f.rt, f.mz]
            ch.cut_smallest_peaks(0.001)
            chromatograms.append(ch)
    return chromatograms

def parse_feature_with_model_xml(filename):
    features = MyCollection()
    tree = ET.parse(filename)
    root = tree.getroot()
    feature_list = root.find("./featureList")
    for i, feature in enumerate(feature_list.findall("./feature")):
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
            print("Skipping small feature, id:", feature.attrib["id"])
            continue
    return features


def feature_from_file_experiment_chromatogram_set(filename, features_filename):
    input_map = parse_ms1_xml(filename)
    features = parse_feature_with_model_xml(features_filename)
    weight = features_to_weight(features)
    print("Parsed file", filename, "\n", features.size(),
          "features found,\nAverage lenght to width:", weight)
    return feature_from_file_features_to_chromatograms(input_map, features, weight)


def parse_feature_from_file_alignment_experiment_chromatogram_sets(
        chromatogram_filenames, features_filenames):
    ch_sets_list = []
    for ch_fname, f_fname in zip(chromatogram_filenames, features_filenames):
        ch_sets_list.append(
            feature_from_file_experiment_chromatogram_set(ch_fname, f_fname))

    return ch_sets_list


def find_features(input_map):
    ff = pyopenms.FeatureFinder()
    ff.setLogType(pyopenms.LogType.CMD)

    # Run the feature finder
    name = "centroided"
    features = pyopenms.FeatureMap()
    seeds = pyopenms.FeatureMap()
    params = pyopenms.FeatureFinder().getParameters(name)

    ff.run(name, input_map, features, params, seeds)

    features.setUniqueIds()

    return features


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


def gather_widths_lengths(features):
    lengths_rt = []
    widths_mz = []
    for f in features:
        ch = f.getConvexHull()
        rt_max, mz_max = ch.getBoundingBox().maxPosition()
        rt_min, mz_min = ch.getBoundingBox().minPosition()
        lengths_rt.append(rt_max - rt_min)
        widths_mz.append(mz_max - mz_min)

    lengths_rt = np.array(lengths_rt)
    widths_mz = np.array(widths_mz)

    return lengths_rt, widths_mz


def get_weight_from_widths_lengths(lengths_rt, widths_mz):
    return np.mean(lengths_rt) / np.mean(widths_mz)


def features_to_weight(features):
    return get_weight_from_widths_lengths(*gather_widths_lengths(features))


# def gather_ch_stats_over_exps(dirname):
#     fnames = []
#     means = []
#     variances = []
#     for i, filename in enumerate(os.listdir(dirname)):
#         if filename.endswith(".mzXML") and i % 5 == 0:
#             print("Now working on:", filename)
#             input_map = parse_ms1_xml(os.path.join(dirname, filename))
#             gc.collect()
#             features = find_features(input_map)
#             lengths, widths = gather_widths_lengths(features)
#             means.append((np.mean(widths), np.mean(lengths),
#                          np.mean(lengths) / np.mean(widths)))
#             variances.append((np.std(widths), np.std(lengths),
#                              np.std(lengths) / np.std(widths)))
#     return means, variances
def feature_to_chromatogram(feature, input_map, w):
    # TODO very inefficient way to do it, correct it
    # TODO it may not work properly (takes too many peaks), but still, everything here is not properly definied
    # 1. choose spectra by RT
    # 2. iterate over mz and check if is enclosed
    max_rt, max_mz = feature.getConvexHull().getBoundingBox().maxPosition()
    min_rt, min_mz = feature.getConvexHull().getBoundingBox().minPosition()

    mzs = []
    rts = []
    ints = []

    for open_spectrum in input_map:
        rt = open_spectrum.getRT()
        if min_rt <= rt and rt <= max_rt:
            for mz, i in zip(*open_spectrum.get_peaks()):
                if min_mz <= mz and mz <= max_mz:
                    mzs.append(mz)
                    rts.append(rt)
                    ints.append(i)
    if len(rts) == 0:
        print("zero length", w)
    ch = Chromatogram(rts, mzs, ints, w)
    ch.scale_rt()
    ch.normalize()
    return ch

def features_to_chromatograms(input_map, features, w):
    chromatograms = []
    for f in features:
        chromatograms.append(feature_to_chromatogram(f, input_map, w))
    return chromatograms


def chromatogram_dist(ch1, ch2, penalty=40):
    dists = cdist(np.column_stack((ch1.rts, ch1.mzs)),
                  np.column_stack((ch2.rts, ch2.mzs)), 'cityblock')
    return distance_dense(ch1.ints, ch2.ints, dists=dists, eps=1, lam=penalty, method="TV")


def mid_dist(ch1, ch2):
    return np.sum(np.abs(ch1.mid - ch2.mid))


def mid_mz_dist(ch1, ch2):
    return abs(ch1.mid[1] - ch2.mid[1])

def calc_two_ch_sets_dists(chromatograms1, chromatograms2,
                           sinkhorn_upper_bound=40):
    total = len(chromatograms1) * len(chromatograms2)

    ch_dists = np.full((len(chromatograms1), len(chromatograms2)), np.inf)

    tick = 0
    pbar = tqdm(total=total)
    print("Calculating distances")
    for i, chi in enumerate(chromatograms1):
        for j, chj in enumerate(chromatograms2):
            # TODO maybe remove mid MZ heuristic
            if mid_dist(chi, chj) <= sinkhorn_upper_bound and mid_mz_dist(chi, chj) < 2:
#                 print(i,j, len(chi), len(chj))
                ch_dists[i,j] = chromatogram_dist(chi, chj, sinkhorn_upper_bound)
            tick += 1
            if tick % 300 == 0:
                pbar.update(300)
    pbar.close()
    print("Calculated dists, number of nans:", np.sum(np.isnan(ch_dists)))
    print("All columns have any row non zero:", np.all(np.any(ch_dists < np.inf, axis=0)))
    print("All rows have any column non zero:", np.all(np.any(ch_dists < np.inf, axis=1)))
    return ch_dists



def match_chromatograms(ch_dists, penalty=40):

    # 1 - from, 2 - to, 0 - s, t node, -1 is trash node
    G = nx.DiGraph()
    ROUNDING_COEF = 10

    # WARNING! Both unmatched nodes pay for not pairing, contrary to pairing,
    # where distance is paid once. So half of penalty should be interpreted as max
    # distance for which it is acceptable to make a matching.

    inds = np.nonzero(ch_dists < np.inf)
    for i, j, dist in zip(*inds, ch_dists[inds]):
        G.add_edge((1, i), (2, j), capacity=1,
                   weight=int(round(dist * ROUNDING_COEF)))

    for i in range(ch_dists.shape[0]):
        G.add_edge((0, "s"), (1, i), capacity=1)
        G.add_edge((1, i), (-1, "trash"), capacity=1,
                   weight=penalty * ROUNDING_COEF)

    if ch_dists.shape[0] > ch_dists.shape[1]:  # if equal, add no extra rubbish path
        G.add_edge((-1, "trash"), (0, "t"),
                   capacity=ch_dists.shape[0] - ch_dists.shape[1])

    for i in range(ch_dists.shape[1]):
        G.add_edge((2, i), (0, "t"), capacity=1)
        G.add_edge((-1, "trash"), (2, i), capacity=1,
                   weight=penalty * ROUNDING_COEF)

    if ch_dists.shape[1] > ch_dists.shape[0]:  # if equal, add no extra rubbish path
        G.add_edge((0, "s"), (-1, "trash"),
                   capacity=ch_dists.shape[1] - ch_dists.shape[0])

    min_cost_flow = nx.max_flow_min_cost(G, (0, "s"), (0, "t"))

    return min_cost_flow


def extract_matching_from_flow(min_cost_flow):
    matchings = []
    matched_from = set()
    matched_to = set()
    # unmatched_from = set()
    # matched_to = set()

    min_cost_flow.pop((0, "s"), None)
    min_cost_flow.pop((0, "t"), None)

    for from_type, from_id in min_cost_flow:
        if from_type == 1:
            for to_type, to_id in min_cost_flow[(from_type, from_id)]:
                if to_type == 2 and min_cost_flow[(from_type, from_id)][(to_type, to_id)]:
                    matchings.append((from_id, to_id))
                    matched_from.add(from_id)
                    matched_to.add(to_id)

    return matchings, matched_from, matched_to  # , unmatched_from, unmatched_to


def chromatogram_sets_from_mzxml(filename):
    input_map = parse_ms1_xml(filename)
    features = find_features(input_map)
    weight = features_to_weight(features)
    print("Parsed file", filename, "\n", features.size(),
          "features found,\nAverage lenght to width:", weight)
    return features_to_chromatograms(input_map, features, weight)


def align_chromatogram_sets(ch_dists, flow_trash_penalty=40):
    return extract_matching_from_flow(match_chromatograms(ch_dists, flow_trash_penalty))


def gather_mids(chromatograms_sets_list):
    # it turns out that the best solution is to do it on python ordinary lists
    mzs = []
    rts = []
    chromatogram_indices = []
    for i, chromatogram_set in enumerate(chromatograms_sets_list):
        for j, chromatogram in enumerate(chromatogram_set):
            mzs.append(np.mean(chromatogram.mzs))
            rts.append(np.mean(chromatogram.rts))
            chromatogram_indices.append((i, j))
    return (np.array(list(zip(rts, mzs))).reshape((-1, 2)),
            np.array(chromatogram_indices))

def cluster_mids(mids, distance_threshold=20):
#     return DBSCAN(eps=2.5, min_samples=5, metric='l1', metric_params=None,
#              algorithm='auto', leaf_size=44, p=None, n_jobs=5).fit_predict(mids)
    return MiniBatchKMeans(n_clusters=16, init='k-means++', max_iter=100,
                           batch_size=100, verbose=0, compute_labels=True,
                           random_state=None, tol=0.0, max_no_improvement=10,
                           init_size=None, n_init=3, reassignment_ratio=0.01).fit_predict(mids)
#     return OPTICS(min_samples=5, max_eps=distance_threshold, eps=distance_threshold,
#                   cluster_method="dbscan", p=1).fit_predict(mids)

def cluster_mids_subsets(mids, distance_threshold=20):
    return AgglomerativeClustering(n_clusters=None, affinity="l1",
                                   linkage='complete',
                                   distance_threshold=distance_threshold,
                                   ).fit_predict(mids)


def create_chrom_sums(chromatograms_sets_list, clusters, chromatogram_indices):
    idx_sort = np.argsort(clusters)
    vals, idx_start, count = np.unique(clusters[idx_sort],
                                       return_counts=True, return_index=True)
    chromatogram_indices_by_clusters = np.split(idx_sort, idx_start[1:])
    print("Average cluster size:", np.mean(
        list(map(len, chromatogram_indices_by_clusters))))
    result_ch_set = []
    for i, one_cluster_indices in enumerate(chromatogram_indices_by_clusters):
        cluster_chroms = []
        for ch_set_id, ch_id in chromatogram_indices[one_cluster_indices]:
            cluster_chroms.append(chromatograms_sets_list[ch_set_id][ch_id])
        new_chromatogram = Chromatogram.sum_chromatograms(cluster_chroms)

        new_chromatogram.cut_smallest_peaks(0.005)
        result_ch_set.append(new_chromatogram)
    return result_ch_set


def find_consensus_features(clustered_chromatogram_set, chromatograms_sets_list,
                            sinkhorn_upper_bound=40, flow_trash_penalty=5):
    consensus_features = [[] for _ in range(len(clustered_chromatogram_set))]
    all_matched_left = []

    for i, ch_set in enumerate(chromatograms_sets_list):
        #         print(i)
        c_dists = calc_two_ch_sets_dists(ch_set, clustered_chromatogram_set,
                                         sinkhorn_upper_bound=sinkhorn_upper_bound)
        matchings, matched_left, matched_right = align_chromatogram_sets(
            c_dists, flow_trash_penalty=flow_trash_penalty)
        for chromatogram_j, feature_ind in matchings:
            consensus_features[feature_ind].append((i, chromatogram_j))

        all_matched_left.append(matched_left)

    return consensus_features, all_matched_left


def dump_consensus_features(consensus_features, filename,
                            chromatograms_sets_list):
    rows = []
    with open(filename, "w") as outfile:
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
                                     sinkhorn_upper_bound=sinkhorn_upper_bound)
    matchings, matched_left, matched_right = align_chromatogram_sets(
        c_dists, flow_trash_penalty=flow_trash_penalty)

    for left_f_ind, right_f_ind in matchings:
        consensus_features.append([(0, left_f_ind), (1, right_f_ind)])

    return consensus_features, [matched_left]
