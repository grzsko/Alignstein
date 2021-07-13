"""Parse mzml files and perform alignment

Usage: align.py -h
       align.py MZML_FILE MZML_FILE...

Arguments:
    MZML_FILE       names of files with chromatograms to be aligned

"""
from docopt import docopt
from pyopenms import *
import numpy as np

from chromatogram import Chromatogram

def parse_ms1_xml(filename):
    options = PeakFileOptions()
    options.setMSLevels([1])
    fh = MzXMLFile()
    fh.setOptions(options)

    input_map = pyopenms.MSExperiment()
    fh.load(filename, input_map)
    input_map.updateRanges()
    return input_map

def find_features(input_map):
    ff = FeatureFinder()
    ff.setLogType(LogType.CMD)

    # Run the feature finder
#     name = "isotope_wavelet"
    name = "centroided"
    features = FeatureMap()
    seeds = FeatureMap()
    params = FeatureFinder().getParameters(name)
#     params.setValue('isotopic_pattern:mass_window_width', 200.0)
#     params.setValue("isotopic_pattern:mz_tolerance", 0.2)

    ff.run(name, input_map, features, params, seeds)

    features.setUniqueIds()
    # fh = FeatureXMLFile()
    # fh.store("output.featureXML", features)
#     print("Found", features.size(), "features")

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
    return  np.mean(lengths_rt) / np.mean(widths_mz)

def features_to_weight(features):
    return get_weight_from_widths_lengths(*gather_widths_lengths(features))

def gather_ch_stats_over_exps(dirname):
    fnames = []
    means = []
    variances = []
    for i, filename in enumerate(os.listdir(dirname)):
        if filename.endswith(".mzXML") and i % 5 == 0:
            print("Now working on:", filename)
            input_map = parse_ms1_xml(os.path.join(dirname, filename))
            gc.collect()
            features = find_features(input_map)
            lengths, widths = gather_widths_lengths(features)
            means.append((np.mean(widths), np.mean(lengths), np.mean(lengths)/ np.mean(widths)))
            variances.append((np.std(widths), np.std(lengths), np.std(lengths)/ np.std(widths)))
    return means, variances

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
    return spectral_distance_dense(ch1.ints, ch2.ints, dists, eps=1, lam=penalty, method="TV")

def mid_dist(ch1, ch2):
    return np.sum(np.abs(ch1.mid - ch2.mid))

def calc_two_ch_sets_dists(chromatograms1, chromatograms2,
                           sinkhorn_upper_bound=40):
    total = len(chromatograms1) * len(chromatograms2)

    ch_dists = np.full((len(chromatograms1), len(chromatograms2)), np.inf)

    tick = 0
    pbar = tqdm(total=total)
    print("Calculating distances")
    for i, chi in enumerate(chromatograms1):
        for j, chj in enumerate(chromatograms2):
            if mid_dist(chi, chj) <= sinkhorn_upper_bound:
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
        G.add_edge((1, i), (2, j), capacity=1, weight=int(round(dist * ROUNDING_COEF)))

    for i in range(ch_dists.shape[0]):
        G.add_edge((0, "s"), (1,i), capacity=1)
        G.add_edge((1, i), (-1, "trash"), capacity=1, weight=penalty * ROUNDING_COEF)

    if ch_dists.shape[0] > ch_dists.shape[1]: # if equal, add no extra rubbish path
        G.add_edge((-1, "trash"), (0, "t"), capacity=ch_dists.shape[0] - ch_dists.shape[1])

    for i in range(ch_dists.shape[1]):
        G.add_edge((2, i), (0, "t"), capacity=1)
        G.add_edge((-1, "trash"), (2, i), capacity=1, weight=penalty * ROUNDING_COEF)

    if ch_dists.shape[1] > ch_dists.shape[0]: # if equal, add no extra rubbish path
        G.add_edge((0, "s"), (-1, "trash"), capacity=ch_dists.shape[1] - ch_dists.shape[0])

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
    # min_cost_flow.pop((0, "pre_s"), None)
    # min_cost_flow.pop((0, "post_t"), None)

    for from_type, from_id in min_cost_flow:
        if from_type == 1:
            for to_type, to_id in min_cost_flow[(from_type, from_id)]:
                if to_type == 2 and min_cost_flow[(from_type, from_id)][(to_type, to_id)]:
                    matchings.append((from_id, to_id))
                    matched_from.add(from_id)
                    matched_to.add(to_id)

#     unmatched_from = set(range(len(chromatograms1))) - matched_from
#     unmatched_to = set(range(len(chromatograms2))) - matched_to

    return matchings, matched_from, matched_to#, unmatched_from, unmatched_to

def chromatogram_sets_from_mzxml(filename):
    input_map = parse_ms1_xml(filename)
    features = find_features(input_map)
    weight = features_to_weight(features)
    print("Parsed file", filename, "\n", features.size(), "features found,\nAverage lenght to width:", weight)
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
            chromatogram_indices.append((i,j))
    return (np.array(list(zip(rts, mzs))).reshape((-1, 2)),
            np.array(chromatogram_indices))

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
    print("Average cluster size:", np.mean(list(map(len, chromatogram_indices_by_clusters))))
    result_ch_set = []
    for i, one_cluster_indices in enumerate(chromatogram_indices_by_clusters):
        cluster_chroms = []
        for ch_set_id, ch_id in chromatogram_indices[one_cluster_indices]:
            cluster_chroms.append(chromatograms_sets_list[ch_set_id][ch_id])
        new_chromatogram = Chromatogram.sum_chromatograms(cluster_chroms)
#         new_chromatogram.normalize()
#         l = len(new_chromatogram)

        new_chromatogram.cut_smallest_peaks(0.005)
#         if (l > 7000):
#             print(l, "->", len(new_chromatogram))
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
    all_caap_features = []
    for i in range(len(chromatograms_sets_list)):
        _, caap_features = parse_feature_xml(
            "examples/LCMS_data/benchmark_datasets/M1_set/M1_features/M1_{}.featureXML".format(i + 1))
        all_caap_features.append(caap_features)
    rows = []
    with open(filename, "w") as outfile:
        for consensus_feature in consensus_features:
            row = []
            for set_i, chromatogram_j in consensus_feature:
                caap_id = chromatograms_sets_list[set_i][chromatogram_j].caap_feature_id
                caap_feature = all_caap_features[set_i][caap_id]
                row.extend([caap_feature.int, caap_feature.rt, caap_feature.mz])
            rows.append(" ".join(map(str, row)))
        outfile.write("\n".join(rows))

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)

    chromatograms_sets_list = [chromatogram_sets_from_mzxml(filename) for
                               filename  in arguments["MZML_FILE"]]
    mids, ch_indices = gather_mids(chromatograms_sets_list)

    clusters = cluster_mids(mids, distance_threshold=20)
    clustered_chromatogram_set = create_chrom_sums(chromatograms_sets_list,
                                                   clusters, ch_indices)
    consensus_features, matched_all_sets = find_consensus_features(
            clustered_chromatogram_set, chromatograms_sets_list,
        #     clustered_chromatogram_set, [chromatograms3],
            sinkhorn_upper_bound=20, flow_trash_penalty=5
    )
    dump_consensus_features(consensus_features, "align.out", chromatograms_sets_list)
