"""Parse mzml files and perform alignment

Usage: align.py -h
       align.py [-f FEATURE_FILE...] MZML_FILE...

Arguments:
    MZML_FILE        names of files with chromatograms to be aligned

Options:
    -f FEATURE_FILE  names of files with detected features in chromatograms,
                     order of filenames should conform order of input data
                     files. If not provided features are detected and dumped
                     into featureXML files.
"""

from docopt import docopt

from align import *
from multialign import *
from parse import *

if __name__ == "__main__":
    arguments = docopt(__doc__)

    # Parsing
    if len(arguments["-f"]) == 0:
        feature_sets_list = [
            detect_features_from_file(fname) for fname in arguments["MZML_FILE"]
        ]
    else:
        feature_sets_list = [
            parse_chromatogram_with_detected_features(ch_fname, fs_fname)
            for ch_fname, fs_fname in zip(arguments["MZML_FILE"],
                                          arguments["-f"])]
    # RT scaling
    C = 5  # We scale additionaly by 5, to make RT more smashed, it works fine.
    weights = [features_to_weight(f_set) for f_set in feature_sets_list]
    average_weight = np.mean(weights)
    for feature_set in feature_sets_list:
        for feature in feature_set:
            feature.scale_rt(average_weight * C)

    # Aligning
    if len(arguments["MZML_FILE"]) > 2:
        mids, ch_indices = gather_mids(feature_sets_list)

        big_clusters = precluster_mids(mids, distance_threshold=20)
        clusters = big_clusters_to_clusters(mids, big_clusters,
                                            distance_threshold=30)
        consensus_features, matched_all_sets = find_consensus_features(
            clusters, feature_sets_list,
            sinkhorn_upper_bound=20, flow_trash_penalty=5
        )
    else:
        consensus_features, matched_all_sets = find_pairwise_consensus_features(
                *feature_sets_list, sinkhorn_upper_bound=20, flow_trash_penalty=5)

    # Dump
    dump_consensus_features(consensus_features, "align.out",
                            feature_sets_list)
