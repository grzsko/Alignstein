"""Parse mzml files and perform alignment

Usage: align.py -h
       align.py [-c SCALING_CONST] [-t MIDS_THRSH] [-f FEATURE_FILE...] MZML_FILE...

Arguments:
    MZML_FILE        names of files with chromatograms to be aligned

Options:
    -f FEATURE_FILE  names of files with detected features in chromatograms,
                     order of filenames should conform order of input data
                     files. If not provided features are detected and dumped
                     into featureXML files.
    -c SCALING_CONST additional contant by which RT should be scaled
    -t MIDS_THRSH    Distance threshold between centroid in one cluster. Not
                     applicable when aligning two chromatograms. [default: 1.5]
    -m MIDS_UP_BOUND Maximum cetroid distance between which GWD will computed.
                     For efficiency reasons should be reasonably small.
                     [default: 10]
    -w GWD_UP_BOUND  Cost of not transporting a part of signal, aka the
                     lambda parameter. Can be interpreted as maximal distance
                     over which signal is transported while computing GWD.
                     [default: 10]
    -p PENALTY       penalty for feature not matching [default: 10]
"""

from docopt import docopt

from .align import *
from .multialign import *
from .parse import *


def main():
    arguments = docopt(__doc__)
    chromatogram_filenames = arguments["MZML_FILE"]
    feature_filenames = arguments["-f"]

    # Parsing
    if len(arguments["-f"]) == 0:
        feature_sets_list = [
            detect_features_from_file(fname) for fname in chromatogram_filenames
        ]
    else:
        feature_sets_list = [
            parse_chromatogram_with_detected_features(ch_fname, fs_fname)
            for ch_fname, fs_fname in zip(chromatogram_filenames,
                                          feature_filenames)]
    # RT scaling
    C = float(arguments["-c"])  # We scale additionaly by C, to make
    # RT more smashed, C between 5 and 10 it works fine.
    weights = [features_to_weight(f_set) for f_set in feature_sets_list]
    average_weight = np.mean(weights)
    for feature_set in feature_sets_list:
        for feature in feature_set:
            feature.scale_rt(average_weight * C)

    # Aligning
    if len(arguments["MZML_FILE"]) > 2:
        mids = gather_mids(feature_sets_list)

        big_clusters = precluster_mids(mids)
        clusters = big_clusters_to_clusters(
            mids, big_clusters, distance_threshold=float(arguments["-t"]))
        consensus_features, matched_all_sets = find_consensus_features(
            clusters, feature_sets_list,
            centroid_upper_bound=float(arguments["-m"]),
            gwd_upper_bound=float(arguments["-w"]),
            matching_penalty=float(arguments["-p"]),
            turns=10  # turn=10 is enough
        )
    else:
        consensus_features, matched_all_sets = find_pairwise_consensus_features(
            *feature_sets_list,
            centroid_upper_bound=float(arguments["-m"]),
            gwd_upper_bound=float(arguments["-w"]),
            matching_penalty=float(arguments["-p"])
        )

    # Dump
    dump_consensus_features(consensus_features, "consensus.csv",
                            feature_sets_list)


if __name__ == "__main__":
    main()
