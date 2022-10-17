"""Parse mzml files and perform alignment

Usage: align.py -h
       align.py [-c SCALING_CONST] [-t MIDS_THRSH] [-m MIDS_UP_BOUND]
                [-w GWD_UP_BOUND] [-p PENALTY] [-o OUT_FILENAME]
                [-f FEATURE_FILE]... MZML_FILE...

Arguments:
    MZML_FILE      names of files with chromatograms to be aligned

Options:
    -i MZML_FILE      names of files with chromatograms to be aligned
    -f FEATURE_FILE   names of files with detected features in chromatograms,
                      order of filenames should conform order of input data
                      files. POSIX compliance requires to every features
                      filename be preceded by -f
    -o OUT_FILENAME   Output consensus filename [default: consensus.out]
    -c SCALING_CONST  Additional constant by which RT should be scaled.
                      [default: 1]
    -t MIDS_THRSH     Distance threshold between centroid in one cluster. Not
                      applicable when aligning two chromatograms. [default: 1.5]
    -m MIDS_UP_BOUND  Maximum cetroid distance between which GWD will be computed.
                      For efficiency reasons should be reasonably small.
                      [default: 10]
    -w GWD_UP_BOUND   Cost of not transporting a part of signal, aka the
                      lambda parameter. Can be interpreted as maximal distance
                      over which signal is transported while computing GWD.
                      [default: 10]
    -p PENALTY        penalty for feature not matching. [default: 10]
"""
# TODO add parameter for verbosity level

from docopt import docopt

from .align import *
from .multialign import *
from .parse import *
from .dump import *


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
    # RT more squeezed, for C between 5 and 10 it works fine.
    weights = [features_to_weight(f_set) for f_set in feature_sets_list]
    average_weight = np.mean(weights)
    for feature_set in feature_sets_list:
        for feature in feature_set:
            feature.scale_rt(average_weight * C)

    # Aligning
    if len(chromatogram_filenames) > 2:
        mids, big_clusters, clusters = cluster_mids(
            feature_sets_list, distance_threshold=float(arguments["-t"]))
        # TODO Pararel function chosen by parameter
        consensus_features = find_consensus_features_paralel(
            clusters, feature_sets_list,
            centroid_upper_bound=float(arguments["-m"]),
            gwd_upper_bound=float(arguments["-w"]),
            matching_penalty=float(arguments["-p"]),
            turns=10,  # 10 is enough
            big_clusters=big_clusters
        )
    else:
        consensus_features, _ = find_pairwise_consensus_features(
            *feature_sets_list,
            centroid_upper_bound=float(arguments["-m"]),
            gwd_upper_bound=float(arguments["-w"]),
            matching_penalty=float(arguments["-p"])
        )

    # Dump
    if len(arguments["-f"]) > 0:
        openms_features = [parse_features_from_file(filename)
                           for filename in arguments["-f"]]
    else:
        openms_features = [detect_openms_features(filename)[1]
                           for filename in chromatogram_filenames]
    dump_consensus_features_caap_style(consensus_features, arguments["-o"],
                                       feature_sets_list, openms_features)


if __name__ == "__main__":
    main()
