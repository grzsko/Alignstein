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

if __name__ == "__main__":
    arguments = docopt(__doc__)

    if len(arguments["-f"]) == 0:
        chromatograms_sets_list = [chromatogram_sets_from_mzxml(filename) for
                                   filename in arguments["MZML_FILE"]]
    else:
        chromatograms_sets_list = parse_feature_from_file_alignment_experiment_chromatogram_sets(
            arguments["MZML_FILE"], arguments["-f"])
    #if len(chromatograms_sets_list) > 2:
    mids, ch_indices = gather_mids(chromatograms_sets_list)

    big_clusters = cluster_mids(mids, distance_threshold=20)
    clusters = big_clusters_to_clusters(mids, big_clusters,
                                        distance_threshold=30)
    clustered_chromatogram_set = create_chrom_sums(chromatograms_sets_list,
                                                clusters, ch_indices)
    consensus_features, matched_all_sets = find_consensus_features(
            clustered_chromatogram_set, chromatograms_sets_list,
            sinkhorn_upper_bound=20, flow_trash_penalty=5
    )
    # else:
    #     pass
    dump_consensus_features(consensus_features, "align.out", chromatograms_sets_list)
