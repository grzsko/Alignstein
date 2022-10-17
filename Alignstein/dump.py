def dump_consensus_features(consensus_features, filename,
                            chromatograms_sets_list):
    rows = []
    with open(filename, "w") as outfile:
        for consensus_feature in consensus_features:
            row = []
            next_set_id = 0
            for set_i, chromatogram_j in sorted(consensus_feature):
                # Leave empty space for not found consensus features
                while set_i > next_set_id:
                    row.append("")
                    next_set_id += 1
                f_id = chromatograms_sets_list[set_i][chromatogram_j].feature_id
                row.append(f_id)
            rows.append(" ".join(map(str, row)))
        outfile.write("\n".join(rows))


def dump_consensus_features_caap_style(consensus_features, out_filename,
                                       chromatograms_sets_list,
                                       all_pyopenms_features):
    rows = []
    with open(out_filename, "w") as outfile:
        for consensus_feature in consensus_features:
            if len(consensus_feature) > 1:
                row = []
                for set_i, chromatogram_j in consensus_feature:
                    caap_id = chromatograms_sets_list[set_i][
                        chromatogram_j].ext_id
                    pyopenms_feature = all_pyopenms_features[set_i][caap_id]
                    row.extend([pyopenms_feature.getIntensity(),
                                pyopenms_feature.getRT(),
                                pyopenms_feature.getMZ()])
                rows.append(" ".join(map(str, row)))
        outfile.write("\n".join(rows))
