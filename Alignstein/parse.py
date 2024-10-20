import os
import xml.etree.ElementTree as ET

import numpy as np
import pyopenms

# from .OpenMSMimicry import MyCollection, OpenMSFeatureMimicry

from .chromatogram import Chromatogram


def parse_ms1_mzxml(filename):
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


def parse_ms1_mzml(filename):
    options = pyopenms.PeakFileOptions()
    options.setMSLevels([1])
    fh = pyopenms.MzMLFile()
    fh.setOptions(options)

    input_map = pyopenms.MSExperiment()
    fh.load(filename, input_map)
    input_map.updateRanges()
    return input_map


def find_features(input_map):
    """
    Detect features using OpenMS FeatureFinderCentroid algorithm.

    Parameters
    ----------
    input_map : pyopenms.InputMap
        chromatogram, which features should be detected

    Returns
    -------
    pyopenms.FeatureMap
        Features detected by OpenMS
    """
    ff = pyopenms.FeatureFinder()
    ff.setLogType(pyopenms.LogType.CMD)

    # Run the openms_feature finder
    name = "centroided"
    features = pyopenms.FeatureMap()
    seeds = pyopenms.FeatureMap()
    params = pyopenms.FeatureFinder().getParameters(name)

    ff.run(name, input_map, features, params, seeds)

    features.setUniqueIds()

    return features


def parse_chromatogram_file(filename):
    """
    Parse data contained in chromatogram file.

    Parameters
    ----------
    filename : str
        Input chromatogram filename, accepted: mzML, mzXML, zmData.

    Returns
    -------
    pyopenms.MSExperiment
        Chromatogram parsed by OpenMS
    """
    # Warn user that it is ms1 or change name to smth better
    file_extension = os.path.splitext(filename)[-1].lower()
    if file_extension == ".mzml":
        return parse_ms1_mzml(filename)
    elif file_extension == ".mzxml":
        return parse_ms1_mzxml(filename)
    elif file_extension == ".mzdata":
        return parse_ms1_mzdata(filename)
    else:
        raise ValueError("Filetype {} not supported for file {}".format(
            file_extension, filename))


def gather_widths_lengths(feature_set: list[Chromatogram]):
    """
    Collect features' widths and lengths.

    Parameters
    ----------
    feature_set : iterable of Chromatogram
        Features for which widths and lengths will  be collected.

    Returns
    -------
    tuple of lists
        Tuple containing features lengths and widths.

    """
    lengths_rt = []
    widths_mz = []
    for f in feature_set:
        rt_max, mz_max, rt_min, mz_min = f.get_bounding_box()
        lengths_rt.append(rt_max - rt_min)
        widths_mz.append(mz_max - mz_min)

    lengths_rt = np.array(lengths_rt)
    widths_mz = np.array(widths_mz)

    return lengths_rt, widths_mz


def openms_feature_bounding_box(openms_feature):
    ch = openms_feature.getConvexHull()
    rt_max, mz_max = ch.getBoundingBox().maxPosition()
    rt_min, mz_min = ch.getBoundingBox().minPosition()
    return rt_max, mz_max, rt_min, mz_min


def get_weight(lengths_rt, widths_mz):
    """
    Compute factor by which RT should be scaled.

    Parameters
    ----------
    lengths_rt : iterable of float
        Features lengths.
    widths_mz : iterable of float
        Features widths.

    Returns
    -------
    float
        Average feature length to width.
    """
    # Omitting zero values as meaningless.
    return (np.mean(lengths_rt[0.0 < lengths_rt]) /
            np.mean(widths_mz[0.0 < widths_mz]))


def features_to_weight(features: list[Chromatogram]):
    return get_weight(*gather_widths_lengths(features))


def openms_feature_to_feature(openms_feature, input_map):
    """
    Gather signal from chromatogram contained in feature bounding box.

    Feature with gathered signal is a chromatograms subset, so we represent it
    using Chromatograms class.

    Algorithm scheme:
        For every OpenMS feature do:
        1. choose spectra by RT
        2. iterate over mz and check if is enclosed
    It is done non-effectively, but how to do it better?

    Parameters
    ----------
    openms_feature : pyopenms.Feature or OpenMSFeatureMimicry
        OpenMS-like object for representing feature.
    input_map : pyopenms.InputMap
        Parsed chromatogram.

    Returns
    -------
    Chromatogram
        A feature with gathered signal.
    """
    def bisect_rt(ms_experiment, rt):
        lo = 0
        hi = len(ms_experiment.getSpectra())

        while lo < hi:
            mid = (lo + hi) // 2
            if ms_experiment[mid].getRT() < rt:
                lo = mid + 1
            else:
                hi = mid
        return lo

    max_rt, max_mz = openms_feature.getConvexHull().getBoundingBox().maxPosition()
    min_rt, min_mz = openms_feature.getConvexHull().getBoundingBox().minPosition()

    mzs = []
    rts = []
    ints = []

    spectra_number = len(input_map.getSpectra())
    spectrum_index = bisect_rt(input_map, min_rt)
    while spectrum_index < spectra_number:
        openms_spectrum = input_map[spectrum_index]
        rt = openms_spectrum.getRT()
        if rt > max_rt:
            break
        openms_mzs, openms_intens = openms_spectrum.get_peaks()
        mz_index = np.searchsorted(openms_mzs, min_mz)
        for mz, inten in zip(openms_mzs[mz_index:], openms_intens[mz_index:]):
            if mz > max_mz:
                break
            mzs.append(mz)
            rts.append(rt)
            ints.append(inten)
        spectrum_index += 1

    ch = Chromatogram(rts, mzs, ints, openms_feature.getMZ())
    ch.normalize(keep_old=True)
    # ch.openms_feature = openms_feature # Usable
    if hasattr(openms_feature, 'getUniqueId'):
        # Sometimes it is useful to have some information from other software
        #
        ch.ext_id = openms_feature.getUniqueId()
    return ch


def openms_features_to_features(input_map, openms_features, omit_empty=True):
    """
    Gather signal over all OpenMS-like features and create Alignstein features.

    We represent features with gathered signal as a subset of chromatogram so
    we use Chromatogram class.

    Parameters
    ----------
    openms_features : pyopenms.FeatureMap or MyCollection
        Iterable with OpenMS-like objects representing features.
    input_map : pyopenms.InputMap
        Parsed chromatogram.
    omit_empty : bool
        Should features marked with empty flag be added to result list of
        features?

    Returns
    -------
    list of Chromatogram
        A list of features with gathered signal.
    """
    chromatograms = []
    # Chromatogram class is universal, so we use it to represent chromatograms
    # subsets, i.e. features.
    for i, openms_feature in enumerate(openms_features):
        feature = openms_feature_to_feature(openms_feature, input_map)
        if (not omit_empty) or (not feature.empty):
            # i.e. omit only when both are true
            feature.feature_id = i
            # i is index of original feature, not omitted
            # We identify features as detected, other enumeration (e.g. from
            # OpenMS) is stored in ext_id
            chromatograms.append(feature)
    for i, feature in enumerate(chromatograms):
        feature.feature_id = i
    return chromatograms

def detect_openms_features(filename):
    input_map = parse_chromatogram_file(filename)
    openms_features = find_features(input_map)
    pyopenms.FeatureXMLFile().store(filename + ".featureXML", openms_features)
    return input_map, openms_features


def detect_features_from_file(filename):
    """
    Parse and detect features from chromatogram contained in file.

    This function parses chromatograms contained in file names `filename`,
    detects features, collects all signal contained within features and
    return features represented as chromatogram subsets.

    Parameters
    ----------
    filename : str
        input chromatogram filename

    Returns
    -------
    list of Chromatogram
        Iterable of parsed features represented as chromatograms subsets.
    """
    input_map, openms_features = detect_openms_features(filename)
    print("Parsed file", filename, "\n", openms_features.size(),
          "OpenMS features found,\n")
    features = openms_features_to_features(input_map, openms_features)
    return features


def parse_features_from_file(feature_filename):
    """
    Parse features from featureXML file.

    Parameters
    ----------
    feature_filename : str
        Filename of features (.featureXML) to be parsed. Must have same order
        as chromatograms_filenames. Every feature of file should contain field
        convexhull which describes the spatial properties of features.

    Returns
    -------
    pyopenms.FeatureMap
        Parsed features with API conforming pyopenms API.
    """
    features = pyopenms.FeatureMap()
    feature_xml_file = pyopenms.FeatureXMLFile()
    feature_xml_file.load(feature_filename, features)
    return features


def parse_chromatogram_with_detected_features(filename, features_filename):
    """
    Parse chromatogram with provided featureXML file.

    Parameters
    ----------
    filename : str
        Filename of chromatogram to be parsed.
    features_filename : str
        Filename of features (.featureXML) to be parsed. Must have the same
        order as chromatograms_filenames. Every feature of file should contain
        field convexhull which describes the spatial properties of features.

    Returns
    -------
    list of Chromatogram
        Parsed features with collected signal.
    """
    input_map = parse_chromatogram_file(filename)
    openms_like_features = parse_features_from_file(features_filename)
    print("Parsed file", filename, "\n", openms_like_features.size(),
          "OpenMS features found.")
    return openms_features_to_features(input_map, openms_like_features)