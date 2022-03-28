import os
from typing import List

import pyopenms
import numpy as np

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
    Detect openms_featues in chromatogram in file.

    Parameters
    ----------
    filename : str
        Input chromatogram filename, accepted: mzML, mzXML, zmData.

    Returns
    -------
    list of Chromatogram
        Detected openms_featues
    """
    # Warn user that it is ms1 or change name to smth beter
    file_extension = os.path.splitext(filename)[-1].lower()
    if file_extension == ".mzml":
        return parse_ms1_mzml(filename)
    elif file_extension == ".mzxml":
        return parse_ms1_mzxml(filename)
    elif file_extension == ".mzxml":
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
    return np.mean(lengths_rt) / np.mean(widths_mz)


def features_to_weight(features):
    return get_weight(*gather_widths_lengths(features))


def openms_feature_to_feature(openms_feature, input_map, weight=None):
    """
    Gather signal from chromatogram contained in feature bounding box.

    Feature with gathered signal is a chromatograms subset, so we represent it
    using Chromatograms class.

    Algorithm scheme:
        For every OpenMS feature do:
        1. choose spectra by RT
        2. iterate over mz and check if is enclosed
    It is done noneffectively, but how to do it better?

    Parameters
    ----------
    openms_feature : pyopenms.Feature or OpenMSFeatureMimicry
        OpenMS-like object for representing feature.
    input_map : pyopenms.InputMap
        Parsed chromatogram.
    weight : float or None
        Weight by which RT should scaled. If None then RT not scaled.

    Returns
    -------
    Chromatogram
        A feature with gathered signal.
    """
    max_rt, max_mz = openms_feature.getConvexHull().getBoundingBox().maxPosition()
    min_rt, min_mz = openms_feature.getConvexHull().getBoundingBox().minPosition()

    mzs = []
    rts = []
    ints = []

    for openms_spectrum in input_map:  # indeed, it's a spectrum (single scan)
        rt = openms_spectrum.getRT()
        if min_rt <= rt <= max_rt:
            for mz, i in zip(*openms_spectrum.get_peaks()):
                if min_mz <= mz <= max_mz:
                    mzs.append(mz)
                    rts.append(rt)
                    ints.append(i)
    if len(rts) == 0:
        print("zero length", weight)
    ch = Chromatogram(rts, mzs, ints, weight)
    if weight is not None:
        ch.scale_rt()
    ch.normalize(keep_old=True)
    return ch


def openms_features_to_features(input_map, openms_features, weight=None):
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
    weight : float
        Weight by which RT should scaled. If None then RT not scaled

    Returns
    -------
    list of Chromatogram
        A list of features with gathered signal.
    """
    chromatograms = []
    # Chromatogram class is universal, so we use it to represent chromatograms
    # subsets, i.e. features.
    for f in openms_features:
        chromatograms.append(
            openms_feature_to_feature(f, input_map, weight))
    return chromatograms


def detect_features_from_file(filename, should_scale=False):
    """
    Parse and detect featues from chromatogram contained in file.

    This function parses chromatograms contained in file names `filename`,
    detects features, collects all signal contained within features and
    return features represented as chromatogram subsets.

    Parameters
    ----------
    filename : str
        input chromatogram filename
    should_scale : bool
        should feature have scaled RT?

    Returns
    -------
    list of Chromatogram
        Iterable of parsed features represented as chromatograms subsets.
    """
    input_map = parse_chromatogram_file(filename)
    openms_features = find_features(input_map)
    if should_scale:
        weight = features_to_weight(openms_features)
        print("Average length to width:", weight)
    print("Parsed file", filename, "\n", openms_features.size(),
          "OpenMS features found,\n")
    features = openms_features_to_features(
        input_map, openms_features, weight if should_scale else None)
    return features
