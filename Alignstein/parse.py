import os

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
    file_extension = os.path.splitext(filename)[-1].lower()
    if file_extension == "mzml":
        return parse_ms1_mzml(filename)
    elif file_extension == "mzxml":
        return parse_ms1_mzxml(filename)
    else:
        return parse_ms1_mzdata(filename)
    # TODO refactor to lesser code repetitions


def gather_widths_lengths(openms_featues):
    lengths_rt = []
    widths_mz = []
    for f in openms_featues:
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


def features_to_weight(openms_features):
    return get_weight_from_widths_lengths(
        *gather_widths_lengths(openms_features))


def openms_feature_to_chromatogram_subset(openms_feature, input_map, weight):
    """
    Gather signal from chromatogram contained in feature bounding box.

    Feature with gathered signal is a chromaotgrams subset so we represent it
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
    weight : float
        Weight by which RT should scaled.

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

    for open_spectrum in input_map:
        rt = open_spectrum.getRT()
        if min_rt <= rt <= max_rt:
            for mz, i in zip(*open_spectrum.get_peaks()):
                if min_mz <= mz <= max_mz:
                    mzs.append(mz)
                    rts.append(rt)
                    ints.append(i)
    if len(rts) == 0:
        print("zero length", weight)
    ch = Chromatogram(rts, mzs, ints, weight)
    ch.scale_rt()
    ch.normalize()
    return ch


def openms_features_to_chromatogram_subsets(input_map, openms_features, weight):
    """
    Gather signal over all OpenMS-like features.

    We represent features with gathered signal as a subset of chromatogram so
    we use Chromatogram class.

    Parameters
    ----------
    openms_features : pyopenms.FeatureMap or MyCollection
        Iterable with OpenMS-like objects representing features.
    input_map : pyopenms.InputMap
        Parsed chromatogram.
    weight : float
        Weight by which RT should scaled.

    Returns
    -------
    list of Chromatogram
        A list of features with gathered signal.
    """
    chromatograms = []
    # Chromatogram class is universal, so we use it to represent chromatograms
    # subsets, i.e. openms_features.
    for f in openms_features:
        chromatograms.append(
            openms_feature_to_chromatogram_subset(f, input_map, weight))
    return chromatograms


def detect_features_from_file(filename):
    """
    Parse and detect featues from chromatogram contained in file.

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
    input_map = parse_chromatogram_file(filename)
    features = find_features(input_map)
    weight = features_to_weight(features)
    print("Parsed file", filename, "\n", features.size(),
          "openms_featues found,\nAverage lenght to width:", weight)
    return openms_features_to_chromatogram_subsets(input_map, features, weight)