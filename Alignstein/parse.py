import os
import xml.etree.ElementTree as ET

import numpy as np
import pyopenms

from .OpenMSMimicry import MyCollection, OpenMSFeatureMimicry
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
    # Omitting zero values as meaningless.
    return (np.mean(lengths_rt[0.0 < lengths_rt]) /
            np.mean(widths_mz[0.0 < widths_mz]))


def features_to_weight(features):
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
    It is done noneffectively, but how to do it better?

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
        print("zero length")
    ch = Chromatogram(rts, mzs, ints)
    ch.normalize(keep_old=True)
    if hasattr(openms_feature, 'ext_id'):
        # Sometimes it is useful to have some information from other software
        ch.ext_id = openms_feature.ext_id
    return ch


# def feature_from_file_features_to_chromatograms(input_map, openms_features):
#     features = []
#     for oms_f in openms_features:
#         f = openms_feature_to_feature(oms_f, input_map)
#         if not f.empty:
#             f.feature_id = [oms_f.intensity, oms_f.rt, oms_f.mz]
#             # TODO change to ext_id
#             f.cut_smallest_peaks(0.001)
#             features.append(f)
#     return features

def openms_features_to_features(input_map, openms_features):
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

    Returns
    -------
    list of Chromatogram
        A list of features with gathered signal.
    """
    chromatograms = []
    # Chromatogram class is universal, so we use it to represent chromatograms
    # subsets, i.e. features.
    for f in openms_features:
        chromatograms.append(openms_feature_to_feature(f, input_map))
    return chromatograms


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
    input_map = parse_chromatogram_file(filename)
    openms_features = find_features(input_map)
    print("Parsed file", filename, "\n", openms_features.size(),
          "OpenMS features found,\n")
    features = openms_features_to_features(input_map, openms_features)
    return features


def parse_features_from_file(filename):
    """
    Parse features from featureXML file.

    Parameters
    ----------
    filename : str
        Filename of features (.featureXML) to be parsed. Must have same order
        as chromatograms_filenames. Every feature of file should contain field
        convexhull which describes the spatial properties of features.

    Returns
    -------
    list of OpenMSMimicry
        Parsed features with API conforming pyopenms API.
    """
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
            mimicry_feature.ext_id = i
            features.append(mimicry_feature)
        else:
            print("Skipping small openms_feature, id:", feature.attrib["id"])
            continue
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
    features = parse_features_from_file(features_filename)
    print("Parsed file", filename, "\n", features.size(),
          "OpenMS features found.")
    return openms_features_to_features(input_map, features)

# def parse_chromatograms_and_features(chromatogram_filenames,
#                                      features_filenames):
#     """
#     Parse chromatograms with provided featureXML files.
#
#     Parameters
#     ----------
#     chromatogram_filenames : iterable of str
#         Filenames of chromatograms to be parsed.
#     features_filenames : iterable of str
#         Filenames of features (.featureXML) to be parsed. Must have same order
#         as chromatograms_filenames
#
#     Returns
#     -------
#     list of lists of Chromatogram
#         Parsed features with collected signal for consecutive chromatogram
#         files.
#     """
#     feature_sets_list = []
#     for chromatogram_fname, features_fname in zip(chromatogram_filenames,
#                                                   features_filenames):
#         feature_sets_list.append(
#             parse_chromatogram_with_detected_features(
#                 chromatogram_fname, features_fname))
#     return feature_sets_list
