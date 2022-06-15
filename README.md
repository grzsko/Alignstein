Alignstein
==============================

LC-MS alignment algorithm based on Wasserstein distance.

Installation
------------
1. Obtain MassSinkhornmetry package from https://github.com/grzsko/MassSinkhornmetry and install into environment.
2. Install requirements `pip install -r requirements.txt`.
3. Install Alignstein into your environment: `python setup.py develop`.

Usage
-----
Alignstein package can be used as a Python3 library and its usage is described in the Jupyter notebook with tutorial under `tutorials` directory.

Moreover, it is possible to use Alignstein from bash.
To perform alignment of chromatograms written in files `file1.mzml`, `file2.mzml` (accepted formats: mzData, mzXML, and mzML) run:
```
$ align file1.mzml file2.mzml ...
```
More help under:
```
$ align --help


Parse mzml files and perform alignment

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

```
As a result the detected features are dumped as well as consenus features consensus.csv file.
