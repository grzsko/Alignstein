Alignstein
==============================

LC-MS alignment algorithm based on Wasserstein distance.

## Installation
1. Obtain MassSinkhornmetry package from https://github.com/grzsko/MassSinkhornmetry and install into environment.
2. Install requirements `pip install -r requirements.txt`.
3. Install Alignstein into your environment: `python setup.py develop`.

### Installation instructions for Windows users
Alignstein is primarly designed for Linux or macOS. For using Alignstein on Windows (or do not want to hussle with installation), the easiest way is to pull a [grzsko/alignstein](https://hub.docker.com/r/grzsko/alignstein) Docker image or build it on your own from a [Dockerfile](docker/Dockerfile).

## Usage
Alignstein package can be used as a Python3 library and its usage is described in the Jupyter notebook with tutorial under `tutorials` directory.

Moreover, it is possible to use Alignstein from bash.
To perform alignment of chromatograms written in files `file1.mzml`, `file2.mzml` (accepted formats: mzData, mzXML, and mzML) run:
```
$ align file1.mzml file2.mzml ...
```
As a result the detected features are dumped as well as consenus features consensus.csv file.

More help under:
```
$ align --help
Determination of memory status is not supported on this
 platform, measuring for memoryleaks will never fail
Parse mzml files and perform alignment

Usage: align.py -h
       align.py [-c SCALING_CONST] [-t MIDS_THRSH] [-m MIDS_UP_BOUND]
                [-i MONOISO_THRSH] [-w GWD_UP_BOUND] [-p PENALTY] [-s]
                [-n WORKERS_NUMB] [-o OUT_FILENAME] [-f FEATURE_FILE]...
                MZML_FILE...

Arguments:
    MZML_FILE      names of files with chromatograms to be aligned

Options:
    -f FEATURE_FILE   names of files with detected features in chromatograms,
                      order of filenames should conform order of input data
                      files. POSIX compliance requires to every features
                      filename be preceded by -f. Detected features are dumped
                      chromatogram directory with additional featureXML
                      extention.
    -o OUT_FILENAME   Output consensus filename [default: consensus.out]
    -c SCALING_CONST  Additional constant by which RT should be scaled.
                      [default: 1]
    -t MIDS_THRSH     Distance threshold between centroid in one cluster. Not
                      applicable when aligning two chromatograms. [default: 1.5]
    -m MIDS_UP_BOUND  Maximum cetroid distance between which GWD will be computed.
                      For efficiency reasons should be reasonably small.
                      [default: 10]
    -i MONOISO_THRSH  Maximum ppm difference of feature monoisotopic massess for
                      which GWD will be computed. [default: 20]
    -w GWD_UP_BOUND   Cost of not transporting a part of signal, aka the
                      lambda parameter. Can be interpreted as maximal distance
                      over which signal is transported while computing GWD.
                      [default: 10]
    -p PENALTY        penalty for feature not matching. [default: 10]
    -s                Should be only indices of features be dumped?
                      [default: False]
    -n WORKERS_NUMB   max number of processes used for parallelization. For
                      multialignment it should not be larger than 16
                      [default: 16]
```

## Cite us!
If you use this tool or find this work interesting, do not forget to cite our paper:
> Grzegorz Skoraczyński, Anna Gambin, Błażej Miasojedow, Alignstein: Optimal transport for improved LC-MS retention time alignment, *GigaScience*, Volume 11, 2022, giac101, https://doi.org/10.1093/gigascience/giac101
