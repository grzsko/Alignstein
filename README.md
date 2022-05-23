Alignstein
==============================

LC-MS alignment algorithm based on Wasserstein distance.

Installation
------------
1. Obtain MassSinkhornmetry package from https://github.com/grzsko/MassSinkhornmetry and install into environment.
2. Install requirements `pip install -r requirements.txt`.
3. Install Alignstein into your environment: `python setup.py develop` (it is still under development, so be ready for continuous updating :-D)

Usage
-----
Alignstein package can be used as a Python3 library and its usage is described in the Jupyter notebook with tutorial under `tutorials` directory.

Moreover, it is possible to use Alignstein from bash.
To perform alignment of chromatograms written in files `file1.mzml`, `file2.mzml` (accepted formats: mzData, mzXML, and mzML) run:
```
$ align file1.mzml file2.mzml ...
```
More help under
```
$ align --help
```
As a result the detected features are dumped as well as consenus features consensus.csv file.
