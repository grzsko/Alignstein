Alignstein
==============================

LC/GC-MS alignment algorithm based on Wasserstein distance

Installation
------------
1. Obtain MS_scaling from https://github.com/grzsko/MS_scaling and install into environment.
2. Run `pip install -U -r requirements.txt`

Usage
-----
To perform alignment of chromatograms written in files `file1.mzml`, `file2.mzml` run:
```
python3 align.py file1.mzml file2.mzml ...
```
More help under
```
python3 align.py --help
```
