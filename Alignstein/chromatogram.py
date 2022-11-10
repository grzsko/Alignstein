import numpy as np
from scipy.spatial import Delaunay


def custom_scale(rts, constant, other_constant=4):
    # return rts / float(constant) / other_constant
    # Exemplary custom scale
    return np.log(rts + 1)


def id_scale(rts, *other):
    return rts


def constant_scale(rts, constant, other_constant=1):
    return rts / float(constant) / float(other_constant)


def RT_scale(*args, **kwargs):
    return constant_scale(*args, **kwargs)
    # return custom_scale(*args, **kwargs)


class Chromatogram:
    """Class representing chromatogram or chromatogram subset (i.e. feature)."""

    # Assuming that MS1 and MS2 are centroided
    def __init__(self, rts, mzs, ints, monoisotopic_mass=None):
        self.empty = len(rts) == 0
        self.rts = np.array(rts)
        self.mzs = np.array(mzs)
        self.ints = np.array(ints)
        self._monoisotopic_mass = monoisotopic_mass
        #         self.tic = np.sum(ints)
        self._weight = None
        self._mid = None
        self.hull = None

    def normalize(self, target_value=1.0, keep_old=False):
        """
        Normalize the intensity values so that they sum up to the target value.
        """
        if not self.empty and self.tic != target_value:
            if keep_old:
                self.nonnormalized_ints = np.copy(self.ints)
            self.ints = target_value / self.tic * self.ints

    def scale_rt(self, weight):
        self.rts = RT_scale(self.rts, weight)
        self._weight = weight
        self._mid = None
        self.hull = None

    def plot(self, ax):
        ax.scatter(self.rts, self.mzs, s=20, c=self.ints, cmap='RdBu')

    @property
    def monoisotopic_mass(self):
        if self._monoisotopic_mass is None:
            self._monoisotopic_mass = np.percentile(self.mzs, 10)
        return self._monoisotopic_mass

    @property
    def confs(self):
        return list(zip(zip(self.rts, self.mzs), self.ints))

    @property
    def tic(self):
        return np.sum(self.ints)

    @property
    def mid(self):
        """
        Calculate chromatogram (subset) centroid.

        Returns
        -------
        tuple of floats
            Centroid of chromatogram subset.
        """
        if self.empty:
            return None
        if self._mid is None:
            self._mid = np.array([np.mean(self.rts), np.mean(self.mzs)])
        return self._mid

    def __len__(self):
        return len(self.rts)

    @staticmethod
    def sum_chromatograms(chromatograms_iterable):
        total_length = np.sum([len(ch) for ch in chromatograms_iterable])
        if total_length == 0:
            return Chromatogram([], [], [])  # empty chromatogram
        rts = np.empty(total_length, dtype=np.float)
        mzs = np.empty(total_length, dtype=np.float)
        ints = np.empty(total_length, dtype=np.float)
        begin = 0
        for chromatogram in chromatograms_iterable:
            end = begin + len(chromatogram)
            rts[begin:end] = chromatogram.rts
            mzs[begin:end] = chromatogram.mzs
            ints[begin:end] = chromatogram.ints
            begin = end
        return Chromatogram(rts, mzs, ints)

    def cut_smallest_peaks(self, removed_intensity=0.01):
        sorted_ints = np.sort(self.ints)
        cumsums = np.cumsum(sorted_ints) / self.tic
        first_index = np.argmax(cumsums >= removed_intensity)
        threshold_intensity = sorted_ints[first_index]
        where_to_cut = self.ints >= threshold_intensity
        self.ints = self.ints[where_to_cut]
        self.mzs = self.mzs[where_to_cut]
        self.rts = self.rts[where_to_cut]
        self.normalize()
        self.empty = len(rts) == 0

    def any_in_hull(self, points):
        """
        Test if points in `points` are in convex hull of chromatogram points.
        """

        if self.hull is None:
            self.hull = Delaunay(np.c_[self.rts, self.mzs])

        return np.any(self.hull.find_simplex(points) >= 0)

    def get_bounding_box(self):
        """
        Get coordinates defining chromatogram set bounding box.

        Returns
        -------
        tuple
            Max. RT, max. M/Z, min. RT, min. MZ
        """
        if not self.empty:
            return (np.max(self.rts), np.max(self.mzs),
                    np.min(self.rts), np.min(self.mzs))
        else:
            return 0, 0, 0, 0
