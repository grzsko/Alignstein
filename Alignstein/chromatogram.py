import numpy as np


def custom_scale(rts, constant, other_constant=4):
    return rts / float(constant) / other_constant


def id_scale(rts, *other):
    return rts


def constant_scale(rts, constant):
    return rts / float(constant)


# def custom_scale(rts, constants):
#     # TODO
#     pass

def RT_scale(*args, **kwargs):
    #     return constant_scale(*args, **kwargs)
    return custom_scale(*args, **kwargs)


class Chromatogram:
    # Assuming that MS1 and MS2 are centroided
    def __init__(self, rts, mzs, ints, weight=1, extra_weight=1):
        self.empty = len(rts) == 0
        self.rts = np.array(rts)
        self.mzs = np.array(mzs)
        self.ints = np.array(ints)
        #         self.tic = np.sum(ints)
        self.weight = weight * extra_weight
        self.mid = np.array([np.mean(self.rts), np.mean(self.mzs)])
        self.hull = None

    def normalize(self, target_value=1.0):
        """
        Normalize the intensity values so that they sum up to the target value.
        """
        if self.tic != target_value:
            self.ints = target_value / self.tic * self.ints

    #             self.tic = np.sum(self.ints)

    def scale_rt(self):
        self.rts = RT_scale(self.rts, self.weight)
        self.mid = np.array([np.mean(self.rts), np.mean(self.mzs)])
        self.hull = None

    def plot(self, ax):
        ax.scatter(self.rts, self.mzs, s=20, c=self.ints, cmap='RdBu')

    @property
    def confs(self):
        return list(zip(zip(self.rts, self.mzs), self.ints))

    #     @property
    #     def centroid(self):
    #         return (np.mean(self.rts), np.mean(self.mzs))
    # TODO, remove it, there is a mid!
    @property
    def tic(self):
        return np.sum(self.ints)

    def __len__(self):
        return len(self.rts)

    @staticmethod
    def sum_chromatograms(chromatograms_iterable):
        total_length = np.sum([len(ch) for ch in chromatograms_iterable])
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

    def in_hull(self, p):
        """
        Test if points in `p` are in convex hull of chromatogram points
        """

        if self.hull is None:
            self.hull = Delaunay(np.c_[self.rts, self.mzs])

        return np.any(self.hull.find_simplex(p) >= 0)
