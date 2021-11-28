from scipy.spatial import ConvexHull

class MyCollection(list):
    def size(self):
        return self.__len__()


class OpenMSFeatureMimicry:
    class MyConvexHull:
        class BoundingBox:
            def __init__(self, max_position, min_position):
                self.max_position = max_position
                self.min_position = min_position

            def maxPosition(self):
                return self.max_position

            def minPosition(self):
                return self.min_position

        def __init__(self, scipy_ch):
            self.scipy_ch = scipy_ch
            vertices = self.scipy_ch.points[self.scipy_ch.vertices]
            max_p = vertices.max(axis=0)
            min_p = vertices.min(axis=0)
            self.bb = self.BoundingBox(max_p, min_p)

        def getBoundingBox(self):
            return self.bb

    def __init__(self, points):
        self.points = points
        self.ch = self.MyConvexHull(ConvexHull(points))

    def getConvexHull(self):
        return self.ch
