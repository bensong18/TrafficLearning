#! /usr/bin/env python
#-*- coding: utf-8 -*-

from math import sqrt
from abc import ABCMeta, abstractmethod
from error_wrongvec import WrongVecError
import numpy as np
import numpy.linalg as linalg
from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf

class Distance():
    """
      abstract class, represent distance of two vector

      Attributes:
      """

    __metaclass__ = ABCMeta

    @abstractmethod
    def distance(self, vec1, vec2):
        """
        Compute distance of two vector(one line numpy array)
        if you use scipy to store the sparse matrix, please use s.getrow(line_num).toarray() build the one dimensional array

        Args:
            vec1: the first line vector, an instance of array
            vec2: the second line vector, an instance of array

        Returns:
            the computed distance

        Raises:
            TypeError: if vec1 or vec2 is not numpy.ndarray and one line array
        """
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
            raise TypeError("type of vec1 or vec2 is not numpy.ndarray")
        if vec1.ndim is not 1 or vec2.ndim is not 1:
            raise WrongVecError("vec1 or vec2 is not one line array")
        if vec1.size != vec2.size:
            raise WrongVecError("vec1 or vec2 is not same size")
        pass
# end Distance

class DTWDistance(Distance):
    # def _traceback(D):
    #     i, j = array(D.shape) - 2
    #     p, q = [i], [j]
    #     while (i > 0) or (j > 0):
    #         tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
    #         if tb == 0:
    #             i -= 1
    #             j -= 1
    #         elif tb == 1:
    #             i -= 1
    #         else:  # (tb == 2):
    #             j -= 1
    #         p.insert(0, i)
    #         q.insert(0, j)
    #     return array(p), array(q)
    w = inf
    warp = 1
    s = 1.0
    def distance(self,vec1,vec2):
        """
        Computes Dynamic Time Warping (DTW) of two sequences.

        :param array x: N1*M array
        :param array y: N2*M array
        :param func dist: distance used as cost measure
        :param int warp: how many shifts are computed.
        :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
        :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
        Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
        """
        # w = inf
        warp = 1
        s = 1.0
        dist=lambda x, y: np.abs(x - y)
        # assert len(vec1)
        # assert len(vec2)
        # warp = 1, w = inf, s = 1.0
        # w = inf
        # assert isinf(w) or (w >= abs(len(vec1) - len(vec2)))
        # assert s == 1.0
        # warp = 1
        # w = inf
        # s = 1.0
        r, c = len(vec1), len(vec2)
        # if not isinf(w):
        #     D0 = full((r + 1, c + 1), inf)
        #     for i in range(1, r + 1):
        #         D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        #     D0[0, 0] = 0
        # else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
        D1 = D0[1:, 1:]  # view
        for i in range(r):
            for j in range(c):
                D1[i, j] = dist(vec1[i],vec2[j])
        C = D1.copy()
        jrange = range(c)
        for i in range(r):
            for j in jrange:
                min_list = [D0[i, j]]
                for k in range(1, warp + 1):
                    i_k = min(i + k, r)
                    j_k = min(j + k, c)
                    min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
                D1[i, j] += min(min_list)
        # print(D1)
        # if len(vec1) == 1:
        #     path = zeros(len(vec2)), range(len(vec2))
        # elif len(vec2) == 1:
        #     path = range(len(vec1)), zeros(len(vec1))
        return D1[-1,-1]
        # else:
            # path = _traceback(D0)
        # return D1[-1, -1], C, D1, path


class ConsineDistance(Distance):
    """
    consine distance
    a sub class of Distance
    """

    def distance(self, vec1, vec2):
        """
        Compute distance of two vector by consine distance
        """
        super(ConsineDistance, self).distance(vec1, vec2)  # super method
        num = np.dot(vec1, vec2)
        denom = linalg.norm(vec1) * linalg.norm(vec2)
        if num == 0:
            return 1
        return - num / denom
# end ConsineDistance
class EucDistance(Distance):
    """
    consine distance
    a sub class of Distance
    """

    def distance(self, vec1, vec2):
        """
        Compute distance of two vector by consine distance
        """
        super(EucDistance, self).distance(vec1, vec2)  # super method
        return np.sqrt(sum((vec1[72:264]-vec2[72:264])**2))
        # return np.linalg.norm(vec1 - vec2)
        # num = np.dot(vec1, vec2)
        # denom = linalg.norm(vec1) * linalg.norm(vec2)
        # if num == 0:
        #     return 1
        # return - num / denom
