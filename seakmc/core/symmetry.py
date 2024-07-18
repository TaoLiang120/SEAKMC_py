import copy
from seakmc.core.util import mat_mag

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


class SymmOP:
    def __init__(self, rotation_matrix, translation_vector):
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector

    def __str__(self):
        thisstr = "Rotation matrix are ({}).".format(self.rotation_matrix)
        thisstr += "\n" + "Translation vector are ({}).".format(self.translation_vector)
        return thisstr

    def __repr__(self):
        return self.__str__()


class PGSymmOps:
    def __init__(self, id, OPs, sch_symbol="C1", tol=0.1):
        self.id = id
        self.OPs = OPs
        self.nOP = len(self.OPs)
        self.sch_symbol = sch_symbol
        self.tol = tol

    def __str__(self):
        return "Point Group Symmetry Operations id ({}).".format(self.id)

    def __repr__(self):
        return self.__str__()

    def copy(self, deep=True):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def deepcopy(self):
        return self.copy(deep=True)

    def validate_OPs(self):
        todel = []
        for i in range(self.nOP):
            if mat_mag(self.OPs[i].translation_vector) > self.tol:
                todel.append(i)
        for i in sorted(todel, reverse=True):
            del self.OPs[i]
        self.nOP = len(self.OPs)

        todel = []
        for i in range(self.nOP):
            for j in range(i + 1, self.nOP):
                if mat_mag(self.OPs[i].rotation_matrix - self.OPs[j].rotation_matrix) < self.tol:
                    todel.append(j)
        todel = list(set(todel))
        for i in sorted(todel, reverse=True):
            del self.OPs[i]
        self.nOP = len(self.OPs)


class SGSymmOps:
    def __init__(self, id, SGtup, tol=0.1):
        self.id = id
        self.SGtup = SGtup
        self.symbol = self.SGtup[0]
        self.number = self.SGtup[1]
        self.OPs = self.SGtup[2]
        self.nOP = len(self.OPs)
        self.tol = tol

    def __str__(self):
        return f"Space Group Symmetry symbol {self.symbol} and number is {self.number}."

    def __repr__(self):
        return self.__str__()

    def copy(self, deep=True):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def deepcopy(self):
        return self.copy(deep=True)

    def validate_OPs(self):
        todel = []
        for i in range(self.nOP):
            for j in range(i + 1, self.nOP):
                if mat_mag(self.OPs[i].rotation_matrix - self.OPs[j].rotation_matrix) < self.tol \
                        and mat_mag(self.OPs[i].translation_vector - self.OPs[j].translation_vector) < self.tol:
                    todel.append(j)
        todel = list(set(todel))
        for i in sorted(todel, reverse=True):
            del self.OPs[i]
        self.nOP = len(self.OPs)
