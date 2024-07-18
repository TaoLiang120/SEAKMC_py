import numpy as np
import copy
import scipy.linalg

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


class DynMat:
    def __init__(
            self,
            id,
            natoms,
            dynmat,
            eig=None,
            eigvec=None,
            ieigvec=None,
            sqrteig=None,
            vib=None,
            vibcut=1.0e-8,
            isValid=True,
            isSNCable=True,
            delimiter=" ",
            LowerHalfMat=False,
    ):
        self.id = id
        self.natoms = natoms
        self.dynmat = dynmat
        self.eig = eig
        self.eigvec = eigvec
        self.ieigvec = ieigvec
        self.sqrteig = sqrteig
        self.vib = vib
        self.vibcut = vibcut
        self.isValid = isValid
        self.isSNCable = isSNCable
        self.delimiter = delimiter
        self.LowerHalfMat = LowerHalfMat

    def __str__(self):
        return "Dynmat matrix id is ({})".format(self.id)

    def __repr__(self):
        return self.__str__()

    def copy(self, deep=True):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def deepcopy(self):
        return self.copy(deep=True)

    @classmethod
    def from_file(cls, filename, id=0, delimiter=None, vibcut=1.0e-8, LowerHalfMat=False):
        dynmat = np.loadtxt(filename, delimiter=delimiter)
        dynmat = dynmat.flatten()

        natoms = int(np.sqrt(dynmat.shape[0]) / 3)
        if dynmat.shape[0] - 9 * natoms * natoms != 0:
            raise ValueError("The file must contain 3N * 3N elements, where N is number of atoms!")

        dynmat = dynmat.reshape([natoms * 3, natoms * 3])
        dm = cls(id, natoms, dynmat, eig=None, eigvec=None, vib=None, vibcut=vibcut, isValid=True,
                 delimiter=delimiter, LowerHalfMat=LowerHalfMat)
        return dm

    def to_hessian(self, data, settings):
        itypes = data.atoms['type'].to_numpy().astype(int) - 1
        itypes = itypes[0:self.natoms]
        itypes = (np.vstack((itypes, itypes, itypes))).flatten()
        masses = settings.potential['masses']
        imasses = masses[itypes]
        ijmasses = np.zeros((imasses.shape[0], imasses.shape[0]), dtype=float)
        for i in range(imasses.shape[0]):
            for j in range(imasses.shape[0]):
                ijmasses[i][j] = np.sqrt(imasses[i] * imasses[j])
        self.dynmat = np.multiply(self.dynmat, ijmasses)

    def diagonize_matrix(self):
        if self.LowerHalfMat:
            lower = 1
        else:
            lower = 0

        self.eig, self.eigvec, info = scipy.linalg.lapack.dsyev(self.dynmat, compute_v=1, lower=lower)
        self.dynmat = None

        eigcomplex = self.eig[np.iscomplex(self.eig)]
        if len(eigcomplex) > 0:
            self.isValid = False
            self.isSNCable = False
            self.eig = None
            self.eigvec = None
            self.dynmat = None
        if self.isValid: self.eigvec = None

    def negative_to_one(self):
        if not isinstance(self.eig, np.ndarray): self.diagonize_matrix()
        if self.isValid:
            vib = np.select([self.eig < self.vibcut, self.eig >= self.vibcut], [1.0, self.eig])
        else:
            vib = np.ones(self.natoms)

        return vib

    def is_SNCable(self):
        if not isinstance(self.eig, np.ndarray): self.diagonize_matrix()

        if self.isValid:
            eignegative = np.compress(self.eig <= 0.0, self.eig, axis=0)
            if eignegative.shape[0] > 0: self.isSNCable = False

    def sqrt_eig(self):
        self.sqrteig = np.sqrt(self.eig)

    def set_vib(self):
        if not isinstance(self.vib, np.ndarray):
            self.vib = self.negative_to_one()

    def get_inv_luf_eigvec(self):
        lu, piv, info = scipy.linalg.lapack.dgetrf(self.eigvec)
        self.ieigvec, info = scipy.linalg.lapack.dgetri(lu, piv)


class VibMat(DynMat):
    def __init__(
            self,
            id,
            natoms,
            dynmat,
            eig=None,
            eigvec=None,
            ieigvec=None,
            sqrteig=None,
            vib=None,
            nfixed=0,
            eigA=None,
            eigvecA=None,
            vibA=None,
            eigF=None,
            eigvecF=None,
            vibF=None,
            vibcut=1.0e-5,
            isValid=True,
            isSNCable=True,
            delimiter=" ",
            LowerHalfMat=False,
    ):
        super().__init__(
            id,
            natoms,
            dynmat,
            eig=eig,
            eigvec=eigvec,
            ieigvec=ieigvec,
            sqrteig=sqrteig,
            vib=vib,
            vibcut=vibcut,
            isValid=isValid,
            isSNCable=isSNCable,
            delimiter=delimiter,
            LowerHalfMat=LowerHalfMat,
        )
        self.nfixed = nfixed
        self.nactive = self.natoms - self.nfixed
        self.eigA = eigA
        self.eigvecA = eigvecA
        self.vibA = vibA
        self.eigF = eigF
        self.eigvecF = eigvecF
        self.vibF = vibF

    @classmethod
    def from_dynmat(cls, dm, nfixed):
        return cls(dm.id, dm.natoms, dm.dynmat, eig=dm.eig, eigvec=dm.eigvec, ieigvec=dm.ieigvec,
                   sqrteig=dm.sqrteig, vib=dm.vib, nfixed=nfixed, vibcut=dm.vibcut, isValid=dm.isValid,
                   isSNCable=dm.isSNCable, delimiter=dm.delimiter, LowerHalfMat=dm.LowerHalfMat)

    def split_vibmat(self):
        if not isinstance(self.eigA, np.ndarray):
            if not isinstance(self.eig, np.ndarray): self.diagonize_matrix()
            self.eigA = self.eig[0:self.nactive]
            self.eigF = self.eig[self.nactive:self.natoms]
            self.eigvecA = self.eigvec[0:self.nactive]
            self.eigvecF = self.eigvec[self.nactive:self.natoms]
        if not isinstance(self.vib, np.ndarray):
            self.set_vib()
            self.vibA = self.vib[0:self.nactive]
            self.vibF = self.vib[self.nactive:self.natoms]


class VibMats:
    def __init__(
            self,
            idav,
            groundvib,
            spvib,
            method="harmonic",
    ):
        self.idav = idav
        self.groundvib = groundvib
        self.spvib = spvib
        self.method = method
        self.UC4prefactor = 9648.5

    def get_prefactor(self):
        if self.method == "harmonic":
            self.groundvib.set_vib()
            self.spvib.set_vib()
            thispre = np.sum(np.log(self.groundvib.vib) - np.log(self.spvib.vib))
            thispre = np.exp(thispre)
            thispre = np.sqrt(self.UC4prefactor * thispre)
            #thispre = np.sqrt(Constants["UC4prefactor"]*thispres[-1])/(2.0*np.pi)
            return thispre
