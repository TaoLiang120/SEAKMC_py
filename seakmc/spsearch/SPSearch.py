import os, warnings
import shutil
import numpy as np
import pandas as pd
import math
import copy
from numpy import pi
import scipy.linalg

from seakmc.core.util import mats_angle, mats_sum_mul, mat_mag, mat_unit, sigmoid_function
from seakmc.dynmat.Dynmat import DynMat, VibMat, VibMats

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


class SPSearch:
    def __init__(
            self,
            idav,
            idsps,
            data,
            sett,
            thiscolor,
            force_evaluator,
            SNC=False,
            dmAV=None,
            pre_disps=[],
            apply_mass=False,
            ikmc=0,
            DynMatOutpath="DynMatOut",
    ):
        self.idav = idav
        self.idsps = idsps
        self.data = data
        self.sett = sett
        self.thiscolor = thiscolor
        self.force_evaluator = force_evaluator
        self.nactive = self.data.nactive
        self.SNC = SNC
        self.dmAV = dmAV
        if SNC:
            if isinstance(self.dmAV, DynMat) or isinstance(self.dmAV, VibMat):
                pass
            else:
                raise ValueError("SPSearch must have DynMat object of the active volume.")
        self.pre_disps = np.array(pre_disps)
        self.apply_mass = apply_mass
        self.ikmc = ikmc
        self.DynMatOutpath = DynMatOutpath

        self.PREF = self.sett.saddle_point["Prefactor"]
        self.ECONV = self.sett.spsearch["EnConv"]
        self.FCONV = self.sett.spsearch["FConv"]
        self.DMAGCUT = self.sett.saddle_point["DmagCut"]
        if isinstance(self.DMAGCUT, str): self.DMAGCUT = self.sett.saddle_point["DAtomCut"] * self.nactive
        self.EBIASCUT = self.sett.saddle_point["EbiasCut"]
        if isinstance(self.EBIASCUT, str): self.EBIASCUT = self.sett.saddle_point["BarrierCut"]
        self.FDMAGCUT = self.sett.saddle_point["DmagCut_FI"]
        if isinstance(self.FDMAGCUT, str): self.FDMAGCUT = self.DMAGCUT
        self.Tol4Connect = self.sett.spsearch["Tol4Connect"]

        self.ALPHA = 1.0
        self.BETA = 0.0

        self.DI_CON = False
        self.DI_MAX = False
        self.FO_CON = False
        self.ST_CON = False
        self.EN_CON = False

        self.ENINIT = 0.0
        self.ENCALC = 0.0
        self.ENLAST = 0.0
        self.EDIFF_MAX = 0.0
        self.EDIFF = 0.0

        self.XDISP = None
        self.DMAT = np.zeros((3, 3), dtype=float)
        self.DVEC = np.zeros(3, dtype=float)
        self.DMAG = 0.0
        self.BARR = 0.0
        self.FXDISP = None
        self.FDMAG = 0.0
        self.EBIAS = 0.0
        self.ISCONNECT = True
        self.ISVALID = True

        self.get_TYPES()
        if self.sett.spsearch["FixTypes_dict"] is None:
            self.FixType = False
            self.Scaling4Types = np.ones([3, self.nactive], dtype=float)
        else:
            self.FixType = True
            self.get_scaling_fixtypes()

        if self.apply_mass:
            self.Scaling4Masses = self.get_scaling_masses()
        else:
            self.Scaling4Masses = np.ones([3, self.nactive], dtype=float)

        self.DMAX_CONN = 0.0
        self.iter = 0

        self.XCR = None

    def __str__(self):
        return "SPSearch is ({})".format(self.data.__str__())

    def __repr__(self):
        return self.__str__()

    def get_TYPES(self):
        if self.data.cusatoms is None:
            types = self.data.atoms['type'].to_numpy().astype(int)
        else:
            types = self.data.cusatoms['type'].to_numpy().astype(int)
        if self.sett.spsearch["OutFix4IterationResults"]:
            self.TYPES_ALL = types[0:len(types)].astype(int)
        else:
            self.TYPES_ALL = None
        self.TYPES = types[0:self.nactive].astype(int)

    def get_scaling_fixtypes(self):
        self.Scaling4Types = np.ones([3, self.nactive], dtype=float)
        for fixtype in self.sett.spsearch["FixTypes_dict"]:
            for i in range(3):
                if i in self.sett.spsearch["FixTypes_dict"][fixtype]:
                    self.Scaling4Types[i] = np.select([self.TYPES == fixtype, self.TYPES != fixtype],
                                                      [0.0, self.Scaling4Types[i]])

    def get_scaling_masses(self):
        masses = copy.deepcopy(self.sett.potential["masses"])
        masses = np.array(masses)
        mean = np.mean(masses)
        Scaling4Masses = masses[self.TYPES - 1]
        Scaling4Masses = np.vstack((Scaling4Masses, Scaling4Masses, Scaling4Masses))
        Scaling4Masses = mean / Scaling4Masses
        return Scaling4Masses

    def convert_coord_forward(self, coords):
        nactive = coords.shape[1]
        y = scipy.linalg.blas.dgemv(self.ALPHA, self.dmAV.ieigvec, coords.T.flatten(), beta=self.BETA)
        y = np.multiply(self.dmAV.sqrteig, y)
        return (y.reshape([nactive, 3])).T

    def convert_coord_backward(self, coords):
        nactive = coords.shape[1]
        TCC1 = np.divide(coords.T.flatten(), self.dmAV.sqrteig)
        TCC2 = scipy.linalg.blas.dgemv(self.ALPHA, self.dmAV.eigvec, TCC1, beta=self.BETA)
        return (TCC2.reshape([nactive, 3])).T

    def convert_forces(self, forces):
        TCC1 = forces.flatten()
        TCC2 = scipy.linalg.blas.dgemv(self.ALPHA, self.dmAV.eigvec.T, TCC1, beta=self.BETA)
        return np.divide(TCC2, self.dmAV.sqrteig)

    def generate_sigmoid_scale(self, x, range=10, r4zero=0.3, MinSpan=2.0):
        xmax = np.max(x)
        xmin = np.min(x)
        n = x.shape[0]
        isRescale = True

        if isinstance(MinSpan, float) or isinstance(MinSpan, int):
            if (xmax - xmin) < MinSpan: isRescale = False
        if isRescale:
            s = (xmax - xmin) / range
            if isinstance(r4zero, str):
                if r4zero.upper() == "MEAN":
                    s0 = np.mean(x)
                elif r4zero.upper() == "MEDIAN":
                    s0 = np.median(x)
                else:
                    s0 = xmin + (xmax - xmin) * 0.3
            elif isinstance(r4zero, float):
                s0 = xmin + (xmax - xmin) * r4zero
            else:
                s0 = xmin + (xmax - xmin) * 0.3

            x = (x - s0) / s
            x = sigmoid_function(x)
            x = x.reshape([1, n])
        else:
            x = np.ones(n)
            x = x.reshape([1, n])
        return x

    def generate_step_scale(self, x, r4zero=0.65, MinSpan=2.0):
        xmax = np.max(x)
        xmin = np.min(x)
        n = x.shape[0]
        isRescale = True
        if isinstance(MinSpan, float) or isinstance(MinSpan, int):
            if xmax - xmin < MinSpan: isRescale = False
        if isRescale:
            if isinstance(r4zero, str):
                s0 = np.mean(x)
            else:
                s0 = xmin + (xmax - xmin) * r4zero
            x = np.select([(x - s0) < 0, (x - s0) >= 0], [0.0, 1.0])
            x = x.reshape([1, n])
        else:
            x = np.ones(n)
            x = x.reshape([1, n])
        return x

    def center_array(self, x):
        s = np.sum(x, axis=1) / x.shape[1]
        return x - s.reshape([3, 1])

    def get_AtomSquence(self, xyz, sort_by="DXYZ", AbsVal=True):
        n = xyz.shape[1]
        x = xyz[0, 0:n]
        y = xyz[1, 0:n]
        z = xyz[2, 0:n]
        if sort_by.upper() == "DXYZ":
            thisx = x * x + y * y + z * z
        elif sort_by.upper() == "DXY":
            thisx = x * x + y * y
        elif sort_by.upper() == "DXZ":
            thisx = x * x + z * z
        elif sort_by.upper() == "DYZ":
            thisx = y * y + z * z
        elif sort_by.upper() == "X":
            if AbsVal:
                thisx = np.absolute(x)
            else:
                thisx = x
        elif sort_by.upper() == "Y":
            if AbsVal:
                thisx = np.absolute(y)
            else:
                thisx = y
        elif sort_by.upper() == "Z":
            if AbsVal:
                thisx = np.absolute(z)
            else:
                thisx = z
        thisx = np.argsort(thisx)
        return np.argsort(thisx)

    def generate_VN(self):
        if self.SNC:
            VN = np.random.rand(3 * self.nactive) - 0.5
            isel = np.random.randint(3, high=3 * self.nactive)
            VN += self.dmAV.eigvec.T[isel]
            VN = VN.reshape((3, self.nactive))
        else:
            VN = np.random.rand(3, self.nactive) - 0.5
        return VN

    def Check_Angle(self, thisSPS):
        if self.DMAG > self.DMAGCUT:
            self.DI_CON = True
            self.DI_MAX = True
        else:
            for i in range(thisSPS.nSP):
                if thisSPS.SPlist[i].disp.shape[1] - self.XDISP.shape[1] > 0:
                    angle = mats_angle(self.XDISP, thisSPS.SPlist[i].disp[:, 0:self.XDISP.shape[1]], Flatten=True)
                elif thisSPS.SPlist[i].disp.shape[1] - self.XDISP.shape[1] < 0:
                    angle = mats_angle(self.XDISP[:, 0:thisSPS.SPlist[i].disp.shape[1]], thisSPS.SPlist[i].disp,
                                       Flatten=True)
                else:
                    angle = mats_angle(self.XDISP, thisSPS.SPlist[i].disp, Flatten=True)
                if angle < self.sett.spsearch["AngCut"]:
                    self.DI_CON = True
                    self.DI_MAX = True
                    break

    def reprepare_runner(self):
        self.force_evaluator.close()
        thisdata = copy.deepcopy(self.data)
        thisdata.to_saddle_point(self.XCR)
        [total_energy, coords, isValid, errormsg] = self.force_evaluator.init_spsearch_runner(thisdata, self.thiscolor,
                                                                                              self.nactive)
        if not isValid:
            print(errormsg)
            quit()
        thisdata = None

    def saddlepoint_check(self, thisSPS):
        isValid = self.ISVALID
        if isValid:
            for i in range(thisSPS.nSP):
                if abs(self.BARR - thisSPS.SPlist[i].barrier) < self.sett.saddle_point["ValidSPs"]["EnTol4AVSP"]:
                    dmax = np.max(np.absolute(self.XDISP - thisSPS.SPlist[i].disp))
                    if dmax < self.sett.saddle_point["ValidSPs"]["Tol4AVSP"]:
                        isValid = False
                if not isValid: break
        self.ISVALID = isValid
        return isValid

    def is_to_be_delete(self):
        if self.ISVALID:
            toDel = False
            if self.sett.saddle_point["ValidSPs"]["CheckConnectivity"]:
                if not self.ISCONNECT: toDel = True
            if not toDel:
                if self.BARR < self.sett.saddle_point["BarrierMin"] or \
                        self.DMAG < self.sett.saddle_point["DmagMin"] or \
                        self.FDMAG > self.FDMAGCUT or \
                        self.FDMAG < self.sett.saddle_point["DmagMin_FI"] or \
                        self.BARR - self.EBIAS < self.sett.saddle_point["BackBarrierMin"] or \
                        self.EBIAS > self.EBIASCUT:
                    toDel = True
                else:
                    if (isinstance(self.sett.saddle_point["EbiasMin"], float) and
                            self.EBIAS < self.sett.saddle_point["EbiasMin"]):
                        toDel = True
        else:
            toDel = True
        return toDel

    def calculate_prefactor(self):
        ## only code for Harmonic
        if self.ISVALID and self.dmAV is not None:
            self.force_evaluator.close()
            self.dmAV.set_vib()
            nfixed = 0

            thisdata = copy.deepcopy(self.data)
            thisdata.update_avcoords_from_disps(self.XDISP)
            [encalc, coords, isValid, errormsg] = self.force_evaluator.run_runner("SPSDYNMAT", thisdata, self.thiscolor,
                                                                                  nactive=self.data.nactive)
            if not isValid:
                print(errormsg)
                quit()
            thisdata = None

            fname = "Runner_" + str(self.thiscolor) + "/dynmat.dat"
            if self.sett.dynamic_matrix["OutDynMat"]:
                outf = "KMC_" + str(self.ikmc) + "_AV_" + str(self.idav)
                outf += "_SPS_" + str(self.idsps) + ".dat"
                outf = os.path.join(self.DynMatOutpath, outf)
                if not os.path.exists(outf):
                    shutil.copy(fname, outf)

            delimiter = self.sett.dynamic_matrix["delimiter"]
            vibcut = self.sett.dynamic_matrix["VibCut"]
            LowerHalfMat = self.sett.dynamic_matrix["LowerHalfMat"]
            vmSP = VibMat.from_file(fname, id=self.thiscolor,
                                    delimiter=delimiter, vibcut=vibcut, LowerHalfMat=LowerHalfMat)
            vmSP.diagonize_matrix()
            vmSP.set_vib()
            vmSP.nfixed = nfixed
            vmSP.nactive = self.data.nactive
            ##if vmSP.isValid and vmSP.nfixed>0:
            ##    vmSP.split_vibmat()

            if vmSP.isValid:
                vms = VibMats(self.thiscolor, self.dmAV, vmSP, self.sett.dynamic_matrix["Method4Prefactor"])
                ##vms = VibMats(self.runner.id, vmAV, vmSP, self.sett.dynamic_matrix["Method4Prefactor"])
                self.PREF = vms.get_prefactor()


class Dimer(SPSearch):
    def __init__(
            self,
            idav,
            idsps,
            data,
            sett,
            thiscolor,
            force_evaluator,
            SNC=False,
            dmAV=None,
            pre_disps=[],
            apply_mass=False,
            ikmc=0,
            DynMatOutpath="DynMatOut",
    ):
        super().__init__(
            idav,
            idsps,
            data,
            sett,
            thiscolor,
            force_evaluator,
            SNC=SNC,
            dmAV=dmAV,
            pre_disps=pre_disps,
            apply_mass=apply_mass,
            ikmc=ikmc,
            DynMatOutpath=DynMatOutpath,
        )

    def dimer_init(self, thisVN=None):
        atoms_array = self.data.atoms_to_array(self.data.atoms, OutIndex=False)
        mask = np.hstack((np.ones(self.data.nactive, dtype=bool), np.zeros(self.data.nbuffer, dtype=bool),
                          np.zeros(self.data.nfixed, dtype=bool)))
        atoms = atoms_array[mask]
        self.X0 = np.vstack((atoms["x"], atoms["y"], atoms["z"]))  #/self.Rescale
        self.XACT = np.copy(self.X0)

        mask = np.hstack((np.zeros(self.data.nactive, dtype=bool), np.ones(self.data.nbuffer, dtype=bool),
                          np.zeros(self.data.nfixed, dtype=bool)))
        atoms = atoms_array[mask]
        XBUF = np.vstack((atoms["x"], atoms["y"], atoms["z"]))
        mask = np.hstack((np.zeros(self.data.nactive, dtype=bool), np.zeros(self.data.nbuffer, dtype=bool),
                          np.ones(self.data.nfixed, dtype=bool)))
        atoms = atoms_array[mask]
        XFIX = np.vstack((atoms["x"], atoms["y"], atoms["z"]))
        self.XNOT = np.hstack((XBUF, XFIX))  #/self.Rescale
        self.XCR = np.copy(self.X0)
        self.XA = np.hstack((self.XCR, self.XNOT))
        if self.SNC: self.X0 = self.convert_coord_forward(self.X0)
        self.XC = np.copy(self.X0)
        self.F0 = np.zeros((3, self.nactive), dtype=float)
        self.F1 = np.copy(self.F0)
        self.F2 = np.copy(self.F0)
        self.FC = np.copy(self.F0)
        self.FE = np.copy(self.F0)
        self.FN = np.copy(self.F0)
        self.FL = np.copy(self.F0)
        self.GN = np.copy(self.F0)
        self.GU = np.copy(self.F0)
        self.TF = np.copy(self.F0)
        self.TG = np.copy(self.F0)
        self.TU = np.copy(self.F0)
        self.SX = None

        self.DI_CON = False
        self.DI_MAX = False
        self.FO_CON = False
        self.ST_CON = False
        self.EN_CON = False

        self.RO_NEW = True
        self.RO_FWD = True
        self.TR_FWD = True

        self.RO_OPT = False
        self.RO_CGI = False

        self.ROTFN1 = 0.0
        self.ROTFN2 = 0.0
        self.ROCURV = 0.0
        self.FNUNIT = 0.0
        self.ROFNRS = 0.0

        self.TRGAMN = 0.0
        self.TRCURV = 0.0
        self.NROITR = 0
        self.NTSITR = 0

        self.RO_IDLE = False
        self.TR_HOR = False

        self.DRATIO = self.sett.spsearch["DRatio4Relax"]
        self.RDIMER = self.sett.spsearch["DimerSep"]
        self.isTR_HOR = self.sett.spsearch["TransHorizon"]

        self.FineStep = False
        if isinstance(self.pre_disps, list) or isinstance(self.pre_disps, np.ndarray):
            if len(self.pre_disps) > 0:
                self.FineStep = True

        if self.FineStep:
            self.TSTEPF = self.sett.spsearch["TrialStepsizeFine"]  # /self.Rescale
            self.TSTEPM = self.sett.spsearch["MaxStepsizeFine"]  # /self.Rescale
            self.RatioStepsize = self.TSTEPM / self.TSTEPF
            self.DECAYSTYLE = "Fixed"
            self.IgnoreTransSteps = self.sett.spsearch["IgnoreStepsFine"]
        else:
            self.TSTEPF = self.sett.spsearch["TrialStepsize"]  # /self.Rescale
            self.TSTEPM = self.sett.spsearch["MaxStepsize"]  # /self.Rescale
            self.RatioStepsize = self.TSTEPM / self.TSTEPF
            self.DECAYSTYLE = self.sett.spsearch["DecayStyle"]
            self.IgnoreTransSteps = self.sett.spsearch["IgnoreSteps"]

        if thisVN is not None:
            self.VN = thisVN.copy()
        else:
            self.VN = self.generate_VN()

        '''
        fthisVN = "VN_"+str(self.idsps)+".dat"
        if os.path.isfile(fthisVN):
            self.VN = np.loadtxt(fthisVN)
            self.VN = self.VN[:, 0:168]
            ndiff = self.nactive-self.VN.shape[1]
            if ndiff>0:
                vnapp = np.zeros([3,ndiff], dtype=float)
                self.VN = np.hstack((self.VN, vnapp))
            elif ndiff<0:
                self.VN = self.VN[:, 0:self.nactive]
            else: pass
        else:
            np.savetxt(fthisVN, self.VN)
            self.VN = self.VN[:, 0:168]
            ndiff = self.nactive-self.VN.shape[1]
            if ndiff>0:
                vnapp = np.zeros([3,ndiff], dtype=float)
                self.VN = np.hstack((self.VN, vnapp))
            elif ndiff<0:
                self.VN = self.VN[:, 0:self.nactive]
            else: pass
        '''

        if self.FixType: self.VN = self.VN * self.Scaling4Types
        self.VN = mat_unit(self.VN)

        self.CenterVN = self.sett.spsearch["HandleVN"]["CenterVN"]
        self.RescaleRAS = None
        self.RescaleLOGV = None
        self.xRescale = None
        self.Rescale = None
        self.isRescale = self.sett.spsearch["HandleVN"]["RescaleVN"]
        if self.SNC: self.isRescale = False
        self.WeightMA = 2.0 / (1 + self.sett.spsearch["HandleVN"]["Period4MA"])
        self.nCompRescale = 0
        self.PowerOnV = self.sett.spsearch["HandleVN"]["PowerOnV"]
        self.RescaleValue = self.sett.spsearch["HandleVN"]["RescaleValue"].upper()
        self.Ratio4Zero4RAS = self.sett.spsearch["HandleVN"]["Ratio4Zero4RAS"]
        self.Ratio4Zero4LOGV = self.sett.spsearch["HandleVN"]["Ratio4Zero4LOGV"]
        if self.isRescale: self.init_Rescales()

        self.isInitWell = True
        self.iter = 0
        [total_energy, coords, isValid, errormsg] = self.force_evaluator.init_spsearch_runner(self.data, self.thiscolor,
                                                                                              self.nactive)
        if not isValid:
            print(errormsg)
            quit()

        self.FC_ALL = None
        if self.sett.spsearch["OutForces4IterationResults"] and self.sett.spsearch["OutFix4IterationResults"]:
            self.FC_ALL = np.zeros([3, self.data.natoms], dtype=float)

    def update_translation_step(self, itstep):
        def get_decayrate(itstep):
            if self.DECAYSTYLE[0:3].upper() == "STA":
                return pow(self.sett.spsearch["DecayRate"], int(itstep / self.sett.spsearch["DecaySteps"]))
            else:
                return pow(self.sett.spsearch["DecayRate"], float(itstep) / float(self.sett.spsearch["DecaySteps"]))

        if self.DECAYSTYLE[0:3].upper() == "FIX":
            pass
        else:
            drate = get_decayrate(itstep)
            if self.FineStep:
                self.TSTEPF = (
                        self.sett.spsearch["TrialStepsizeFine"] * drate + self.sett.spsearch["MinStepsizeFine"])
                self.TSTEPM = (self.sett.spsearch["MaxStepsizeFine"] * drate + self.sett.spsearch["MinStepsizeFine"] *
                               self.sett.spsearch["RatioStepsizeFine"])
            else:
                self.TSTEPF = (
                        self.sett.spsearch["TrialStepsize"] * drate + self.sett.spsearch["MinStepsize"])
                self.TSTEPM = (self.sett.spsearch["MaxStepsize"] * drate + self.sett.spsearch["MinStepsize"] *
                               self.sett.spsearch["RatioStepsize"])

    def move_atoms_to_predisp(self):
        self.X0 = np.copy(self.XACT)
        ndiff = self.X0.shape[1] - self.pre_disps.shape[1]
        if ndiff > 0:
            thisdisp = np.hstack((self.pre_disps, np.zeros((3, ndiff), dtype=float)))
            if self.FixType: thisdisp = thisdisp * self.Scaling4Types
            self.X0 = self.X0 + thisdisp
            if self.sett.spsearch["HandleVN"]["ResetVN04Preload"]:
                self.VN = self.VN * self.sett.spsearch["HandleVN"]["RatioVN04Preload"] + thisdisp
        else:
            thisdisp = self.pre_disps[:, 0:self.X0.shape[1]]
            if self.FixType: thisdisp = thisdisp * self.Scaling4Types
            self.X0 = self.X0 + thisdisp
            if self.sett.spsearch["HandleVN"]["ResetVN04Preload"]:
                self.VN = self.VN * self.sett.spsearch["HandleVN"]["RatioVN04Preload"] + thisdisp
        thisdisp = None
        #if self.SNC: self.VN = self.convert_coord_forward(self.VN)
        if self.FixType: self.VN = self.VN * self.Scaling4Types
        self.VN = mat_unit(self.VN)

        self.XCR = np.copy(self.X0)
        self.XA = np.hstack((self.XCR, self.XNOT))
        if self.SNC: self.X0 = self.convert_coord_forward(self.X0)
        self.XC = np.copy(self.X0)

    def init_Rescales(self):
        self.xRescale = np.zeros(self.nactive)
        self.RescaleLOGV = np.ones(self.nactive, dtype=int)
        self.RescaleLOGV = self.RescaleLOGV.reshape([1, self.nactive])
        self.RescaleRAS = copy.deepcopy(self.RescaleLOGV)
        self.Rescale = copy.deepcopy(self.RescaleLOGV)
        if "RAS" in self.RescaleValue:
            RAS = self.nactive - np.arange(self.nactive)
            if "SIG" in self.sett.spsearch["HandleVN"]["RescaleStyle4RAS"].upper():
                self.RescaleRAS = self.generate_sigmoid_scale(RAS, range=self.sett.spsearch["HandleVN"]["XRange4RAS"],
                                                              r4zero=self.Ratio4Zero4RAS,
                                                              MinSpan=self.sett.spsearch["HandleVN"]["MinSpan4RAS"])
            else:
                self.RescaleRAS = self.generate_step_scale(RAS, r4zero=self.Ratio4Zero4RAS,
                                                           MinSpan=self.sett.spsearch["HandleVN"]["MinSpan4RAS"])
            self.Rescale = copy.deepcopy(self.RescaleRAS)

    def compute_xRescale_logdisps(self):
        if "SUM" in self.sett.spsearch["HandleVN"]["RescaleValue"].upper():
            d = self.XCR - self.XACT
            d = mat_unit(d)
        else:
            if self.SNC:
                d = self.convert_coord_backward(self.VN)
            else:
                d = self.VN.copy()
        xa = np.sqrt(np.sum(d * d, axis=0))
        xa = np.log(np.power(xa, self.PowerOnV) + np.exp(self.sett.spsearch["HandleVN"]["MinValue4LOGV"]))
        self.xRescale = (xa - self.xRescale) * self.WeightMA + self.xRescale

    def compute_RescaleRAS(self, multiplier=10, sort_by="DXYZ", AbsVal=True, range=50, r4zero=0.65, MinSpan=2.0):
        ds = self.XCR - self.XACT
        sds = np.sum(ds, axis=1)
        dc = sds * multiplier / ds.shape[1]

        xyz = self.XACT + dc.reshape([3, 1])
        thisx = self.get_AtomSquence(xyz, sort_by=sort_by, AbsVal=AbsVal)
        thisx = ds.shape[1] - 1 - thisx

        if "SIG" in self.sett.spsearch["HandleVN"]["RescaleStyle4RAS"].upper():
            RescaleRAS = self.generate_sigmoid_scale(thisx, range=range,
                                                     r4zero=r4zero, MinSpan=MinSpan)
        else:
            RescaleRAS = self.generate_step_scale(thisx, r4zero=r4zero, MinSpan=MinSpan)
        return RescaleRAS

    def compute_Rescale(self):
        if "LOG" in self.RescaleValue:
            if "SIG" in self.sett.spsearch["HandleVN"]["RescaleStyle4LOGV"].upper():
                self.RescaleLOGV = self.generate_sigmoid_scale(self.xRescale,
                                                               range=self.sett.spsearch["HandleVN"]["XRange4LOGV"],
                                                               r4zero=self.Ratio4Zero4LOGV,
                                                               MinSpan=self.sett.spsearch["HandleVN"]["MinSpan4LOGV"])
            else:
                self.RescaleLOGV = self.generate_step_scale(self.xRescale, r4zero=self.Ratio4Zero4LOGV,
                                                            MinSpan=self.sett.spsearch["HandleVN"]["MinSpan4LOGV"])

            if "RAS" in self.RescaleValue:
                if self.sett.spsearch["HandleVN"]["TakeMin4MixedRescales"]:
                    self.Rescale = np.select([self.RescaleLOGV <= self.RescaleRAS, self.RescaleLOGV > self.RescaleRAS],
                                             [self.RescaleLOGV, self.RescaleRAS])
                else:
                    self.Rescale = np.select([self.RescaleLOGV <= self.RescaleRAS, self.RescaleLOGV > self.RescaleRAS],
                                             [self.RescaleRAS, self.RescaleLOGV])
            else:
                self.Rescale = self.RescaleLOGV.copy()
        else:
            self.Rescale = self.RescaleRAS.copy()

    def apply_center_VN(self):
        if self.NTSITR > self.sett.spsearch["HandleVN"]["IgnoreSteps"]:
            if self.NTSITR % self.sett.spsearch["HandleVN"]["NSteps4CenterVN"] == 0:
                self.VN = self.center_array(self.VN)

    def apply_rescale_VN(self):
        #if self.isRescale and self.ROCURV>0.0 and self.TRCURV<0.0 and self.isInitWell:
        if "LOG" in self.sett.spsearch["HandleVN"]["RescaleValue"].upper():
            self.compute_xRescale_logdisps()
        if self.NTSITR > self.sett.spsearch["HandleVN"]["IgnoreSteps"]:
            if self.nCompRescale % self.sett.spsearch["HandleVN"]["Int4ComputeScale"] == 0:
                self.compute_Rescale()
            self.VN = self.VN * self.Rescale
            self.nCompRescale += 1

    def dimer_force(self):
        if self.SNC:
            self.XCR = self.convert_coord_backward(self.XC)
        else:
            self.XCR = self.XC.copy()
        self.XA = np.hstack((self.XCR, self.XNOT))

        [encalc, FC, isValid, errormsg] = self.force_evaluator.get_spsearch_forces(self.XA.T, self.data, self.thiscolor,
                                                                                   self.nactive)
        if not isValid:
            print(errormsg)
            quit()

        if self.FC_ALL is not None:
            self.FC_ALL = (FC.reshape([self.data.natoms, 3])).T
        mask = np.ones(3 * self.nactive, dtype=bool)
        mask = np.append(mask, np.zeros(3 * (self.data.natoms - self.nactive), dtype=bool))
        FC = FC[mask]
        if self.SNC: FC = self.convert_forces(FC)

        self.FC = (FC.reshape([self.nactive, 3])).T  #*self.Rescale
        if self.FixType: self.FC = self.FC * self.Scaling4Types
        if self.apply_mass: self.FC = self.FC * self.Scaling4Masses
        self.ENCALC = encalc

    def dimer_projection(self, TR_HOR=False):
        fs = np.multiply(self.VN, mats_sum_mul(self.F0, self.VN))
        if TR_HOR:
            fn = self.F0 - np.multiply(self.VN, mats_sum_mul(self.F0, self.VN))
        if self.ROCURV < 0:
            self.FE = self.F0 - 2.0 * fs
        else:
            if TR_HOR:
                self.FE = fn
            else:
                self.FE = 0.0 - fs

    def dimer_update(self):
        self.F1 = self.FC.copy()
        self.F2 = 2.0 * self.F0 - self.F1
        self.FN = 0.5 / self.RDIMER * (self.F1 - self.F2 - np.multiply(self.VN,
                                                                       mats_sum_mul(self.F1, self.VN) - mats_sum_mul(
                                                                           self.F2, self.VN)))
        self.FNUNIT = mat_mag(self.FN)

    def dimer_transform(self, V1, V2, THETA):
        cost = math.cos(THETA)
        sint = math.sin(THETA)
        V3 = V1.copy()
        V1 = V1 * cost + V2 * sint
        V2 = V2 * cost - V3 * sint
        return V1, V2

    def dimer_rotatestep(self):
        if self.RO_FWD:
            self.RO_FWD = False
            if self.RO_CGI:
                self.RO_CGI = False
                self.FL = self.FN.copy()
                self.GN = self.FN.copy()
            ROCGA1 = abs(mats_sum_mul(self.FN, self.FL))
            ROCGA2 = mats_sum_mul(self.FL, self.FL)

            if ROCGA1 <= 0.5 * ROCGA2 and ROCGA2 != 0.0:
                ROGAMN = mats_sum_mul(self.FN, self.FN - self.FL) / ROCGA2
            else:
                ROGAMN = 0.0
            if abs(ROGAMN) > 1.0:
                logstr = "ROGAMN is larger than 1.0."
                logstr += "\n" + f"ROCGA1:{ROCGA1}  ROCGA2:{ROCGA2}  ROGAMN:{ROGAMN}"
                print(logstr)
            self.GN = self.FN + self.GN * ROGAMN
            #if self.FixType: self.GN = self.GN*self.Scaling4Types

            self.GU = mat_unit(self.GN)
            self.ROTFN1 = mats_sum_mul(self.FN, self.GU)

            self.VN, self.GU = self.dimer_transform(self.VN, self.GU, 0.25 * pi)
            #if self.FixType: self.VN = self.VN*self.Scaling4Types

            self.XC = self.X0 + self.VN * self.RDIMER
            self.ROFNRS = self.FNUNIT

        else:
            self.RO_FWD = True
            self.NROITR += 1
            self.ROTFN2 = mats_sum_mul(self.FN, self.GU)
            if self.ROTFN2 != 0.0:
                RTHETA = -0.5 * math.atan(self.ROTFN1 / self.ROTFN2)
            else:
                RTHETA = -0.5 * pi
            if self.ROTFN2 > 0.0: RTHETA += 0.5 * pi
            RTHETA -= 0.25 * pi

            self.VN, self.GU = self.dimer_transform(self.VN, self.GU, RTHETA)
            if self.CenterVN: self.apply_center_VN()
            ##if self.sett.spsearch["HandleVN"]["AppRescaleRot"]:
            if self.isRescale and self.ROCURV > 0.0 and self.TRCURV < 0.0 and self.isInitWell:
                self.apply_rescale_VN()
            if self.FixType: self.VN = self.VN * self.Scaling4Types
            self.VN = mat_unit(self.VN)
            self.XC = self.X0 + np.multiply(self.VN, self.RDIMER)

    def dimer_rotate(self):
        if self.RO_NEW:
            self.RO_NEW = False
            self.RO_FWD = True
            self.NROITR = 1
            self.X0 = self.XC.copy()
            self.F0 = self.FC.copy()
            self.dimer_projection(TR_HOR=self.TR_HOR)
            self.XC = self.X0 + self.VN * self.RDIMER
        else:
            self.dimer_update()
            if self.RO_FWD:
                self.ROCURV = 0.5 / self.RDIMER * (mats_sum_mul(self.F2, self.VN) - mats_sum_mul(self.F1, self.VN))
                self.dimer_projection(TR_HOR=self.TR_HOR)
            if self.RO_IDLE:
                self.RO_OPT = True
            else:
                if self.FNUNIT < self.sett.spsearch["FMin4Rot"]:
                    self.RO_OPT = True
                else:
                    self.dimer_rotatestep()
                    if self.RO_FWD:
                        if self.NROITR > self.sett.spsearch["NMax4Rot"]:
                            self.RO_OPT = True
                        elif self.ROFNRS < self.sett.spsearch["FThres4Rot"]:
                            self.RO_OPT = True
                        self.dimer_projection(TR_HOR=self.TR_HOR)
            if self.RO_OPT:
                self.XC = self.X0.copy()
                self.RO_NEW = True
                self.RO_FWD = True
                self.NROITR = 0
                self.NTSITR += 1

                if self.ROCURV < 0.0:
                    self.isInitWell = False
                if self.isTR_HOR and self.TRCURV < -0.01 and self.ROCURV > 0.01:
                    if self.isInitWell:
                        self.TR_HOR = False
                    else:
                        if self.EDIFF > self.sett.spsearch["En4TransHorizon"]:
                            self.TR_HOR = True
                        else:
                            self.TR_HOR = False
                else:
                    self.TR_HOR = False

                '''
                if self.NTSITR % 10 == 0:
                    print("AFTER DIMER ROTATE RO_OPT == T")
                    print(f"iter: {self.iter} NR: {self.NROITR} NTSITER:{self.NTSITR}")
                    print(f"ROCURV: {self.ROCURV} TRCURV: {self.TRCURV} TR_HOR: {self.TR_HOR} IB:{self.isInitWell}")
                    print('===========')
                '''
            self.FC = self.FE.copy()

    def dimer_rotate_optimum(self):
        self.F0 = self.FC.copy()
        self.dimer_projection(TR_HOR=self.TR_HOR)
        self.FC = self.FE.copy()

    def dimer_translate(self):
        if self.TR_FWD:
            self.TR_FWD = False
            self.RO_OPT = True

            TRCGA1 = abs(mats_sum_mul(self.FC, self.TF))
            TRCGA2 = mats_sum_mul(self.TF, self.TF)
            if TRCGA1 <= 0.5 * TRCGA2 and TRCGA2 != 0.0:
                self.TRGAMN = mats_sum_mul(self.FC, self.FC - self.TF) / TRCGA2
            else:
                self.TRGAMN = 0.0

            self.TG = self.FC + self.TG * self.TRGAMN
            #if self.FixType: self.TG = self.TG*self.Scaling4Types
            self.TU = mat_unit(self.TG)
            self.TF = self.FC.copy()
            self.XC = self.XC + self.TU * self.TSTEPF

            '''
            print("DIMER TRANSLATE TR_FWD")
            print(f"iter: {self.iter} NTSITER:{self.NTSITR} NROITR: {self.NROITR}")
            print(f"TRCGA1: {TRCGA1}, TRCGA2: {TRCGA2}")
            print(f"TRGAMN: {self.TRGAMN}")
            print('-----------')
            '''
        else:
            self.TR_FWD = True
            self.RO_OPT = False
            TRTFP1 = mats_sum_mul(self.TF, self.TU)
            TRTFP2 = mats_sum_mul(self.FC, self.TU)
            self.TRCURV = (TRTFP1 - TRTFP2) / self.TSTEPF
            if self.TRCURV < 0.0:
                TSTEPC = self.TSTEPM
            else:
                TSTEPC = 0.5 * (TRTFP1 + TRTFP2) / (self.TRCURV + 1.0e-12)

                if abs(TSTEPC) > self.TSTEPM:
                    if TSTEPC >= 0:
                        TSTEPC = self.TSTEPM - self.TSTEPF
                    else:
                        TSTEPC = self.TSTEPF - self.TSTEPM
                else:
                    TSTEPC = TSTEPC - 0.5 * self.TSTEPF

            self.XC = self.XC + self.TU * TSTEPC

            '''
            print("DIMER TRANSLATE TR_FWD=F")
            print(f"iter: {self.iter} NTSITER:{self.NTSITR} NROITR: {self.NROITR}")
            print(f"TRTFP1: {TRTFP1}, TRTFP2: {TRTFP2}")
            print(f"TRCURV: {self.TRCURV}  TSTEPC: {TSTEPC}")
            print('-----------')
            '''

    def init_iteration_results(self):
        os.makedirs("ITERATION_RESULTS", exist_ok=True)
        thisdir = "ITERATION_RESULTS/" + str(self.ikmc) + "_" + str(self.idav) + "_" + str(self.idsps)
        os.makedirs(thisdir, exist_ok=True)
        summary_array = np.array([])
        summary_cols = ["iter", "ntrans", "curvature", "ediff", "ediffmax"]
        return thisdir, summary_array, summary_cols

    def IR_append_summary_array(self, summary_array):
        summary_array = np.append(summary_array, [int(self.iter), int(self.NTSITR), round(self.ROCURV, 3),
                                                  round(self.EDIFF, 3), round(self.EDIFF_MAX, 3)])
        return summary_array

    def IR_output_VN(self, thisdir):
        filename = thisdir + "/VectorN_" + str(self.NTSITR)  + ".csv"
        if not os.path.isfile(filename):
            df = pd.DataFrame(self.VN.T, columns=["x", "y", "z"])
            df.to_csv(filename)

    def IR_output_ReaxCoords(self, thisdir):
        filename = thisdir + "/Coords_" + str(self.NTSITR) + ".csv"
        if not os.path.isfile(filename):
            if self.sett.spsearch["OutForces4IterationResults"]:
                columns = ['type', 'x', 'y', 'z', 'fx', 'fy', 'fz']
                if self.sett.spsearch["OutFix4IterationResults"]:
                    data = np.vstack((self.TYPES_ALL, self.XA, self.FC_ALL))
                else:
                    data = np.vstack((self.TYPES, self.XCR, self.FC))
            else:
                columns = ['type', 'x', 'y', 'z']
                if self.sett.spsearch["OutFix4IterationResults"]:
                    data = np.vstack((self.TYPES_ALL, self.XA))
                else:
                    data = np.vstack((self.TYPES, self.XCR))
            df = pd.DataFrame(data.T, columns=columns)
            df.to_csv(filename)


    def IR_output_summary(self, thisdir, summary_array, summary_cols):
        ncols = len(summary_cols)
        n = summary_array.shape[0]
        nrow = int(n / ncols)
        summary_array = summary_array.reshape([nrow, ncols])
        df_summary = pd.DataFrame(summary_array, columns=summary_cols)
        df_summary.to_csv(thisdir + "/summary_array.csv")

    def dimer_search(self, thisSPS, ReSearch=False):
        if self.sett.spsearch["ShowIterationResults"]:
            thisdir, summary_array, summary_cols = self.init_iteration_results()
        while not self.DI_CON:
            if self.sett.spsearch["ShowIterationResults"]:
                if self.NTSITR % self.sett.spsearch["Inteval4ShowIterationResults"] == 0:
                    summary_array = self.IR_append_summary_array(summary_array)
                    if self.sett.spsearch["ShowVN4ShowIterationResults"]:
                        self.IR_output_VN(thisdir)
                    if self.sett.spsearch["ShowCoords4ShowIterationResults"]:
                        self.IR_output_ReaxCoords(thisdir)

            self.dimer_force()
            if self.iter == 0:
                self.ENINIT = self.ENCALC
                if not ReSearch:
                    if len(self.pre_disps) > 0:
                        self.move_atoms_to_predisp()
                        self.reprepare_runner()
                        self.dimer_force()
                        self.IgnoreTransSteps = 1

            self.EDIFF = self.ENCALC - self.ENINIT
            if self.EDIFF > self.EDIFF_MAX: self.EDIFF_MAX = self.EDIFF

            if self.TR_FWD and self.NROITR == 0:
                if abs(self.ENCALC - self.ENLAST) < self.ECONV:
                    self.EN_CON = True
                    self.ENLAST = self.ENCALC
                else:
                    self.EN_CON = False

            if self.RO_OPT:
                self.dimer_rotate_optimum()
            else:
                self.dimer_rotate()
            ftotal = mat_mag(self.FC)

            if ftotal < self.FCONV:
                self.FO_CON = True
            else:
                self.FO_CON = False
            if self.NTSITR > self.IgnoreTransSteps: self.ST_CON = True
            if self.FO_CON and self.ST_CON: self.DI_CON = True

            if self.ST_CON and self.sett.spsearch["CheckAng"] and self.NTSITR % self.sett.spsearch[
                "CheckAngSteps"] == 0:
                self.XDISP = self.XCR - self.XACT
                self.DMAG = mat_mag(self.XDISP)
                self.Check_Angle(thisSPS)

            if not self.DI_CON:
                if self.RO_OPT:
                    self.update_translation_step(self.NTSITR)
                    self.dimer_translate()

            if self.NTSITR >= self.sett.spsearch["NMax4Trans"]:
                self.DI_MAX = True
                self.DI_CON = True

            '''
            if self.NTSITR % 10 == 0:
                print("DIMER SEARCH")
                print(f"iter: {self.iter} NR: {self.NROITR} NTSITER:{self.NTSITR}")
                print(f"ROCURV: {self.ROCURV} TRCURV: {self.TRCURV} TR_HOR: {self.TR_HOR} IB:{self.isInitWell}")
                print(f"FTOTAL: {ftotal} EDIFF:{self.EDIFF} EDIFF_MAX: {self.EDIFF_MAX}")
                print("+++++++++++++++++++++++++++")
            '''
            self.iter += 1
        self.BARR += self.ENCALC - self.ENINIT
        self.ENLAST = self.ENCALC

        if self.sett.spsearch["ShowIterationResults"]:
            self.IR_output_summary(thisdir, summary_array, summary_cols)


    def dimer_re_search(self, thisSPS, nactive=None):
        if not isinstance(nactive, int): nactive = self.data.nactive + self.data.nbuffer
        ndiff = nactive - self.nactive
        nfixed = self.data.natoms - nactive
        isValid = True
        if ndiff <= 0:
            warnings.warn(
                "The number of atoms for this dimer search is smaller or equal than number of atoms of last one.")
            isValid = False
        if nfixed < 0:
            warnings.warn("The number of atoms for dimer search is larger than number of atoms in active volume.")
            isValid = False

        if isValid and self.DI_MAX is False:
            self.nactive = nactive
            self.reprepare_runner()
            XBUF = self.XNOT[:, 0:ndiff]
            self.XNOT = self.XNOT[:, ndiff:self.XNOT.shape[1]]

            if self.SNC: self.X0 = self.convert_coord_backward(self.X0)
            self.XC = np.hstack((self.X0, XBUF))
            self.XCR = np.hstack((self.XCR, XBUF))
            self.X0 = np.hstack((self.X0, XBUF))
            self.XACT = np.hstack((self.XACT, XBUF))

            self.F0 = np.zeros((3, self.nactive), dtype=float)
            self.F1 = np.copy(self.F0)
            self.F2 = np.copy(self.F0)
            self.FC = np.copy(self.F0)
            self.FE = np.copy(self.F0)
            self.FN = np.copy(self.F0)
            self.FL = np.copy(self.F0)
            self.GN = np.copy(self.F0)
            self.GU = np.copy(self.F0)
            self.TF = np.copy(self.F0)
            self.TG = np.copy(self.F0)
            self.TU = np.copy(self.F0)

            self.DI_CON = False
            self.FO_CON = False
            self.ST_CON = False
            self.EN_CON = False

            self.RO_NEW = True
            self.RO_FWD = True
            self.TR_FWD = True

            self.RO_OPT = False
            self.RO_CGI = False

            self.ROTFN2 = 0.0
            self.ROTFN1 = 0.0
            self.ROCURV = 0.0
            self.FNUNIT = 0.0
            self.ROFNRS = 0.0

            self.TRGAMN = 0.0
            self.TRCURV = 0.0
            self.NROITR = 0

            if self.SNC: self.VN = self.convert_coord_backward(self.VN)
            #self.VN = np.hstack((self.VN, np.random.rand(3, ndiff)-0.5))
            self.VN = np.hstack((self.VN, np.zeros((3, ndiff))))

            if self.sett.spsearch["FixTypes_dict"] is None:
                self.FixType = False
                self.Scaling4Types = np.ones([3, self.nactive], dtype=float)
            else:
                self.FixType = True
                self.get_scaling_fixtypes()

            if self.FixType: self.VN = self.VN * self.Scaling4Types
            self.VN = mat_unit(self.VN)

            if self.apply_mass:
                self.Scaling4Masses = self.get_scaling_masses()
            else:
                self.Scaling4Masses = np.ones([3, self.nactive], dtype=float)

            self.SNC = False
            self.isRescale = False
            self.xRescale = None
            self.Rescale = None
            self.isTR_HOR = False
            self.TR_HOR = False
            self.CenterVN = False

            self.FineStep = True
            self.TSTEPF = self.sett.spsearch["TrialStepsizeFine"]  # /self.Rescale
            self.TSTEPM = self.sett.spsearch["MaxStepsizeFine"]  # /self.Rescale
            self.RatioStepsize = self.TSTEPM / self.TSTEPF
            self.DECAYSTYLE = "FIXED"
            self.IgnoreTransSteps = self.sett.spsearch["IgnoreStepsFine"]
            #self.TSTEPF = self.sett.spsearch["MinStepsize"]
            #self.TSTEPM = self.sett.spsearch["MinStepsize"]*self.sett.spsearch["RatioStepsize"]

            self.ENINIT = 0.0
            self.iter = 0
            if self.FC_ALL is not None:
                self.FC_ALL = np.zeros([3, self.data.natoms], dtype=float)
            self.dimer_search(thisSPS, ReSearch=True)

    def dimer_relaxation(self):
        fxact = np.dot(self.data.box.inv_matrix.T, self.XACT)

        def shift_coords(coords):
            fcoords = np.dot(self.data.box.inv_matrix.T, coords)
            for idim in range(self.data.dimension):
                if self.data.PBC[idim]:
                    diff = fcoords[idim] - fxact[idim]
                    fcoords[idim] = np.select([diff < -0.5, diff < 0.5, diff >= 0.5],
                                              [fcoords[idim] + 1.0, fcoords[idim], fcoords[idim] - 1.0])
            coords = np.dot(self.data.box.matrix.T, fcoords)
            return coords

        X1 = self.DRATIO * (self.XCR - self.X0) + self.X0
        X2 = self.DRATIO * (self.X0 - self.XCR) + self.X0
        X12 = [X1, X2]
        enlist = []
        xfinlist = []
        dmaglist = []
        for i in range(len(X12)):
            self.XA = np.hstack((X12[i], self.XNOT))
            thisdata = copy.deepcopy(self.data)
            thisdata.update_coords(self.XA)
            self.force_evaluator.close()
            encalc, coords, isValid, errormsg = self.force_evaluator.run_runner("SPSRELAX", thisdata, self.thiscolor,
                                                                                nactive=self.nactive)
            if not isValid:
                print(errormsg)
                quit()
            mask = np.ones(3 * self.nactive, dtype=bool)
            mask = np.append(mask, np.zeros(3 * (self.data.natoms - self.nactive), dtype=bool))
            coords = coords[mask]
            coords = (coords.reshape([self.nactive, 3])).T
            coords = shift_coords(coords)
            enlist.append(encalc)
            xfinlist.append(coords)
            dmaglist.append(mat_mag(coords - self.XACT))
        thisdata = None

        dmagmax = max(dmaglist)
        idmax = dmaglist.index(dmagmax)
        idmin = int((idmax + 1) % 2)
        dmax = np.max(np.absolute(xfinlist[idmax] - self.XACT))
        self.DMAX_CONN = np.max(np.absolute(xfinlist[idmin] - self.XACT))
        if dmax < self.Tol4Connect:
            self.ISVALID = False
            self.ISCONNECT = False
        else:
            if self.DMAX_CONN >= -self.Tol4Connect and self.DMAX_CONN < self.Tol4Connect:
                self.ISVALID = True
                self.ISCONNECT = True
            else:
                self.ISVALID = True
                self.ISCONNECT = False
        return enlist[idmax], X12[idmax], xfinlist[idmax]

    def remove_invalid_outputs(self):
        if not self.ISVALID:
            if self.sett.spsearch["ShowIterationResults"]:
                thisdir = "ITERATION_RESULTS/" + str(self.ikmc) + "_" + str(self.idav) + "_" + str(self.idsps)
                if os.path.isdir(thisdir): shutil.rmtree(thisdir)

            if self.sett.dynamic_matrix["OutDynMat"]:
                outf = "KMC_" + str(self.ikmc) + "_AV_" + str(self.idav)
                outf += "_SPS_" + str(self.idsps) + ".dat"
                outf = os.path.join(self.DynMatOutpath, outf)
                if os.path.isfile(outf): os.remove(outf)

    def dimer_finalize(self):
        if self.DI_MAX:
            self.BARR = self.sett.saddle_point["BarrierCut"] + 10.0
            self.XCR = self.XCR[:, 0:self.data.nactive]
            self.XACT = self.XACT[:, 0:self.data.nactive]
            self.XDISP = self.XCR - self.XACT
            self.DMAT = self.data.get_disp_mat(self.XCR, self.XACT)
            self.DVEC = np.sum(self.XDISP, axis=1)
            self.DMAG = mat_mag(self.XDISP)
            self.FXDISP = self.XDISP.copy()
            self.FDMAG = self.DMAG
            self.DMAX_CONN = self.DMAG
            self.EBIAS = self.BARR
            self.ISVALID = False
            self.ISCONNECT = False
        else:
            if self.SNC:
                self.X0 = self.convert_coord_backward(self.X0)  #*self.Rescale
                self.XCR = self.convert_coord_backward(self.XC)
            else:
                self.XCR = self.XC.copy()
            if self.sett.spsearch["LocalRelax"]["LocalRelax"]:
                EFIN, SX, XFIN = self.dimer_relaxation()
            else:
                X1 = self.DRATIO * (self.XCR - self.X0) + self.X0
                X2 = self.DRATIO * (self.X0 - self.XCR) + self.X0
                if mat_mag(X1 - self.XACT) >= mat_mag(X2 - self.XACT):
                    SX = X1.copy()
                else:
                    SX = X2.copy()
                XFIN = SX.copy()
                EFIN = self.ENINIT
                self.ISVALID = True
                self.ISCONNECT = True
                self.DMAX_CONN = 0.0
            if self.sett.spsearch["ActiveOnly4SPConfig"]:
                SX = SX[:, 0:self.data.nactive]
                XFIN = XFIN[:, 0:self.data.nactive]
                self.XACT = self.XACT[:, 0:self.data.nactive]

            self.XDISP = SX - self.XACT
            self.DMAT = self.data.get_disp_mat(SX, self.XACT)
            self.DVEC = np.sum(self.XDISP, axis=1)
            self.DMAG = mat_mag(self.XDISP)

            self.FXDISP = XFIN - self.XACT
            self.FDMAG = mat_mag(self.FXDISP)

            self.EBIAS = self.BARR + EFIN - self.ENLAST

            if self.BARR > self.sett.saddle_point["BarrierCut"]:
                self.ISVALID = False
            elif self.DMAG > self.DMAGCUT:
                self.ISVALID = False
        self.remove_invalid_outputs()

    def dimer_finish(self):
        self.force_evaluator.close()

        self.X0 = None
        self.XACT = None
        self.XNOT = None
        self.XCR = None
        self.XA = None
        self.XC = None
        self.F0 = None
        self.F1 = None
        self.F2 = None
        self.FC = None
        self.FE = None
        self.FN = None
        self.FL = None
        self.GN = None
        self.GU = None
        self.TF = None
        self.TG = None
        self.TU = None
        self.RescaleRAS = None
        self.RescaleLOGV = None
        self.xRescale = None
        self.Rescale = None
        self.data = None
        self.dmAV = None
        self.pre_disps = []
        self.FC_ALL = None
        self.TYPES = None
        self.TYPES_ALL = None
        self.Scaling4Types = None
        self.Scaling4Masses = None



