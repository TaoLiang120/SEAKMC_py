import os, warnings
import numpy as np
import pandas as pd
import copy

from seakmc.core.util import mat_lengths, mat_angles, mats_angles, mats_angle, mat_mag
from seakmc.input.Input import SP_COMPACT_HEADER, SP_COMPACT_HEADER4Delete, DEFECTBANK_ATOMS_HEADER
from seakmc.input.Input import DEFECTBANK_DISPS_HEADER
from seakmc.input.Input import SP_DATA_HEADER, NDISPARRAY, NENTRY_COMPACT_DISP

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


class SaddlePoint:
    def __init__(
            self,
            idav,
            idsps,
            itype,
            barrier,
            prefactor,
            ebias,
            isconnect,
            disp,
            dmag,
            dmat,
            dvec,
            fdisp,
            fdmag,
            ISVALID,
            iters=1000000,
            ntrans=100000,
            emax=100.0,
            rdcut=0.01,
            dcut=0.01,
            dyncut=False,
            tol=0.1,
    ):
        self.idav = idav
        self.idsps = idsps
        self.itype = itype
        self.barrier = barrier
        self.prefactor = prefactor
        self.ebias = ebias
        self.isconnect = isconnect

        self.disp = disp
        self.dmag = dmag
        self.dmat = dmat
        self.dvec = dvec

        self.fdisp = fdisp
        self.fdmag = fdmag

        self.ISVALID = ISVALID
        self.iters = iters
        self.ntrans = ntrans
        self.emax = emax
        self.rdcut = rdcut
        self.dcut = dcut
        self.DynCut = dyncut
        self.tol = tol

        self.dtot, self.dmax, self.dna, self.dsum, self.adsum = self.info_from_displacement(DispStyle="SP")
        self.fdtot, self.fdmax, self.fdna, self.fdsum, self.fadsum = self.info_from_displacement(DispStyle="FI")
        self.fsdtot, self.fsdmax, self.fsdna, self.fsdsum, self.fsadsum = self.info_from_displacement(DispStyle="FS")
        self.fsdmag = mat_mag(self.fdisp - self.disp)

        self.dlen = mat_lengths(self.dmat)
        self.natoms = self.disp.shape[1]

    def __str__(self):
        return "This saddle point id is ({}) and active volume id is ({}).".format(self.idsps, self.idav)

    def __repr__(self):
        return self.__str__()

    def info_from_displacement(self, DispStyle="SP"):
        def get_info(x, rdcut, dcut, DynCut=False):
            xa = np.sqrt(np.sum(x * x, axis=0))
            dtot = np.sum(xa)
            dmax = np.max(xa)
            if DynCut:
                s = np.std(xa)
                n = xa.shape[0]
                cut = np.log(n) * s
            else:
                cut = max(rdcut * dmax, dcut)
            xa = np.compress(xa > cut, xa, axis=0)
            dna = len(xa)
            dsum = mat_mag(np.sum(x, axis=1))
            adsum = max(mat_mag(np.sum(np.absolute(x), axis=1)), 1.0e-6)
            return dtot, dmax, dna, dsum, adsum

        if DispStyle[0:2].upper() == "FS":
            dtot, dmax, dna, dsum, adsum = get_info(self.fdisp - self.disp, self.rdcut, self.dcut, DynCut=self.DynCut)
        elif DispStyle[0:2].upper() == "FI":
            dtot, dmax, dna, dsum, adsum = get_info(self.fdisp, self.rdcut, self.dcut, DynCut=self.DynCut)
        else:
            dtot, dmax, dna, dsum, adsum = get_info(self.disp, self.rdcut, self.dcut, DynCut=self.DynCut)
        return dtot, dmax, dna, dsum, adsum

    def array_partition_info(self, x):
        dmags = np.sqrt(np.sum(x * x, axis=1))
        return dmags

    def array_ddisplacement_info(self, x, SumAbs=False, AbsVal=True):
        if SumAbs:
            dsums = np.sum(np.absolute(x), axis=1)
        else:
            dsums = np.sum(x, axis=1)
        if AbsVal: dsums = np.absolute(dsums)
        return dsums

    def vector_partition_info(self, x, Part_Max=False, AbsVal=True):
        if Part_Max:
            xa = np.sum(x * x, axis=0)
            idmax = np.argmax(xa)
            dmaxs = np.array([x[0][idmax], x[1][idmax], x[2][idmax]])
            if AbsVal: dmaxs = np.absolute(dmaxs)
        else:
            xa = np.sqrt(x * x)
            dmaxs = np.max(xa, axis=1)
            #if AbsVal: dmaxs = np.absolute(dmaxs)
        return dmaxs

    def get_disp_value(self, Str="SP", Dtype="DMAG", AbsVal=True):
        Dtype = Dtype.upper()
        if Str[0:2].upper() == "FI":
            if "DMAG" in Dtype:
                dmags = self.array_partition_info(self.fdisp)
            if "DSUM" in Dtype:
                dsums = self.array_ddisplacement_info(self.fdisp, SumAbs=False, AbsVal=AbsVal)
                if "_RABS" in Dtype:
                    absdsums = self.array_ddisplacement_info(self.fdisp, SumAbs=True, AbsVal=False)
            if "DMAX" in Dtype:
                dmaxs = self.vector_partition_info(self.fdisp, Part_Max=True, AbsVal=AbsVal)
            if "VMAX" in Dtype:
                vmaxs = self.vector_partition_info(self.fdisp, Part_Max=False, AbsVal=AbsVal)
        elif Str[0:2].upper() == "FS" or Str[0:2].upper() == "SF":
            if "DMAG" in Dtype:
                dmags = self.array_partition_info(self.fdisp - self.disp)
            if "DSUM" in Dtype:
                dsums = self.array_ddisplacement_info(self.fdisp - self.disp, SumAbs=False, AbsVal=AbsVal)
                if "_RABS" in Dtype:
                    absdsums = self.array_ddisplacement_info(self.fdisp - self.disp, SumAbs=True, AbsVal=False)
            if "DMAX" in Dtype:
                dmaxs = self.vector_partition_info(self.fdisp - self.disp, Part_Max=True, AbsVal=AbsVal)
            if "VMAX" in Dtype:
                vmaxs = self.vector_partition_info(self.fdisp - self.disp, Part_Max=False, AbsVal=AbsVal)
        else:
            if "DMAG" in Dtype:
                dmags = self.array_partition_info(self.disp)
            if "DSUM" in Dtype:
                dsums = self.array_ddisplacement_info(self.disp, SumAbs=False, AbsVal=AbsVal)
                if "_RABS" in Dtype:
                    absdsums = self.array_ddisplacement_info(self.disp, SumAbs=True, AbsVal=False)
            if "DMAX" in Dtype:
                dmaxs = self.vector_partition_info(self.disp, Part_Max=True, AbsVal=AbsVal)
            if "VMAX" in Dtype:
                vmaxs = self.vector_partition_info(self.disp, Part_Max=False, AbsVal=AbsVal)

        if Dtype == "DMAG":
            thisval = mat_mag(dmags)
        elif Dtype == "DMAGX":
            thisval = dmags[0]
        elif Dtype == "DMAGY":
            thisval = dmags[1]
        elif Dtype == "DMAGZ":
            thisval = dmags[2]
        elif Dtype == "DMAG_DMIN":
            thisval = np.argmin(dmags)
        elif Dtype == "DMAG_DMAX":
            thisval = np.argmax(dmags)
        elif Dtype == "DMAG_DRX":
            thisval = abs(dmags[0]) / max(mat_mag(dmags), 1.0e-6)
        elif Dtype == "DMAG_DRY":
            thisval = abs(dmags[1]) / max(mat_mag(dmags), 1.0e-6)
        elif Dtype == "DMAG_DRZ":
            thisval = abs(dmags[2]) / max(mat_mag(dmags), 1.0e-6)
        elif Dtype == "DMAG_DRXY":
            thisval = abs(dmags[0]) / max(abs(dmags[1]), 1.0e-6)
        elif Dtype == "DMAG_DRXZ":
            thisval = abs(dmags[0]) / max(abs(dmags[2]), 1.0e-6)
        elif Dtype == "DMAG_DRYZ":
            thisval = abs(dmags[1]) / max(abs(dmags[2]), 1.0e-6)
        elif Dtype == "DSUM":
            thisval = mat_mag(dsums)
        elif Dtype == "DSUMX":
            thisval = dsums[0]
        elif Dtype == "DSUMY":
            thisval = dsums[1]
        elif Dtype == "DSUMZ":
            thisval = dsums[2]
        elif Dtype == "DSUM_DMIN":
            thisval = np.argmin(dsums)
        elif Dtype == "DSUM_DMAX":
            thisval = np.argmax(dsums)
        elif Dtype == "DSUM_DRX":
            thisval = abs(dsums[0]) / max(mat_mag(dsums), 1.0e-6)
        elif Dtype == "DSUM_DRY":
            thisval = abs(dsums[1]) / max(mat_mag(dsums), 1.0e-6)
        elif Dtype == "DSUM_DRZ":
            thisval = abs(dsums[2]) / max(mat_mag(dsums), 1.0e-6)
        elif Dtype == "DSUM_DRXY":
            thisval = abs(dsums[0]) / max(abs(dsums[1]), 1.0e-6)
        elif Dtype == "DSUM_DRXZ":
            thisval = abs(dsums[0]) / max(abs(dsums[2]), 1.0e-6)
        elif Dtype == "DSUM_DRYZ":
            thisval = abs(dsums[1]) / max(abs(dsums[2]), 1.0e-6)
        elif Dtype == "DSUM_RABS":
            thisval = mat_mag(dsums) / max(mat_mag(absdsums), 1.0e-6)
        elif Dtype == "DSUMX_RABS":
            thisval = dsums[0] / max(absdsums[0], 1.0e-6)
        elif Dtype == "DSUMY_RABS":
            thisval = dsums[1] / max(absdsums[1], 1.0e-6)
        elif Dtype == "DSUMZ_RABS":
            thisval = dsums[2] / max(absdsums[2], 1.0e-6)
        elif Dtype == "DMAX":
            thisval = mat_mag(dmaxs)
        elif Dtype == "DMAXX":
            thisval = dmaxs[0]
        elif Dtype == "DMAXY":
            thisval = dmaxs[1]
        elif Dtype == "DMAXZ":
            thisval = dmaxs[2]
        elif Dtype == "DMAX_DMIN":
            thisval = np.argmin(dmaxs)
        elif Dtype == "DMAX_DMAX":
            thisval = np.argmax(dmaxs)
        elif Dtype == "DMAX_DRX":
            thisval = abs(dmaxs[0]) / max(mat_mag(dmaxs), 1.0e-6)
        elif Dtype == "DMAX_DRY":
            thisval = abs(dmaxs[1]) / max(mat_mag(dmaxs), 1.0e-6)
        elif Dtype == "DMAX_DRZ":
            thisval = abs(dmaxs[2]) / max(mat_mag(dmaxs), 1.0e-6)
        elif Dtype == "DMAX_DRXY":
            thisval = abs(dmaxs[0]) / max(abs(dmaxs[1]), 1.0e-6)
        elif Dtype == "DMAX_DRXZ":
            thisval = abs(dmaxs[0]) / max(abs(dmaxs[2]), 1.0e-6)
        elif Dtype == "DMAX_DRYZ":
            thisval = abs(dmaxs[1]) / max(abs(dmaxs[2]), 1.0e-6)
        elif Dtype == "VMAX":
            thisval = mat_mag(vmaxs)
        elif Dtype == "VMAXX":
            thisval = vmaxs[0]
        elif Dtype == "VMAXY":
            thisval = vmaxs[1]
        elif Dtype == "VMAXZ":
            thisval = vmaxs[2]
        elif Dtype == "VMAX_DMIN":
            thisval = np.argmin(vmaxs)
        elif Dtype == "VMAX_DMAX":
            thisval = np.argmax(vmaxs)
        elif Dtype == "VMAX_DRX":
            thisval = abs(vmaxs[0]) / max(mat_mag(vmaxs), 1.0e-6)
        elif Dtype == "VMAX_DRY":
            thisval = abs(vmaxs[1]) / max(mat_mag(vmaxs), 1.0e-6)
        elif Dtype == "VMAX_DRZ":
            thisval = abs(vmaxs[2]) / max(mat_mag(vmaxs), 1.0e-6)
        elif Dtype == "VMAX_DRXY":
            thisval = abs(vmaxs[0]) / max(abs(vmaxs[1]), 1.0e-6)
        elif Dtype == "VMAX_DRXZ":
            thisval = abs(vmaxs[0]) / max(abs(vmaxs[2]), 1.0e-6)
        elif Dtype == "VMAX_DRYZ":
            thisval = abs(vmaxs[1]) / max(abs(vmaxs[2]), 1.0e-6)
        else:
            raise ValueError("Unrecognized displacement type for validation!")
            quit()
        return thisval

    def get_energy_value(self, Etype="EBIAS"):
        if Etype[0:4].upper() == "BARR":
            thisval = self.barrier
        elif Etype[0:4].upper() == "BBAR":
            thisval = self.barrier - self.ebias
        else:
            thisval = self.ebias
        return thisval

    def copy(self, deep=True):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def deepcopy(self):
        return self.copy(deep=True)

    def get_disp_from_opmat(self, coords, opmat, radius):
        thisdisp = np.zeros((3, coords.shape[1]))
        thisfdisp = np.zeros((3, coords.shape[1]))
        isValid = True
        for i in range(coords.shape[1]):
            ixyz = np.array([coords[0][i], coords[1][i], coords[2][i]])
            newxyz = np.dot(opmat, ixyz.T)
            thisd = coords - newxyz.reshape([3, 1])
            thisdsq = np.sum(thisd * thisd, axis=0)
            ai = np.argmin(thisdsq, axis=0)
            if np.sqrt(thisdsq[ai]) < self.tol * 3.0:
                disxyz = np.array([self.disp[0][ai], self.disp[1][ai], self.disp[2][ai]])
                disxyz = np.dot(disxyz, opmat)
                thisdisp[0][i] = disxyz[0]
                thisdisp[1][i] = disxyz[1]
                thisdisp[2][i] = disxyz[2]
                fdisxyz = np.array([self.fdisp[0][ai], self.fdisp[1][ai], self.fdisp[2][ai]])
                fdisxyz = np.dot(fdisxyz, opmat)
                thisfdisp[0][i] = fdisxyz[0]
                thisfdisp[1][i] = fdisxyz[1]
                thisfdisp[2][i] = fdisxyz[2]
            elif mat_mag(newxyz) > radius:
                a = np.argsort(thisdsq, axis=0)
                disxyz = np.zeros(3)
                fdisxyz = np.zeros(3)
                for j in range(4):
                    aj = a[j]
                    disp = np.array([self.disp[0][aj], self.disp[1][aj], self.disp[2][aj]])
                    disxyz += np.dot(disp, opmat)
                    fdisp = np.array([self.fdisp[0][aj], self.fdisp[1][aj], self.fdisp[2][aj]])
                    fdisxyz += np.dot(fdisp, opmat)

                thisdisp[0][i] = disxyz[0] / 4.0
                thisdisp[1][i] = disxyz[1] / 4.0
                thisdisp[2][i] = disxyz[2] / 4.0

                thisfdisp[0][i] = fdisxyz[0] / 4.0
                thisfdisp[1][i] = fdisxyz[1] / 4.0
                thisfdisp[2][i] = fdisxyz[2] / 4.0
            else:
                warnings.warn("The symmetry operater is not valid.")
                isValid = False
                break
        return thisdisp, thisfdisp, isValid

    def get_SPs_from_symmetry(self, coords, symmOPs):
        rl = []
        for i in range(3):
            rl.append(0.5 * (max(coords[i]) - min(coords[i])))
        radiusmin = min(rl)
        IDENTITY = np.eye(3)
        newSPs = []
        for i in range(symmOPs.nOP):
            if mat_mag(symmOPs.OPs[i].rotation_matrix - IDENTITY) <= self.tol:
                pass
            else:
                thisdisp, thisfdisp, isValid = self.get_disp_from_opmat(coords, symmOPs.OPs[i].rotation_matrix,
                                                                        radiusmin)
                if isValid:
                    isDup = False
                    for j in range(len(newSPs)):
                        if np.max(np.absolute(thisdisp - newSPs[j].disp)) < self.tol:
                            isDup = True
                            break
                    if not isDup:
                        if np.max(np.absolute(thisdisp - self.disp)) < self.tol:
                            isDup = True
                    if not isDup:
                        thisdmat = np.dot(thisdisp, coords.T)
                        thisdvec = np.sum(thisdisp, axis=1)
                        thisdmag = mat_mag(thisdisp)
                        thisfdmag = mat_mag(thisfdisp)
                        thissp = SaddlePoint(self.idav, self.idsps, self.itype, self.barrier, self.prefactor, self.ebias,
                                             self.isconnect,
                                             thisdisp, thisdmag, thisdmat, thisdvec,
                                             thisfdisp, thisfdmag, self.ISVALID,
                                             iters=self.iters, ntrans=self.ntrans, emax=self.emax,
                                             rdcut=self.rdcut, dcut=self.dcut, dyncut=self.DynCut, tol=self.tol)
                        newSPs.append(thissp)
        return newSPs


class AV_SPs:
    def __init__(
            self,
            idav,
            coords,
            SPlist,
            ntype,
            sett,
            float_precision=3,
    ):
        self.idav = idav
        self.coords = coords
        self.SPlist = SPlist
        self.nSP = len(self.SPlist)
        self.ntype = ntype
        self.sett = sett
        self.float_precision = float_precision

        self.get_this_thres()
        self.dmagmin = self.this_dmag_cut
        self.barriermin = self.this_barr_cut
        self.barriermax = self.this_min_barr
        self.dmagmax = self.this_min_dmax

        self.nvalid = self.nSP
        self.type_info = None

    def __str__(self):
        return "This saddle points id is ({}).".format(self.idav)

    def __repr__(self):
        return self.__str__()

    def copy(self, deep=True):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def deepcopy(self):
        return self.copy(deep=True)

    def get_this_thres(self):
        self.this_barr_cut = self.sett["BarrierCut"]
        self.this_min_barr = self.sett["BarrierMin"]
        self.this_ebias_cut = self.sett["EbiasCut"]
        self.this_min_ebias = self.sett["EbiasMin"]
        self.this_min_backbarr = self.sett["BackBarrierMin"]
        if isinstance(self.this_ebias_cut, str): self.this_ebias_cut = self.this_barr_cut * self.coords.shape[1]
        #if isinstance(this_min_ebias, str): this_min_ebias = -self.sett["BarrierCut"]*self.coords.shape[1]
        self.this_dmag_cut = self.sett["DmagCut"]
        self.this_min_dmag = self.sett["DmagMin"]
        self.this_dtot_cut = self.sett["DtotCut"]
        self.this_min_dtot = self.sett["DtotMin"]
        self.this_dmax_cut = self.sett["DmaxCut"]
        self.this_min_dmax = self.sett["DmaxMin"]
        self.this_dsum_cut = self.sett["DsumCut"]
        self.this_min_dsum = self.sett["DsumMin"]
        self.this_dsum_rcut = self.sett["DsumrCut"]
        self.this_rmin_dsum = self.sett["DsumrMin"]

        self.this_fdmag_cut = self.sett["DmagCut_FI"]
        self.this_min_fdmag = self.sett["DmagMin_FI"]
        self.this_fdtot_cut = self.sett["DtotCut_FI"]
        self.this_min_fdtot = self.sett["DtotMin_FI"]
        self.this_fdmax_cut = self.sett["DmaxCut_FI"]
        self.this_min_fdmax = self.sett["DmaxMin_FI"]
        self.this_fdsum_cut = self.sett["DsumCut_FI"]
        self.this_min_fdsum = self.sett["DsumMin_FI"]
        self.this_fdsum_rcut = self.sett["DsumrCut_FI"]
        self.this_rmin_fdsum = self.sett["DsumrMin_FI"]

        self.this_fsdmag_cut = self.sett["DmagCut_FS"]
        self.this_min_fsdmag = self.sett["DmagMin_FS"]
        self.this_fsdtot_cut = self.sett["DtotCut_FS"]
        self.this_min_fsdtot = self.sett["DtotMin_FS"]
        self.this_fsdmax_cut = self.sett["DmaxCut_FS"]
        self.this_min_fsdmax = self.sett["DmaxMin_FS"]
        self.this_fsdsum_cut = self.sett["DsumCut_FS"]
        self.this_min_fsdsum = self.sett["DsumMin_FS"]
        self.this_fsdsum_rcut = self.sett["DsumrCut_FS"]
        self.this_rmin_fsdsum = self.sett["DsumrMin_FS"]

        self.DATOMCUT = self.sett["DAtomCut"]
        self.DTOTCUT = self.DATOMCUT * self.coords.shape[1]
        if isinstance(self.this_dmag_cut, str): self.this_dmag_cut = self.DTOTCUT
        if isinstance(self.this_min_dmag, str): self.this_min_dmag = 0.0
        if isinstance(self.this_dtot_cut, str): self.this_dtot_cut = self.DTOTCUT
        if isinstance(self.this_min_dtot, str): self.this_min_dtot = 0.0
        if isinstance(self.this_dmax_cut, str): self.this_dmax_cut = self.DTOTCUT
        if isinstance(self.this_min_dmax, str): self.this_min_dmax = 0.0
        if isinstance(self.this_dsum_cut, str): self.this_dsum_cut = self.DTOTCUT
        if isinstance(self.this_min_dsum, str): self.this_min_dsum = 0.0
        if isinstance(self.this_dsum_rcut, str): self.this_dsum_rcut = 1.1
        if isinstance(self.this_rmin_dsum, str): self.this_rmin_dsum = 0.0

        if isinstance(self.this_fdmag_cut, str): self.this_fdmag_cut = self.DTOTCUT
        if isinstance(self.this_min_fdmag, str): self.this_min_fdmag = 0.0
        if isinstance(self.this_fdtot_cut, str): self.this_fdtot_cut = self.DTOTCUT
        if isinstance(self.this_min_fdtot, str): self.this_min_fdtot = 0.0
        if isinstance(self.this_fdmax_cut, str): self.this_fdmax_cut = self.DTOTCUT
        if isinstance(self.this_min_fdmax, str): self.this_min_fdmax = 0.0
        if isinstance(self.this_fdsum_cut, str): self.this_fdsum_cut = self.DTOTCUT
        if isinstance(self.this_min_fdsum, str): self.this_min_fdsum = 0.0
        if isinstance(self.this_fdsum_rcut, str): self.this_fdsum_rcut = 1.1
        if isinstance(self.this_rmin_fdsum, str): self.this_rmin_fdsum = 0.0

        if isinstance(self.this_fsdmag_cut, str): self.this_fsdmag_cut = self.DTOTCUT
        if isinstance(self.this_min_fsdmag, str): self.this_min_fsdmag = 0.0
        if isinstance(self.this_fsdtot_cut, str): self.this_fsdtot_cut = self.DTOTCUT
        if isinstance(self.this_min_fsdtot, str): self.this_min_fsdtot = 0.0
        if isinstance(self.this_fsdmax_cut, str): self.this_fsdmax_cut = self.DTOTCUT
        if isinstance(self.this_min_fsdmax, str): self.this_min_fsdmax = 0.0
        if isinstance(self.this_fsdsum_cut, str): self.this_fsdsum_cut = self.DTOTCUT
        if isinstance(self.this_min_fsdsum, str): self.this_min_fsdsum = 0.0
        if isinstance(self.this_fsdsum_rcut, str): self.this_fsdsum_rcut = 1.1
        if isinstance(self.this_rmin_fsdsum, str): self.this_rmin_fsdsum = 0.0

        self.Check_barr_ratio = True
        self.Check_dmag_ratio = True
        self.this_barr_ratio = self.sett["ValidSPs"]["MaxRatio4Barr"]
        self.this_dmag_ratio = self.sett["ValidSPs"]["MaxRatio4Dmag"]
        if isinstance(self.this_barr_ratio, str):
            self.Check_barr_ratio = False
            self.this_barr_ratio = 10.0
        if isinstance(self.this_dmag_ratio, str):
            self.Check_dmag_ratio = False
            self.this_dmag_ratio = 10.0

    def insert_SP(self, sp):
        if isinstance(sp, SaddlePoint):
            self.SPlist.append(sp)
            self.nvalid += 1
        elif isinstance(sp, list):
            self.SPlist += sp
            self.nvalid += len(sp)
        else:
            warnings.warn("Insert must be an instance of SaddlePoint or a list of SaddlePoint.")
        self.nSP = len(self.SPlist)

    def get_min_bar_dmag(self, barriercut=1000000):
        typemask = np.zeros(self.ntype, dtype=bool)
        for i in range(self.nSP):
            if not typemask[self.SPlist[i].itype - 1]:
                if self.SPlist[i].dmag < self.dmagmin: self.dmagmin = self.SPlist[i].dmag
                if self.SPlist[i].barrier < self.barriermin: self.barriermin = self.SPlist[i].barrier
                if self.SPlist[i].dmag > self.dmagmax: self.dmagmax = self.SPlist[i].dmag
                if self.SPlist[i].barrier > self.barriermax: self.barriermax = self.SPlist[i].barrier
                typemask[self.SPlist[i].itype - 1] = True

    def init_compact_arrays(self):
        arraylist = []
        idavs = np.array([], dtype=int)
        idims = np.array([], dtype=int)
        itypes = np.array([], dtype=int)
        iters = np.array([], dtype=int)
        ntrans = np.array([], dtype=int)
        emaxs = np.array([])
        barrs = np.array([])
        prefs = np.array([])
        biass = np.array([])
        arraylist = [idavs, idims, itypes, iters, ntrans, emaxs, barrs, prefs, biass]

        for i in range(NDISPARRAY * NENTRY_COMPACT_DISP):
            a = np.array([])
            arraylist.append(a)
        isConns = np.array([], dtype=int)
        arraylist.append(isConns)
        return arraylist

    def append_SP2compact_arrays(self, thisSP, arraylist):
        arraylist[0] = np.append(arraylist[0], [thisSP.idav])
        arraylist[1] = np.append(arraylist[1], [thisSP.idsps])
        arraylist[2] = np.append(arraylist[2], [thisSP.itype])
        arraylist[3] = np.append(arraylist[3], [thisSP.iters])
        arraylist[4] = np.append(arraylist[4], [thisSP.ntrans])
        arraylist[5] = np.append(arraylist[5], [round(thisSP.emax, self.float_precision)])

        arraylist[6] = np.append(arraylist[6], [round(thisSP.barrier, self.float_precision)])
        arraylist[7] = np.append(arraylist[7], [round(thisSP.prefactor, self.float_precision)])
        arraylist[8] = np.append(arraylist[8], [round(thisSP.ebias, self.float_precision)])

        arraylist[9] = np.append(arraylist[9], [round(thisSP.dtot, self.float_precision)])
        arraylist[10] = np.append(arraylist[10], [round(thisSP.dmag, self.float_precision)])
        arraylist[11] = np.append(arraylist[11], [round(thisSP.dmax, self.float_precision)])
        arraylist[12] = np.append(arraylist[12], [round(thisSP.dsum, self.float_precision)])
        arraylist[13] = np.append(arraylist[13], [round(thisSP.adsum, self.float_precision)])
        arraylist[14] = np.append(arraylist[14], [thisSP.dna])

        arraylist[15] = np.append(arraylist[15], [round(thisSP.fsdtot, self.float_precision)])
        arraylist[16] = np.append(arraylist[16], [round(thisSP.fsdmag, self.float_precision)])
        arraylist[17] = np.append(arraylist[17], [round(thisSP.fsdmax, self.float_precision)])
        arraylist[18] = np.append(arraylist[18], [round(thisSP.fsdsum, self.float_precision)])
        arraylist[19] = np.append(arraylist[19], [round(thisSP.fsadsum, self.float_precision)])
        arraylist[20] = np.append(arraylist[20], [thisSP.fsdna])

        arraylist[21] = np.append(arraylist[21], [round(thisSP.fdtot, self.float_precision)])
        arraylist[22] = np.append(arraylist[22], [round(thisSP.fdmag, self.float_precision)])
        arraylist[23] = np.append(arraylist[23], [round(thisSP.fdmax, self.float_precision)])
        arraylist[24] = np.append(arraylist[24], [round(thisSP.fdsum, self.float_precision)])
        arraylist[25] = np.append(arraylist[25], [round(thisSP.fadsum, self.float_precision)])
        arraylist[26] = np.append(arraylist[26], [thisSP.fdna])

        arraylist[27] = np.append(arraylist[27], [thisSP.isconnect])
        return arraylist

    def get_compact_df(self, arraylist):
        marray = zip(arraylist[0], arraylist[1], arraylist[2], arraylist[3], arraylist[4], arraylist[5],
                     arraylist[6], arraylist[7], arraylist[8], arraylist[9], arraylist[10], arraylist[11],
                     arraylist[12], arraylist[13], arraylist[14].astype(int), arraylist[15], arraylist[16],
                     arraylist[17],
                     arraylist[18], arraylist[19], arraylist[20].astype(int), arraylist[21], arraylist[22],
                     arraylist[23],
                     arraylist[24], arraylist[25], arraylist[26].astype(int), arraylist[27].astype(int))
        array = list(tuple(marray))
        df = pd.DataFrame(array, columns=SP_COMPACT_HEADER)
        return df

    def to_dataframe(self, iSPs="All"):
        if isinstance(iSPs, int):
            iSPs = np.array([iSPs])
        elif isinstance(iSPs, list) or isinstance(iSPs, np.ndarray):
            iSPs = np.array(iSPs).astype(int)
        else:
            iSPs = np.arange(self.nSP, dtype=int)
        arraylist = self.init_compact_arrays()
        for i in iSPs:
            arraylist = self.append_SP2compact_arrays(self.SPlist[i], arraylist)
        df_compact = self.get_compact_df(arraylist)
        return df_compact

    def insert_reason_df(self, df, reasons):
        if len(reasons) > 0:
            if len(df) > 0:
                if "reason" in df.columns: df = df.drop(["reason"], axis=1)
            im = SP_COMPACT_HEADER4Delete.index("reason")
            df.insert(im, "reason", reasons)
        return df

    def remove_small_barrier(self, minbarrier=0.0):
        typemask = np.zeros(self.ntype, dtype=bool)
        typedelmask = np.zeros(self.ntype, dtype=bool)
        todel = []
        for i in range(self.nSP):
            if not typemask[self.SPlist[i].itype - 1]:
                if self.SPlist[i].barrier < minbarrier:
                    todel.append(i)
                    typedelmask[self.SPlist[i].itype - 1] = True
                typemask[self.SPlist[i].itype - 1] = True
            else:
                if typedelmask[self.SPlist[i].itype - 1]: todel.append(i)

        arraylist = self.init_compact_arrays()
        todel = list(set(todel))
        for i in sorted(todel, reverse=True):
            arraylist = self.append_SP2compact_arrays(self.SPlist[i], arraylist)
            del self.SPlist[i]
        self.nSP = len(self.SPlist)
        self.nvalid = self.nSP

        if len(todel) > 0:
            df_delete_this = self.get_compact_df(arraylist)
            reasons = ["minB"] * len(df_delete_this)
            df_delete_this = self.insert_reason_df(df_delete_this, reasons)
        else:
            df_delete_this = pd.DataFrame(columns=SP_COMPACT_HEADER4Delete)

        return df_delete_this

    def is_duplicate(self, d1, d2, Tolerance=0.1):
        isDup = False
        if np.max(np.absolute(d1 - d2)) < Tolerance:
            isDup = True
        return isDup

    def check_this_duplicate(self, sp):
        isDup = False
        for i in range(self.nSP):
            if abs(self.SPlist[i].barrier - sp.barrier) < self.sett["ValidSPs"]["EnTol4AVSP"]:
                isDup = self.is_duplicate(self.SPlist[i].disp, sp.disp, Tolerance=self.sett["ValidSPs"]["Tol4AVSP"])
                if isDup: break
        if isDup:
            reasons = []
            arraylist = self.init_compact_arrays()
            arraylist = self.append_SP2compact_arrays(sp, arraylist)
            reasons.append("DupAV")
            df_delete_this = self.get_compact_df(arraylist)
            df_delete_this = self.insert_reason_df(df_delete_this, reasons)
        else:
            df_delete_this = pd.DataFrame(columns=SP_COMPACT_HEADER4Delete)
        return isDup, df_delete_this

    def this_screen_condition(self, v, xmin, xmax, NA_to=False):
        isInside = True
        NA = 0
        if isinstance(xmin, float) or isinstance(xmin, int):
            if v >= xmin:
                isInside = True
            else:
                isInside = False
        else:
            NA += 1
        if isinstance(xmax, float) or isinstance(xmax, int):
            if v <= xmax:
                pass
            else:
                isInside = False
        else:
            NA += 1
        if NA == 2: isInside = NA_to
        return isInside

    def get_Delete_Disp(self, thisSP, Conn=False):
        Delete_Disp = None
        if_Check = False
        if self.sett["ValidSPs"]["NScreenDisp"] > 0:
            if Conn:
                if self.sett["ValidSPs"]["toScreenDisp"].upper() == "ALL" or self.sett["ValidSPs"][
                    "toScreenDisp"].upper() == "CONN": if_Check = True
            else:
                if self.sett["ValidSPs"]["toScreenDisp"].upper() == "ALL" or self.sett["ValidSPs"][
                    "toScreenDisp"].upper() == "NOTCONN": if_Check = True
        if if_Check:
            Delete_Disp = False
            for idisp in range(self.sett["ValidSPs"]["NScreenDisp"]):
                thisval = thisSP.get_disp_value(Str=self.sett["ValidSPs"]["ScreenDisp"]["Str4ScreenD"][idisp],
                                                Dtype=self.sett["ValidSPs"]["ScreenDisp"]["Type4ScreenD"][idisp],
                                                AbsVal=self.sett["ValidSPs"]["ScreenDisp"]["AbsVal4ScreenD"][idisp])
                if self.sett["ValidSPs"]["ScreenDisp"]["AND4ScreenD"][idisp] and not Delete_Disp:
                    isInside = self.this_screen_condition(thisval,
                                                          self.sett["ValidSPs"]["ScreenDisp"]["MinVal4ScreenD"][idisp],
                                                          self.sett["ValidSPs"]["ScreenDisp"]["MaxVal4ScreenD"][idisp],
                                                          NA_to=False)
                    if isInside: Delete_Disp = True
                if Delete_Disp and not self.sett["ValidSPs"]["ScreenDisp"]["AND4ScreenD"][idisp]:
                    isRevoke = self.this_screen_condition(thisval,
                                                          self.sett["ValidSPs"]["ScreenDisp"]["MinVal4ScreenD"][idisp],
                                                          self.sett["ValidSPs"]["ScreenDisp"]["MaxVal4ScreenD"][idisp],
                                                          NA_to=False)
                    if isRevoke: Delete_Disp = False
        return Delete_Disp

    def get_Delete_Eng(self, thisSP, Conn=False):
        Delete_Eng = None
        if_Check = False
        if self.sett["ValidSPs"]["NScreenEng"] > 0:
            if Conn:
                if self.sett["ValidSPs"]["toScreenEng"].upper() == "ALL" or self.sett["ValidSPs"][
                    "toScreenEng"].upper() == "CONN": if_Check = True
            else:
                if self.sett["ValidSPs"]["toScreenEng"].upper() == "ALL" or self.sett["ValidSPs"][
                    "toScreenEng"].upper() == "NOTCONN": if_Check = True
        if if_Check:
            Delete_Eng = False
            for ieng in range(self.sett["ValidSPs"]["NScreenEng"]):
                thisval = thisSP.get_energy_value(Etype=self.sett["ValidSPs"]["ScreenEng"]["Type4ScreenE"][ieng])
                if self.sett["ValidSPs"]["ScreenEng"]["AND4ScreenE"][ieng] and not Delete_Eng:
                    isInside = self.this_screen_condition(thisval,
                                                          self.sett["ValidSPs"]["ScreenEng"]["MinVal4ScreenE"][ieng],
                                                          self.sett["ValidSPs"]["ScreenEng"]["MaxVal4ScreenE"][ieng],
                                                          NA_to=False)
                    if isInside: Delete_Eng = True
                if Delete_Eng and not self.sett["ValidSPs"]["ScreenEng"]["AND4ScreenE"][ieng]:
                    isRevoke = self.this_screen_condition(thisval,
                                                          self.sett["ValidSPs"]["ScreenEng"]["MinVal4ScreenE"][ieng],
                                                          self.sett["ValidSPs"]["ScreenEng"]["MaxVal4ScreenE"][ieng],
                                                          NA_to=False)
                    if isRevoke: Delete_Eng = False
        return Delete_Eng

    def valid_SPs(self, arraylist, reasons, Delete=True):
        if Delete:
            if self.sett["ValidSPs"]["CheckConnectivity"] or self.sett["ValidSPs"]["NScreenDisp"] > 0 or \
                    self.sett["ValidSPs"]["NScreenEng"] > 0:
                for i in range(self.nSP - 1, -1, -1):
                    if not self.SPlist[i].isconnect:
                        if self.sett["ValidSPs"]["CheckConnectivity"]:
                            Delete_this = True
                            Delete_Disp = None
                            Delete_Eng = None
                        else:
                            Delete_this = False
                            Delete_Disp = self.get_Delete_Disp(self.SPlist[i], Conn=False)
                            Delete_Eng = self.get_Delete_Eng(self.SPlist[i], Conn=False)
                    else:
                        Delete_this = False
                        Delete_Disp = self.get_Delete_Disp(self.SPlist[i], Conn=True)
                        Delete_Eng = self.get_Delete_Eng(self.SPlist[i], Conn=True)

                    if Delete_Disp is None and Delete_Eng is None:
                        if Delete_this: reasons.append("Conn")
                    elif Delete_Disp is None:
                        if Delete_Eng:
                            Delete_this = True
                            reasons.append("SE")
                        else:
                            if Delete_this: reasons.append("Conn")
                    elif Delete_Eng is None:
                        if Delete_Disp:
                            Delete_this = True
                            reasons.append("SD")
                        else:
                            if Delete_this: reasons.append("Conn")
                    else:
                        if self.sett["ValidSPs"]["AND4ScreenDE"]:
                            if Delete_Disp and Delete_Eng:
                                Delete_this = True
                                reasons.append("SDSE")
                            else:
                                if Delete_this: reasons.append("Conn")
                        else:
                            if Delete_Disp or Delete_Eng:
                                Delete_this = True
                                reasons.append("SDorSE")
                            else:
                                if Delete_this: reasons.append("Conn")
                    if Delete_this:
                        arraylist = self.append_SP2compact_arrays(self.SPlist[i], arraylist)
                        del self.SPlist[i]
                self.nSP = len(self.SPlist)
            self.nvalid = self.nSP
        return arraylist, reasons

    def validate_SPs(self, Delete=False):
        self.nvalid = self.nSP
        arraylist = self.init_compact_arrays()
        reasons = []
        arraylist, reasons = self.valid_SPs(arraylist, reasons, Delete=Delete)

        if self.Check_barr_ratio or self.Check_dmag_ratio: self.get_min_bar_dmag(barriercut=self.sett["BarrierCut"])
        if not self.Check_barr_ratio: self.barriermin = self.this_barr_cut
        if not self.Check_dmag_ratio: self.dmagmin = self.this_dmag_cut

        typemask = np.zeros(self.ntype, dtype=bool)
        typedelmask = np.zeros(self.ntype, dtype=bool)
        todel = []
        for i in range(self.nSP):
            if not typemask[self.SPlist[i].itype - 1]:
                if self.SPlist[i].barrier > self.this_barr_cut or \
                        self.SPlist[i].barrier < self.this_min_barr or \
                        self.SPlist[i].ebias > self.this_ebias_cut or \
                        self.SPlist[i].barrier / self.barriermin > self.this_barr_ratio or \
                        self.SPlist[i].dmag / self.dmagmin > self.this_dmag_ratio or \
                        self.SPlist[i].barrier - self.SPlist[i].ebias < self.this_min_backbarr or \
                        self.SPlist[i].dmag > self.this_dmag_cut or \
                        self.SPlist[i].dmag < self.this_min_dmag or \
                        self.SPlist[i].dtot > self.this_dtot_cut or \
                        self.SPlist[i].dtot < self.this_min_dtot or \
                        self.SPlist[i].dmax > self.this_dmax_cut or \
                        self.SPlist[i].dmax < self.this_min_dmax or \
                        self.SPlist[i].dsum > self.this_dsum_cut or \
                        self.SPlist[i].dsum < self.this_min_dsum or \
                        self.SPlist[i].dsum / self.SPlist[i].adsum > self.this_dsum_rcut or \
                        self.SPlist[i].dsum / self.SPlist[i].adsum < self.this_rmin_dsum or \
                        self.SPlist[i].fdmag > self.this_fdmag_cut or \
                        self.SPlist[i].fdmag < self.this_min_fdmag or \
                        self.SPlist[i].fdtot > self.this_fdtot_cut or \
                        self.SPlist[i].fdtot < self.this_min_fdtot or \
                        self.SPlist[i].fdmax > self.this_fdmax_cut or \
                        self.SPlist[i].fdmax < self.this_min_fdmax or \
                        self.SPlist[i].fdsum > self.this_fdsum_cut or \
                        self.SPlist[i].fdsum < self.this_min_fdsum or \
                        self.SPlist[i].fdsum / self.SPlist[i].fadsum > self.this_fdsum_rcut or \
                        self.SPlist[i].fdsum / self.SPlist[i].fadsum < self.this_rmin_fdsum or \
                        self.SPlist[i].fsdmag > self.this_fsdmag_cut or \
                        self.SPlist[i].fsdmag < self.this_min_fsdmag or \
                        self.SPlist[i].fsdtot > self.this_fsdtot_cut or \
                        self.SPlist[i].fsdtot < self.this_min_fsdtot or \
                        self.SPlist[i].fsdmax > self.this_fsdmax_cut or \
                        self.SPlist[i].fsdmax < self.this_min_fsdmax or \
                        self.SPlist[i].fsdsum > self.this_fsdsum_cut or \
                        self.SPlist[i].fsdsum < self.this_min_fsdsum or \
                        self.SPlist[i].fsdsum / self.SPlist[i].fsadsum > self.this_fsdsum_rcut or \
                        self.SPlist[i].fsdsum / self.SPlist[i].fsadsum < self.this_rmin_fsdsum:
                    todel.append(i)
                    typedelmask[self.SPlist[i].itype - 1] = True
                else:
                    if isinstance(self.this_min_ebias, float) and self.SPlist[i].ebias < self.this_min_ebias:
                        todel.append(i)
                        typedelmask[self.SPlist[i].itype - 1] = True
                typemask[self.SPlist[i].itype - 1] = True
            else:
                if typedelmask[self.SPlist[i].itype - 1]: todel.append(i)

        todel = list(set(todel))
        self.nvalid = self.nSP - len(todel)

        if Delete:
            for i in sorted(todel, reverse=True):
                arraylist = self.append_SP2compact_arrays(self.SPlist[i], arraylist)
                reasons.append("BorD")
                del self.SPlist[i]
            self.nSP = len(self.SPlist)
            self.nvalid = self.nSP

        if Delete:
            df_delete_this = self.get_compact_df(arraylist)
            df_delete_this = self.insert_reason_df(df_delete_this, reasons)
        else:
            df_delete_this = pd.DataFrame(columns=SP_COMPACT_HEADER4Delete)
        return df_delete_this

    def realtime_valid_thisSP(self, thisSP):
        reason = None
        Delete_this = False
        if self.sett["ValidSPs"]["CheckConnectivity"] or self.sett["ValidSPs"]["NScreenDisp"] > 0 or \
                self.sett["ValidSPs"]["NScreenEng"] > 0:
            if not thisSP.isconnect:
                if self.sett["ValidSPs"]["CheckConnectivity"]:
                    Delete_this = True
                    Delete_Disp = None
                    Delete_Eng = None
                else:
                    Delete_this = False
                    Delete_Disp = self.get_Delete_Disp(thisSP, Conn=False)
                    Delete_Eng = self.get_Delete_Eng(thisSP, Conn=False)
            else:
                Delete_this = False
                Delete_Disp = self.get_Delete_Disp(thisSP, Conn=True)
                Delete_Eng = self.get_Delete_Eng(thisSP, Conn=True)

            if Delete_Disp is None and Delete_Eng is None:
                if Delete_this: reason = "Conn"
            elif Delete_Disp is None:
                if Delete_Eng:
                    Delete_this = True
                    reason = "SE"
                else:
                    if Delete_this: reason = "Conn"
            elif Delete_Eng is None:
                if Delete_Disp:
                    Delete_this = True
                    reason = "SD"
                else:
                    if Delete_this: reason = "Conn"
            else:
                if self.sett["ValidSPs"]["AND4ScreenDE"]:
                    if Delete_Disp and Delete_Eng:
                        Delete_this = True
                        reason = "SDSE"
                    else:
                        if Delete_this: reason = "Conn"
                else:
                    if Delete_Disp or Delete_Eng:
                        Delete_this = True
                        reason = "SDorSE"
                    else:
                        if Delete_this: reason = "Conn"
        return reason, Delete_this

    def realtime_validate_thisSP(self, thisSP):
        df_delete_this = pd.DataFrame(columns=SP_COMPACT_HEADER4Delete)
        arraylist = self.init_compact_arrays()
        reason = None
        reason, Delete = self.realtime_valid_thisSP(thisSP)
        if Delete:
            arraylist = self.append_SP2compact_arrays(thisSP, arraylist)
            df_delete_this = self.get_compact_df(arraylist)
            df_delete_this = self.insert_reason_df(df_delete_this, [reason])
        else:
            if self.Check_barr_ratio or self.Check_dmag_ratio: self.get_min_bar_dmag(barriercut=self.sett["BarrierCut"])
            if not self.Check_barr_ratio: self.barriermin = self.this_barr_cut
            if not self.Check_dmag_ratio: self.dmagmin = self.this_dmag_cut
            if thisSP.barrier > self.this_barr_cut or \
                    thisSP.barrier < self.this_min_barr or \
                    thisSP.ebias > self.this_ebias_cut or \
                    thisSP.barrier / self.barriermin > self.this_barr_ratio or \
                    thisSP.dmag / self.dmagmin > self.this_dmag_ratio or \
                    thisSP.barrier - thisSP.ebias < self.this_min_backbarr or \
                    thisSP.dmag > self.this_dmag_cut or \
                    thisSP.dmag < self.this_min_dmag or \
                    thisSP.dtot > self.this_dtot_cut or \
                    thisSP.dtot < self.this_min_dtot or \
                    thisSP.dmax > self.this_dmax_cut or \
                    thisSP.dmax < self.this_min_dmax or \
                    thisSP.dsum > self.this_dsum_cut or \
                    thisSP.dsum < self.this_min_dsum or \
                    thisSP.dsum / thisSP.adsum > self.this_dsum_rcut or \
                    thisSP.dsum / thisSP.adsum < self.this_rmin_dsum or \
                    thisSP.fdmag > self.this_fdmag_cut or \
                    thisSP.fdmag < self.this_min_fdmag or \
                    thisSP.fdtot > self.this_fdtot_cut or \
                    thisSP.fdtot < self.this_min_fdtot or \
                    thisSP.fdmax > self.this_fdmax_cut or \
                    thisSP.fdmax < self.this_min_fdmax or \
                    thisSP.fdsum > self.this_fdsum_cut or \
                    thisSP.fdsum < self.this_min_fdsum or \
                    thisSP.fdsum / thisSP.fadsum > self.this_fdsum_rcut or \
                    thisSP.fdsum / thisSP.fadsum < self.this_rmin_fdsum or \
                    thisSP.fsdmag > self.this_fsdmag_cut or \
                    thisSP.fsdmag < self.this_min_fsdmag or \
                    thisSP.fsdtot > self.this_fsdtot_cut or \
                    thisSP.fsdtot < self.this_min_fsdtot or \
                    thisSP.fsdmax > self.this_fsdmax_cut or \
                    thisSP.fsdmax < self.this_min_fsdmax or \
                    thisSP.fsdsum > self.this_fsdsum_cut or \
                    thisSP.fsdsum < self.this_min_fsdsum or \
                    thisSP.fsdsum / thisSP.fsadsum > self.this_fsdsum_rcut or \
                    thisSP.fsdsum / thisSP.fsadsum < self.this_rmin_fsdsum:
                Delete = True
                reason = "BorD"
            else:
                if isinstance(self.this_min_ebias, float) and thisSP.ebias < self.this_min_ebias:
                    Delete = True
                    reason = "BorD"
            if Delete:
                arraylist = self.append_SP2compact_arrays(thisSP, arraylist)
                df_delete_this = self.get_compact_df(arraylist)
                df_delete_this = self.insert_reason_df(df_delete_this, [reason])
        return df_delete_this, Delete

    def build_arrays4checkdup(self):
        darrays = []
        earrays = []
        for isp in range(self.nSP):
            darrays.append(self.SPlist[isp].disp)
            earrays.append(self.SPlist[isp].barrier)
        return np.array(darrays), np.array(earrays)

    def check_duplicate(self, nstart=0):
        todel = []
        if nstart >= self.nSP:
            pass
        else:
            if self.nSP <= self.coords.shape[1]:
                thismask = np.zeros(self.nSP, dtype=bool)
                for i in range(nstart, self.nSP - 1):
                    for j in range(i + 1, self.nSP):
                        if not thismask[j]:
                            isDup = False
                            if abs(self.SPlist[i].barrier - self.SPlist[j].barrier) < self.sett["ValidSPs"][
                                "EnTol4AVSP"]:
                                isDup = self.is_duplicate(self.SPlist[i].disp, self.SPlist[j].disp,
                                                          Tolerance=self.sett["ValidSPs"]["Tol4AVSP"])
                            if isDup:
                                todel.append(j)
                                thismask[j] = True
            else:
                darrays, earrays = self.build_arrays4checkdup()
                for i in range(nstart, self.nSP - 1):
                    e = earrays[i + 1:self.nSP]
                    ediff = np.absolute(e - earrays[i])
                    isps = np.arange(self.nSP - i - 1, dtype=int)
                    ind = np.arange(i + 1, self.nSP, dtype=int)
                    isps = np.compress(ediff < self.sett["ValidSPs"]["EnTol4AVSP"], isps)
                    if isps.shape[0] > 0:
                        d = darrays[i + 1:self.nSP][:][:]
                        d = d[isps]
                        diff = np.absolute(d - darrays[i][:][:])
                        diff = np.max(np.max(diff, axis=1), axis=1)
                        isps = np.compress(diff < self.sett["ValidSPs"]["Tol4AVSP"], isps)
                        ind = ind[isps]
                        todel += list(ind)

            todel = list(set(todel))

        arraylist = self.init_compact_arrays()
        reasons = []
        for i in sorted(todel, reverse=True):
            arraylist = self.append_SP2compact_arrays(self.SPlist[i], arraylist)
            reasons.append("DupAV")
            del self.SPlist[i]
        self.nSP = len(self.SPlist)
        self.nvalid = self.nSP

        df_delete_this = self.get_compact_df(arraylist)
        df_delete_this = self.insert_reason_df(df_delete_this, reasons)

        return df_delete_this

    def group_saddles(self, group_by="dmat", SPlist=None):
        if not isinstance(SPlist, list): SPlist = self.SPlist.copy()
        nSP = len(SPlist)
        id_chain = np.zeros(nSP, dtype=int)
        chain_id = np.arange(nSP, dtype=int)
        nchain = 0
        idthis = 0
        if_loop = 0
        inext = nSP - 1
        group_info = [[] for i in range(nSP)]
        for i in range(nSP):
            isp = chain_id[i]
            idisp = id_chain[isp]
            if group_by == "type":
                ilens = mat_lengths(SPlist[isp].dmat, axis=1)
                ai = ilens.argsort(axis=0)
                ilens.sort(axis=0)
                iangs = mat_angles(SPlist[isp].dmat, axis=1)
                iangs = iangs[ai]

            if_loop = 0
            if idisp == 0:
                nchain += 1
                id_chain[isp] = nchain
                idisp = id_chain[isp]
                idthis += 1
                if_loop = 1
                group_info[idisp - 1].append(isp)

            for j in range(nSP):
                isgroup = False
                if j != isp:
                    idjsp = id_chain[j]
                    if idjsp == 0:
                        angles = mats_angles(SPlist[isp].dmat, SPlist[j].dmat)
                        angle = mats_angle(SPlist[isp].dvec, SPlist[j].dvec)
                        if group_by == "dmat":
                            if abs(SPlist[isp].barrier - SPlist[j].barrier) < self.sett["ValidSPs"]["EnCut4GSP"] and \
                                    abs(SPlist[isp].dlen[0] - SPlist[j].dlen[0]) < self.sett["ValidSPs"][
                                "MagCut4GSP"] and \
                                    abs(SPlist[isp].dlen[1] - SPlist[j].dlen[1]) < self.sett["ValidSPs"][
                                "MagCut4GSP"] and \
                                    abs(SPlist[isp].dlen[2] - SPlist[j].dlen[2]) < self.sett["ValidSPs"][
                                "MagCut4GSP"] and \
                                    abs(SPlist[isp].dmag - SPlist[j].dmag) < self.sett["ValidSPs"]["MagCut4GSP"]:
                                if angle < self.sett["ValidSPs"]["AngCut4GSP"] and \
                                        angles[0] < self.sett["ValidSPs"]["AngCut4GSP"] and \
                                        angles[1] < self.sett["ValidSPs"]["AngCut4GSP"] and \
                                        angles[2] < self.sett["ValidSPs"]["AngCut4GSP"]:
                                    isgroup = True
                        if group_by == "type":
                            if SPlist[isp].itype == SPlist[j].itype:
                                isgroup = True
                            else:
                                jlens = mat_lengths(SPlist[j].dmat, axis=1)
                                aj = jlens.argsort(axis=0)
                                jlens.sort(axis=0)
                                jangs = mat_angles(SPlist[j].dmat)
                                jangs = jangs[aj]
                                if abs(SPlist[isp].barrier - SPlist[j].barrier) < self.sett["ValidSPs"][
                                    "EnCut4Type"] and \
                                        abs(SPlist[isp].dmag - SPlist[j].dmag) < self.sett["ValidSPs"][
                                    "MagCut4Type"] and \
                                        abs(ilens[0] - jlens[0]) < self.sett["ValidSPs"]["LenCut4Type"] and \
                                        abs(ilens[1] - jlens[1]) < self.sett["ValidSPs"]["LenCut4Type"] and \
                                        abs(ilens[2] - jlens[2]) < self.sett["ValidSPs"]["LenCut4Type"]:
                                    if_angle = True
                                    for ii in range(iangs.shape[0]):
                                        if abs(iangs[ii] - jangs[ii]) > self.sett["ValidSPs"]["AngCut4Type"] and \
                                                abs(iangs[ii] + jangs[ii] - 180.0) > self.sett["ValidSPs"][
                                            "AngCut4Type"]:
                                            if_angle = False
                                            break
                                    if if_angle: isgroup = True

                    if isgroup:
                        id_chain[j] = idisp
                        chain_id[idthis] = j
                        if j == inext: if_loop = 1
                        idthis += 1
                        group_info[idisp - 1].append(j)

            if idthis < nSP:
                if if_loop == 0:
                    chain_id[idthis] = inext
                else:
                    for jj in range(inext, 0, -1):
                        if id_chain[jj] == 0:
                            chain_id[idthis] = jj
                            inext = jj
                            break

        group_info = group_info[0:nchain]
        return group_info

    def get_GSPs(self, SPlist=None):
        if not isinstance(SPlist, list): SPlist = self.SPlist.copy()
        if self.sett["ValidSPs"]["GroupSP"]:
            group_info = self.group_saddles(group_by="dmat", SPlist=SPlist)
            GSPs = []
            todel = []
            for i in range(len(group_info)):
                bl = []
                sumpre = 0.0
                mag = []
                for k in range(len(group_info[i])):
                    thisid = group_info[i][k]
                    bl.append(self.SPlist[thisid].barrier)
                    sumpre += self.SPlist[thisid].prefactor
                    mag.append(self.SPlist[thisid].dmag)
                    if k >= 1: todel.append(thisid)
                idm = mag.index(max(mag))
                idm = group_info[i][idm]
                id0 = group_info[i][0]
                if idm != id0:
                    self.SPlist[id0] = copy.deepcopy(self.SPlist[idm])
                self.SPlist[id0].barrier = max(bl)
                self.SPlist[id0].prefactor = sumpre
                GSPs.append(self.SPlist[id0])

            if len(todel) > 0:
                todel = list(set(todel))
                df_delete_this = self.to_dataframe(iSPs=todel)
                reasons = ["DupAV"] * len(df_delete_this)
                df_delete_this = self.insert_reason_df(df_delete_this, reasons)
            else:
                df_delete_this = pd.DataFrame(columns=SP_COMPACT_HEADER4Delete)
            self.SPlist = GSPs[0:len(GSPs)]
            self.nSP = len(self.SPlist)
        else:
            group_info = np.arange(self.nSP, dtype=int).reshape([self.nSP, 1])
            df_delete_this = pd.DataFrame(columns=SP_COMPACT_HEADER4Delete)
        return group_info, df_delete_this

    def get_SP_type(self, SPlist=None):
        if not isinstance(SPlist, list): SPlist = self.SPlist.copy()
        group_info = self.group_saddles(group_by="type", SPlist=SPlist)
        for i in range(len(group_info)):
            for k in range(len(group_info[i])):
                thisid = group_info[i][k]
                SPlist[thisid].itype = i + 1
        self.type_info = group_info
        self.ntype = len(self.type_info)
        return self.type_info


def find_common_atoms_AVs(AVitags, idav, jdav, iavna, javna, ncommonmin=40):
    isCommon = True
    itags = AVitags[idav]
    itags = itags[0:iavna]
    jtags = AVitags[jdav]
    jtags = jtags[0:javna]
    ijcom, icomind, jcomind = np.intersect1d(itags, jtags, return_indices=True)
    if len(ijcom) < ncommonmin: isCommon = False
    return ijcom, icomind, jcomind, isCommon


def check_oneside4dup(thisdisp, thiscomind, tol=0.1):
    invert_ind = np.ones(thisdisp.shape[1], dtype=bool)
    invert_ind[thiscomind] = False
    dispc = thisdisp[:, thiscomind]
    if thiscomind.shape[0] >= thisdisp.shape[1]:
        dmax = 0.0
    else:
        dispuc = thisdisp[:, invert_ind]
        dmax = np.max(np.absolute(dispuc))
    if dmax > tol:
        isCheck = False
    else:
        isCheck = True
    return dispc, isCheck


def check_dup_SPs(idisp, jdisp, ijcom, icomind, jcomind, tol=0.1):
    isDup = True
    idispc = idisp[:, icomind]
    jdispc = jdisp[:, jcomind]
    dmax = np.max(np.absolute(idispc - jdispc))
    if dmax > tol: isDup = False
    return isDup


def check_dup_dSPs(idispc, jdispc, tol=0.1):
    isDup = True
    dmax = np.max(np.absolute(idispc - jdispc))
    if dmax > tol: isDup = False
    return isDup


def check_duplicate_avSPs(avSPs, de_neighbors, AVitags, SPsett):
    df_delete_this = pd.DataFrame(columns=SP_COMPACT_HEADER4Delete)
    for i in range(len(avSPs) - 1):
        idav = avSPs[i].idav
        nlist = de_neighbors[idav]
        if len(nlist) == 0:
            pass
        else:
            for j in range(i + 1, len(avSPs)):
                jdav = avSPs[j]
                if jdav not in nlist:
                    pass
                else:
                    todel = []
                    iavna = min(avSPs[i].SPlist[0].disp.shape[1], SPsett["ValidSPs"]["NMax4Dup"])
                    javna = min(avSPs[j].SPlist[0].disp.shape[1], SPsett["ValidSPs"]["NMax4Dup"])
                    thisnmin = max(SPsett["ValidSPs"]["NCommonMin"], int(min(iavna, javna) * 0.5))
                    ijcom, icomind, jcomind, isCommon = find_common_atoms_AVs(AVitags, idav, jdav, iavna, javna,
                                                                              ncommonmin=thisnmin)
                    if isCommon:
                        for ii in range(avSPs[i].nSP):
                            isp = avSPs[i].SPlist[ii]
                            idisp = isp.disp.copy()
                            idisp = idisp[:, 0:iavna]
                            itol = min(SPsett["ValidSPs"]["Tol4Disp"], isp.dmax * SPsett["ValidSPs"]["R2Dmax4Tol"])
                            isDup = False
                            idispc, isCheck = check_oneside4dup(idisp, icomind, tol=itol)
                            if isCheck:
                                for jj in range(avSPs[j].nSP):
                                    jsp = avSPs[j].SPlist[jj]
                                    jdisp = jsp.disps.copy()
                                    jdisp = jdisp[:, 0:javna]
                                    jtol = min(SPsett["ValidSPs"]["Tol4Disp"],
                                               jsp.dmax * SPsett["ValidSPs"]["R2Dmax4Tol"])
                                    jdispc, isCheck = check_oneside4dup(jdisp, jcomind, tol=jtol)
                                    if isCheck:
                                        thistol = min(itol, jtol)
                                        isDup = check_dup_dSPs(idispc, jdispc, tol=thistol)
                                    if isDup: break
                            if isDup: todel.append(ii)

                    if len(todel) > 0:
                        todel = list(set(todel))
                        df_tmp = avSPs[i].to_dataframe(iSPs=todel)
                        reasons = ["Dup"] * len(df_tmp)
                        df_tmp = avSPs[i].insert_reason_df(df_tmp, reasons)

                        for ii in sorted(todel, reverse=True):
                            del avSPs[i].SPlist[ii]
                        avSPs[i].nSP = len(avSPs[i].SPlist)
                        df_delete_this = pd.concat([df_delete_this, df_tmp], ignore_index=True)

    return avSPs, df_delete_this


def interplate_disp(local_coords, predisps, sel_disp, Tolerance=0.1):
    ##sel_disp is wrt local_coords
    ##when local_coords has predisps
    thiscoords = local_coords + predisps
    nout = thiscoords.shape[1]
    thisdisp = np.zeros((3, nout))
    navg = 8
    sqT = Tolerance * Tolerance
    for i in range(thiscoords.shape[1]):
        ixyz = np.array([[thiscoords[0][i]], [thiscoords[1][i]], [thiscoords[2][i]]])
        thisd = local_coords - ixyz
        thisdsq = np.sum(thisd * thisd, axis=0)
        ai = np.argsort(thisdsq, axis=0)
        if thisdsq[ai[0]] <= 4.0 * sqT:
            thisdisp[0][i] += sel_disp[0][ai[0]]
            thisdisp[1][i] += sel_disp[1][ai[0]]
            thisdisp[2][i] += sel_disp[2][ai[0]]
        else:
            ds = np.zeros(navg, dtype=float)
            for j in range(navg):
                ds[j] = 1.0 / np.sqrt(thisdsq[ai[j]])
            ds = np.divide(ds, np.sum(ds, axis=0))
            for j in range(navg):
                thisdisp[0][i] += sel_disp[0][ai[j]] * ds[j]
                thisdisp[1][i] += sel_disp[1][ai[j]] * ds[j]
                thisdisp[2][i] += sel_disp[2][ai[j]] * ds[j]
    return thisdisp


class Data_SPs:
    def __init__(
            self,
            istep,
            ndefects,
            nSP=0,
            df_SPs=None,
            idavs=[],
            disps=[],
            fdisps=[],
            AV_masks=None,
            llocalisp=None,
            localiav=None,
            localisp=None,
            ispstart=[0],
    ):
        self.istep = istep
        self.ndefects = ndefects
        self.nSP = nSP
        self.df_SPs = df_SPs
        self.idavs = idavs
        self.disps = disps
        self.fdisps = fdisps
        self.AV_masks = AV_masks
        self.llocalisp = llocalisp
        self.localiav = localiav
        self.localisp = localisp
        self.ispstart = ispstart

    def __str__(self):
        return "The KMC step of this data saddle points is ({}).".format(self.istep)

    def __repr__(self):
        return self.__str__()

    def initialization(self):
        self.nSP = 0
        self.df_SPs = []
        self.idavs = []
        self.disps = []
        self.fdisps = []
        self.AV_masks = np.zeros(self.ndefects, dtype=bool)
        self.llocalisp = [[] for _ in range(self.ndefects)]
        self.localiav = []
        self.localisp = np.array([], dtype=int)
        self.ispstart = [0]
        self.barriermin = 1000000.0

    def insert_AVSPs(self, thisSPS, idav):
        if thisSPS.nSP > 0:
            arraylist = thisSPS.init_compact_arrays()
            for i in range(thisSPS.nSP):
                arraylist = thisSPS.append_SP2compact_arrays(thisSPS.SPlist[i], arraylist)
                self.disps.append(thisSPS.SPlist[i].disp.astype("f4"))
                self.fdisps.append(thisSPS.SPlist[i].fdisp.astype("f4"))
            df_AVSP = thisSPS.get_compact_df(arraylist)
            self.df_SPs.append(df_AVSP[SP_DATA_HEADER])
            self.idavs.append(idav)
            self.AV_masks[idav] = True
            self.llocalisp[idav] = self.nSP + np.arange(thisSPS.nSP, dtype=int)
            self.localiav += [len(self.idavs) - 1] * thisSPS.nSP
            self.localisp = np.append(self.localisp, np.arange(thisSPS.nSP, dtype=int))
            self.nSP += thisSPS.nSP
            self.ispstart.append(self.nSP)
            if thisSPS.barriermin < self.barriermin: self.barriermin = thisSPS.barriermin
        else:
            df_AVSP = pd.DataFrame(columns=SP_COMPACT_HEADER)

        return self, df_AVSP

    def check_dup_avSP(self, idav, thisAVSP, de_neighbors, AVitags, SPsett):
        df_delete_this = pd.DataFrame(columns=SP_COMPACT_HEADER4Delete)
        if thisAVSP.nSP > 0:
            nlist = de_neighbors[idav]
            thismask = np.zeros(len(nlist), dtype=bool)
            thismask[self.AV_masks[nlist]] = True
            nlist = nlist[thismask]
            if len(nlist) == 0:
                pass
            else:
                todel = []
                iavna = min(thisAVSP.SPlist[0].disp.shape[1], SPsett["ValidSPs"]["NMax4Dup"])
                for jn in range(len(nlist)):
                    jdav = nlist[jn]
                    jav0 = self.llocalisp[jdav][0]
                    javna = min(self.disps[jav0].shape[1], SPsett["ValidSPs"]["NMax4Dup"])
                    thisnmin = max(SPsett["ValidSPs"]["NCommonMin"], int(min(iavna, javna) * 0.5))
                    ijcom, icomind, jcomind, isCommon = find_common_atoms_AVs(AVitags, idav, jdav, iavna, javna,
                                                                              ncommonmin=thisnmin)
                    if isCommon:
                        for i in range(thisAVSP.nSP):
                            isp = thisAVSP.SPlist[i]
                            idisp = isp.disp.copy()
                            idisp = idisp[:, 0:iavna]
                            isDup = False
                            itol = min(SPsett["ValidSPs"]["Tol4Disp"], isp.dmax * SPsett["ValidSPs"]["R2Dmax4Tol"])
                            idispc, isCheck = check_oneside4dup(idisp, icomind, tol=itol)
                            if isCheck:
                                for jj in self.llocalisp[jdav]:
                                    jdisp = self.disps[jj].copy()
                                    jdisp = jdisp.astype(float)
                                    javlocal = self.localiav[jj]
                                    jsplocal = self.localisp[jj]
                                    jdmax = self.df_SPs[javlocal].at[jsplocal, "dmax"]
                                    jtol = min(SPsett["ValidSPs"]["Tol4Disp"], jdmax * SPsett["ValidSPs"]["R2Dmax4Tol"])
                                    jdispc, isCheck = check_oneside4dup(jdisp, jcomind, tol=jtol)
                                    if isCheck:
                                        thistol = min(itol, jtol)
                                        isDup = check_dup_dSPs(idispc, jdispc, tol=thistol)
                                    if isDup: break
                            if isDup: todel.append(i)

                if len(todel) > 0:
                    todel = list(set(todel))
                    df_tmp = thisAVSP.to_dataframe(iSPs=todel)
                    reasons = ["Dup"] * len(df_tmp)
                    df_tmp = thisAVSP.insert_reason_df(df_tmp, reasons)
                    df_delete_this = pd.concat([df_delete_this, df_tmp], ignore_index=True)
                    for i in sorted(todel, reverse=True):
                        del thisAVSP.SPlist[i]
                    thisAVSP.nSP = len(thisAVSP.SPlist)

        return thisAVSP, df_delete_this

    def check_dup_avSPs(self, idav, thisAVSPs, de_neighbors, AVitags, SPsett):
        df_delete_this = pd.DataFrame(columns=SP_COMPACT_HEADER4Delete)
        for thisAVSP in thisAVSPs:
            thisAVSP, df_tmp = self.check_dup_avSP(idav, thisAVSP, de_neighbors, AVitags, SPsett)
            df_delete_this = pd.concat([df_delete_this, df_tmp], ignore_index=True)
        return df_delete_this

    def reorganization(self, to_delete_localisp):
        if len(to_delete_localisp) > 0:
            for isp in sorted(to_delete_localisp, reverse=True):
                iavlocal = self.localiav[isp]
                isplocal = self.localisp[isp]
                self.df_SPs[iavlocal].drop(isplocal, inplace=True)
                del self.disps[isp]
                del self.fdisps[isp]
            self.nSP = 0
            self.llocalisp = [[] for _ in range(self.ndefects)]
            self.localiav = []
            self.localisp = np.array([], dtype=int)
            self.ispstart = [0]
            self.barriermin = 1000000.0
            to_delete_av = []
            for i in range(len(self.df_SPs)):
                self.df_SPs[i].reset_index(inplace=True)
                thisnSP = len(self.df_SPs[i])
                if thisnSP == 0:
                    to_delete_av.append(i)
                else:
                    idav = self.df_SPs[i].at[0, "idav"]
                    self.idavs.append(idav)
                    self.llocalisp[idav] = self.nSP + np.arange(thisnSP, dtype=int)
                    self.localiav += [len(self.idavs) - 1] * thisnSP
                    self.localisp = np.append(self.localisp, np.arange(thisnSP, dtype=int))
                    self.nSP += thisnSP
                    self.ispstart.append(self.nSP)
                    barriermin = np.min(self.df_SPs[i]["barrier"].to_numpy())
                    if barriermin < self.barriermin: self.barriermin = barriermin

            for iav in sorted(to_delete_av, reverse=True):
                del self.df_SPs[iav]


class DefectBank:
    def __init__(
            self,
            id,
            atoms,
            disps,
            sch_symbol="C1",
            namax=70,
            namin=10,
            significant_figures=6,
    ):
        self.id = id
        self.atoms = atoms
        self.disps = disps
        self.sch_symbol = sch_symbol
        self.namax = namax
        self.namin = namin
        self.natoms = len(self.atoms)
        self.significant_figures = significant_figures

    def __str__(self):
        return "DefectBank is ({}).".format(self.id)

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
    def from_AV_SPs(cls, data, AV_SPs, id=None, sch_symbol="C1", scaling=1.0, namax=70, namin=10, Style='Type',
                    significant_figures=6):
        if data.nactive < namin:
            return False
        else:
            if not id: id = AV_SPs.idav
            thisna = min(namax, data.nactive)
            atoms = data.atoms.truncate(after=data.atoms.index[thisna - 1])
            atoms["x"] = atoms["x"].apply(lambda x: x / scaling)
            atoms["y"] = atoms["y"].apply(lambda x: x / scaling)
            atoms["z"] = atoms["z"].apply(lambda x: x / scaling)
            disps = []
            if Style.upper() == "TYPE":
                for i in range(AV_SPs.ntype):
                    j = AV_SPs.type_info[i][0]
                    disps.append(AV_SPs.SPlist[i].disp[:, 0:thisna])
            else:
                for i in range(AV_SPs.nSP):
                    disps.append(AV_SPs.SPlist[i].disp[:, 0:thisna])
            disps = np.array(disps) / scaling
            return (cls(id, atoms, disps, sch_symbol=sch_symbol, namax=namax, namin=namin,
                        significant_figures=significant_figures))

    @classmethod
    def from_files(cls, id, fheader, sch_symbol="C1", filepath=None, namax=70, namin=10, SortDisps=False,
                   significant_figures=6):
        if isinstance(filepath, str):
            pass
        else:
            filepath = os.getcwd()
        thisfn = fheader + "_basin_" + str(id) + "_" + sch_symbol + ".csv"
        thisfn = os.path.join(filepath, thisfn)
        thisna = 0
        if os.path.isfile(thisfn):
            atoms = pd.read_csv(thisfn)
            thisna = min(namax, len(atoms))
            atoms = atoms.truncate(after=atoms.index[thisna - 1])
            atoms = atoms[DEFECTBANK_ATOMS_HEADER]
        thisDB = False
        if thisna >= namin:
            thisfhead = fheader + "_b_" + str(id) + "_disp_"
            disps = []
            fids = []
            for file in os.listdir(filepath):
                if thisfhead in file:
                    thisdf = pd.read_csv(os.path.join(filepath, file))
                    thisdf = thisdf.truncate(after=thisdf.index[thisna - 1])
                    disps.append(np.vstack((thisdf["dx"].to_numpy(), thisdf["dy"].to_numpy(), thisdf["dz"].to_numpy())))
                    if SortDisps:
                        fid = file.replace(thisfhead, "")
                        fid = fid.replace(".csv", "")
                        fids.append(fid)
            if len(disps) > 0:
                disps = np.array(disps)
                if SortDisps:
                    ai = np.argsort(fids)
                    ai = np.expand_dims(ai, axis=1)
                    ai = np.expand_dims(ai, axis=2)
                    disps = np.take_along_axis(disps, ai, axis=0)
                thisDB = cls(id, atoms, disps, sch_symbol=sch_symbol, namax=namax, namin=namin,
                             significant_figures=significant_figures)
        return thisDB

    def is_same_structure(self, atoms_in, sch_symbol_in, scaling=1.0, ignore_type=False, tolerance=0.1):
        if sch_symbol_in == self.sch_symbol:
            isSame = True
            nacheck = min(len(atoms_in), self.namax, self.natoms)
            if nacheck < self.namin: isSame = False
        else:
            isSame = False
        if isSame:
            itmp = self.atoms.copy(deep=True)
            jtmp = atoms_in.copy(deep=True)
            itmp = itmp.truncate(after=itmp.index[nacheck - 1])
            jtmp = jtmp.truncate(after=jtmp.index[nacheck - 1])

            ixyzs = np.vstack((itmp['x'], itmp['y'], itmp['z'])) * scaling
            jxyzs = np.vstack((jtmp['x'], jtmp['y'], jtmp['z']))
            if ignore_type:
                diffs = jxyzs - ixyzs
            else:
                jtype = jtmp["type"].to_numpy().astype(int)
                jtype = jtype.reshape([1, nacheck])
                itype = itmp["type"].to_numpy().astype(int)
                itype = itype.reshape([1, nacheck])
                diffs = jxyzs * jtype - ixyzs * itype
            maxd = np.max(np.absolute(diffs))
            if maxd > tolerance: isSame = False
        return isSame

    def load_disps(self, scaling=1.0, LoadRatio=1.0):
        if scaling == 1.0 and LoadRatio == 1.0:
            return self.disps
        else:
            disps = []
            for i in range(len(self.disps)):
                disps.append(self.disps[i] * scaling * LoadRatio)
            return disps

    def to_files(self, fheader, filepath=None, OutIndex=True):
        float_format = '%.' + str(self.significant_figures) + 'f'
        atoms = self.atoms[DEFECTBANK_ATOMS_HEADER]
        thisfn1 = fheader + "_basin_" + str(self.id) + "_" + self.sch_symbol + ".csv"
        if isinstance(filepath, str): thisfn1 = os.path.join(filepath, thisfn1)
        atoms.to_csv(thisfn1, index=OutIndex, float_format=float_format)
        for i in range(len(self.disps)):
            thisdf = pd.DataFrame(self.disps[i].T, columns=DEFECTBANK_DISPS_HEADER)
            thisfn = fheader + "_b_" + str(self.id) + "_disp_" + str(i) + ".csv"
            if isinstance(filepath, str): thisfn = os.path.join(filepath, thisfn)
            thisdf.to_csv(thisfn, index=OutIndex, float_format=float_format)
