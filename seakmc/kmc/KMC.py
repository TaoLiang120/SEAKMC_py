import numpy as np
import scipy.linalg
import pandas as pd
import copy

from seakmc.input.Input import SP_KMC_HEADER, SP_DATA_HEADER
from seakmc.input.Input import SP_KMC_SELECTED_HEADER, NENTRY_SELECTED_HEADER, NDISPARRAY, SEQUENCE_DISPARRAY
from seakmc.core.util import mat_mag

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"

KB = 8.617333262145e-5


class Basin:
    def __init__(
            self,
            id,
            istep,
            data,
            AVitags,
            DataSPs,
            sett,
            VerySmallNumber=1.0e-20,
    ):
        self.id = id
        self.istep = istep
        self.data = data
        self.AVitags = AVitags
        self.sett = sett
        self.VerySmallNumber = VerySmallNumber
        self.temp = self.sett["Temp"]

        self.nAV = len(DataSPs.idavs)
        self.nSP = DataSPs.nSP
        if self.nSP <= 0:
            raise ValueError("No saddle point in KMC step!")

        self.localiba = np.array([self.id] * self.nSP, dtype=int)
        self.localiav = np.array(DataSPs.localiav, dtype=int)
        self.localisp = np.arange(self.nSP, dtype=int)
        self.ispstart = np.array(DataSPs.ispstart, dtype=int)
        self.idavs = np.array(DataSPs.idavs, dtype=int)
        ##self.freqs = np.concatenate(DataSPs.freqs).flatten()
        self.barrs = np.array([])
        self.freqs = np.array([])
        self.bias = np.array([])
        self.meanpref = 0.0
        for iav in range(self.nAV):
            prefs = DataSPs.df_SPs[iav]["prefactor"].to_numpy()
            barrs = DataSPs.df_SPs[iav]["barrier"].to_numpy()
            thisbias = DataSPs.df_SPs[iav]["ebias"].to_numpy()
            thisfreqs = np.multiply(prefs, np.exp(np.divide(-barrs, KB * self.temp)))
            self.barrs = np.append(self.barrs, barrs)
            self.bias = np.append(self.bias, thisbias)
            self.freqs = np.append(self.freqs, thisfreqs)
            self.meanpref += np.sum(prefs)
        self.sumfreq = np.sum(self.freqs)
        if self.sumfreq <= self.VerySmallNumber: self.sumfreq = self.VerySmallNumber
        self.disps = copy.deepcopy(DataSPs.disps)
        self.fdisps = copy.deepcopy(DataSPs.fdisps)
        self.meanpref = self.meanpref / self.nSP
        if self.sett["Sorting"]:
            inds = np.flip(np.argsort(self.freqs))
            self.freqs = np.take_along_axis(self.freqs, inds, axis=0)
            self.localiav = np.take_along_axis(self.localiav, inds, axis=0)
            self.localisp = np.take_along_axis(self.localisp, inds, axis=0)

    def __str__(self):
        return "Basin id is ({}).".format(self.id)

    def __repr__(self):
        return self.__str__()


class SuperBasin:
    def __init__(
            self,
            Basin_list,
            temp,
    ):
        self.Basin_list = Basin_list
        self.temp = temp

    def initialization(self):
        self.now = 0
        self.fbarr = 0.0
        self.bbarr = 0.0
        self.frate = 1.0
        self.brate = 1.0
        self.nBasin = len(self.Basin_list)
        self.meantimes = np.array([])
        self.transit = np.eye(1) - np.eye(1)
        self.mrtimes = np.zeros(1)
        self.sumtime = np.sum(self.mrtimes)
        self.iba_sels = []
        self.iav_sels = []
        self.isp_sels = []
        self.isRelease = False
        self.nSP = 0
        self.meanpref = 0.0

    def update_transition_matrix(self, ibl, ibn, fr, br, nsumfreq):
        if isinstance(fr, float) or isinstance(fr, int):
            self.transit[ibn][ibl] = fr / self.Basin_list[ibl].sumfreq
        if isinstance(br, float) or isinstance(br, int):
            self.transit[ibl][ibn] = br / nsumfreq

    def add_transition_matrix(self, ibl, ibn, fr):
        n = self.transit.shape[0]
        a = np.zeros((n, 1))
        self.transit = np.append(self.transit, a, axis=1)
        n = self.transit.shape[1]
        a = np.zeros((1, n))
        self.transit = np.append(self.transit, a, axis=0)
        if isinstance(fr, float) or isinstance(fr, int):
            self.transit[ibn][ibl] = fr / self.Basin_list[ibl].sumfreq
        #if isinstance(br, float) or isinstance(br, int):
        #    self.transit[ibl][ibn] = br/self.Basin_list[ibn].sumfreq

    def compute_mean_residence_time(self, idnow):
        n = self.transit.shape[0]
        inDelta = np.zeros(n)
        inDelta[idnow] = 1.0
        ainv = scipy.linalg.inv(np.eye(n) - self.transit)
        self.mrtime = np.multiply(self.meantimes, np.dot(inDelta, ainv.T))
        self.sumtime = np.sum(self.mrtime)
        '''
        print("inside mean residence time")
        print(f"transit:{self.transit}")
        print(f"indelta:{inDelta} occu:{np.dot(inDelta, ainv.T)}")
        print(f"mrtime:{self.mrtime} sumtime:{self.sumtime}")
        print("========")
        '''

    def insert_Basin(self, thisBasin):
        self.Basin_list.append(thisBasin)
        self.meantimes = np.append(self.meantimes, [1.0 / thisBasin.sumfreq])
        self.meanpref = self.meanpref * self.nSP + thisBasin.meanpref * thisBasin.nSP
        self.nSP += thisBasin.nSP
        self.meanpref = self.meanpref / self.nSP
        self.now = thisBasin.id
        self.nBasin += 1

    def remove_backward_SP(self, thisBasin, Tol4Disp=0.1, Tol4Barr=0.02, ForwardRate="NA", Handle_no_Backward="OUT"):
        if self.nBasin < 1:
            return True
        else:
            iblast = self.iba_sels[len(self.iba_sels) - 1]
            iavlast = self.iav_sels[len(self.iba_sels) - 1]
            isplast = self.isp_sels[len(self.isp_sels) - 1]
            blast = self.Basin_list[iblast]
            idavlast = blast.idavs[iavlast]
            itaglast0 = blast.AVitags[idavlast][0]
            lastatomid = int(blast.data.atoms.index[itaglast0])
            lasttags = blast.data.atoms.index.to_numpy()
            lastatomarray = blast.data.atoms_to_array(blast.data.atoms, OutIndex=True)

            tags = thisBasin.data.atoms.index.to_numpy()
            thisatomarray = thisBasin.data.atoms_to_array(thisBasin.data.atoms, OutIndex=True)
            potAVs = np.zeros(thisBasin.nAV, dtype=bool)
            isBackward = False
            for iav in range(thisBasin.nAV):
                if not potAVs[iav] and not isBackward:
                    idav = thisBasin.idavs[iav]
                    itags = thisBasin.AVitags[idav]
                    ispstart = thisBasin.ispstart[iav]
                    ispend = thisBasin.ispstart[iav + 1]
                    thisna = thisBasin.fdisps[ispstart].shape[1]
                    itags = itags[0:thisna]
                    thistags = tags[itags]
                    try:
                        idx = np.where(thistags == lastatomid)
                        idx = idx[0][0]
                    except:
                        idx = -1
                    if idx != -1:
                        commons, lastind, thisind = np.intersect1d(lasttags, thistags, return_indices=True)
                        lastarray = lastatomarray[lastind]
                        lastxyzns = np.vstack((lastarray['x'], lastarray['y'], lastarray['z']))
                        lastxyzns = np.dot(blast.data.box.inv_matrix.T, lastxyzns)
                        thisarray = thisatomarray[itags]
                        thisarray = thisarray[thisind]
                        thisxyzs = np.vstack((thisarray['x'], thisarray['y'], thisarray['z']))
                        for isp in range(ispstart, ispend):
                            if isp in thisBasin.localisp:
                                thisbarr = thisBasin.barrs[isp]
                                if abs(thisbarr - self.bbarr) >= Tol4Barr:
                                    pass
                                else:
                                    thisdisp = thisBasin.fdisps[isp][:, 0:thisna]
                                    thisdisp = thisdisp[:, thisind]
                                    thisdisp = thisdisp.astype(float)
                                    thisxyzns = thisxyzs + thisdisp
                                    thisxyzns = np.dot(thisBasin.data.box.inv_matrix.T, thisxyzns)
                                    thisdiff = thisxyzns - lastxyzns - np.around(thisxyzns - lastxyzns)
                                    diffs = np.dot(thisdiff.T, thisBasin.data.box.matrix)
                                    maxdiff = np.max(np.absolute(diffs))
                                    if maxdiff <= Tol4Disp:
                                        isBackward = True
                                        ind = np.where(thisBasin.localisp == isp)
                                        ind = ind[0][0]
                                        br = thisBasin.freqs[ind]
                                        thisBasin.freqs = np.delete(thisBasin.freqs, ind)
                                        thisBasin.localiba = np.delete(thisBasin.localiba, ind)
                                        thisBasin.localiav = np.delete(thisBasin.localiav, ind)
                                        thisBasin.localisp = np.delete(thisBasin.localisp, ind)
                                        thisBasin.nSP -= 1
                                        self.update_transition_matrix(iblast, thisBasin.id, ForwardRate, br,
                                                                      thisBasin.sumfreq)
                            else:
                                pass
                    potAVs[iav] = True

            if not isBackward:
                if "IN" in Handle_no_Backward.upper():
                    ## backward rate may diff from self.brate since meanpref might be different
                    br = thisBasin.meanpref * np.exp(np.divide(-self.bbarr, KB * self.temp))
                    if thisBasin.id < self.nBasin:
                        nsumfreq = self.Basin_list[thisBasin.id].sumfreq + br
                        r = self.Basin_list[thisBasin.id].sumfreq / nsumfreq
                        self.Basin_list[thisBasin.id].sumfreq = nsumfreq
                        self.meantimes[thisBasin.id] = 1.0 / self.Basin_list[thisBasin.id].sumfreq
                        for i in range(self.transit.shape[0]):
                            self.transit[i][thisBasin.id] *= r
                    else:
                        nsumfreq = thisBasin.sumfreq + br
                    self.update_transition_matrix(iblast, thisBasin.id, ForwardRate, br, nsumfreq)
                else:
                    self.Basin_list = []
                    self.initialization()
            return isBackward

    def update_info(self, iba, isp, fb, bb):
        ind = np.where(self.Basin_list[iba].localisp == isp)
        ind = ind[0][0]
        self.Basin_list[iba].freqs = np.delete(self.Basin_list[iba].freqs, ind)
        self.Basin_list[iba].localiba = np.delete(self.Basin_list[iba].localiba, ind)
        self.Basin_list[iba].localiav = np.delete(self.Basin_list[iba].localiav, ind)
        self.Basin_list[iba].localisp = np.delete(self.Basin_list[iba].localisp, ind)
        self.fbarr = fb
        self.bbarr = bb
        self.frate = self.Basin_list[iba].meanpref * np.exp(np.divide(-fb, KB * self.temp))
        self.brate = self.Basin_list[iba].meanpref * np.exp(np.divide(-bb, KB * self.temp))
        self.now = iba
        self.Basin_list[iba].nSP -= 1

    def check_isVisit(self, iba, iav, isp, Tol4Disp=0.1, Tol4Barr=0.03):
        isVisit = False
        if self.nBasin <= 2:
            pass
        else:
            thisBasin = self.Basin_list[iba]
            idav = thisBasin.idavs[iav]
            thisdata = copy.deepcopy(thisBasin.data)
            thisdata.update_coords_from_disps(idav, (thisBasin.fdisps[isp]).astype(float), thisBasin.AVitags,
                                              Reset_MolID=False)
            thistags = thisdata.atoms.index.to_numpy()
            thisxyzns = np.vstack((thisdata.atoms['x'], thisdata.atoms['y'], thisdata.atoms['z']))
            thisxyzns = np.dot(thisdata.box.inv_matrix.T, thisxyzns)
            thisdata = None
            for ib in range(self.nBasin):
                if isVisit:
                    pass
                else:
                    if iba == ib:
                        pass
                    else:
                        bas = self.Basin_list[ib]
                        t = bas.data.atoms.index.to_numpy()
                        commons, cind, thisind = np.intersect1d(t, thistags, return_indices=True)
                        cxyzns = np.vstack((bas.data.atoms['x'], bas.data.atoms['y'], bas.data.atoms['z']))
                        cxyzns = np.dot(bas.data.box.inv_matrix.T, cxyzns)
                        cxyzns = cxyzns[:, cind]
                        txyzns = thisxyzns[:, thisind]

                        thisdiff = txyzns - cxyzns - np.around(txyzns - cxyzns)
                        diffs = np.sum(np.dot(thisdiff.T, thisBasin.data.box.matrix) ** 2, axis=1)
                        maxdiff = np.sqrt(np.max(diffs))
                        if maxdiff <= Tol4Disp:
                            self.remove_backward_SP(bas, Tol4Disp=Tol4Disp, Tol4Barr=Tol4Barr, ForwardRate=self.frate,
                                                    Handle_no_Backward="STAYIN")
                            self.now = bas.id
                            isVisit = True

            return isVisit

    def prepare_next(self, sett):
        if sett["AccStyle"][0:3].upper() == "MRM":
            if isinstance(sett["NMaxBasin"], int):
                if self.nBasin >= sett["NMaxBasin"]: self.isRelease = True
            if self.isRelease:
                self.Basin_list = []
                self.initialization()
            else:
                self.add_transition_matrix(self.now, self.nBasin, self.frate)
        else:
            self.Basin_list = []
            self.initialization()


class DataKMC:
    def __init__(
            self,
            id,
            sett,
            float_precision=3,
    ):
        self.id = id
        self.sett = sett
        self.float_precision = float_precision
        self.temp = self.sett["Temp"]

    def initialization(self, thisSuperBasin, thisBasin):
        if thisBasin.nSP <= 1:
            thisSuperBasin.Basin_list = []
            thisSuperBasin.nBasin = 0
            thisSuperBasin.initialization()
            thisBasin.id = 0
            thisBasin.localiba = np.array([thisBasin.id] * thisBasin.nSP, dtype=int)

        isBackward = thisSuperBasin.remove_backward_SP(thisBasin,
                                                       Tol4Disp=self.sett["Tol4Disp"], Tol4Barr=self.sett["Tol4Barr"],
                                                       ForwardRate="NA",
                                                       Handle_no_Backward=self.sett["Handle_no_Backward"])
        if not isBackward:
            if thisSuperBasin.nBasin == 0:
                thisBasin.id = 0
                thisBasin.localiba = np.array([thisBasin.id] * thisBasin.nSP, dtype=int)
            else:
                br = thisBasin.meanpref * np.exp(np.divide(-thisSuperBasin.bbarr, KB * thisSuperBasin.temp))
                thisBasin.sumfreq += br

        self.thisprobs = 100.0 * thisBasin.freqs / thisBasin.sumfreq
        thisSuperBasin.insert_Basin(thisBasin)

        self.nBasin = len(thisSuperBasin.Basin_list)
        self.localiba = np.array([], dtype=int)
        self.localiav = np.array([], dtype=int)
        self.localisp = np.array([], dtype=int)
        ##self.freqs = np.concatenate(DataSPs.freqs).flatten()
        self.freqs = np.array([])
        self.sumfreq = 0.0
        self.nSP = 0
        for ib in range(self.nBasin):
            self.localiba = np.append(self.localiba, thisSuperBasin.Basin_list[ib].localiba)
            self.localiav = np.append(self.localiav, thisSuperBasin.Basin_list[ib].localiav)
            self.localisp = np.append(self.localisp, thisSuperBasin.Basin_list[ib].localisp)
            self.freqs = np.append(self.freqs, thisSuperBasin.Basin_list[ib].freqs)
            self.sumfreq += thisSuperBasin.Basin_list[ib].sumfreq
            self.nSP += thisSuperBasin.Basin_list[ib].nSP

        self.isps = np.arange(self.nSP, dtype=int)
        self.idnow = thisBasin.id
        self.iba_sels = []
        self.iav_sels = []
        self.isp_sels = []
        self.gsp_sels = []
        self.isels = []
        self.timeelapse = 0.0
        self.iter = 0
        self.isValid = True

    def __str__(self):
        return "KMCstep id is ({}).".format(self.id)

    def __repr__(self):
        return self.__str__()

    def reorganization(self, thisSuperBasin):
        self.nBasin = len(thisSuperBasin.Basin_list)
        self.localiba = np.array([], dtype=int)
        self.localiav = np.array([], dtype=int)
        self.localisp = np.array([], dtype=int)
        ##self.freqs = np.concatenate(DataSPs.freqs).flatten()
        self.freqs = np.array([])
        self.sumfreq = 0.0
        self.nSP = 0
        for ib in range(self.nBasin):
            self.localiba = np.append(self.localiba, thisSuperBasin.Basin_list[ib].localiba)
            self.localiav = np.append(self.localiav, thisSuperBasin.Basin_list[ib].localiav)
            self.localisp = np.append(self.localisp, thisSuperBasin.Basin_list[ib].localisp)
            self.freqs = np.append(self.freqs, thisSuperBasin.Basin_list[ib].freqs)
            self.sumfreq += thisSuperBasin.Basin_list[ib].sumfreq
            self.nSP += thisSuperBasin.Basin_list[ib].nSP
        self.isps = np.arange(self.nSP, dtype=int)

    def get_timestep(self, thisSuperBasin):
        rnd = np.random.rand(1)
        if self.sett["Temp"] == self.sett["Temp4Time"]:
            thistime = -np.log(rnd[0]) * thisSuperBasin.sumtime
        else:
            logtime = np.log(thisSuperBasin.sumtime)
            thisbarr = (np.log(thisSuperBasin.meanpref) + logtime) * KB * self.temp
            thissumtime = 1.0 / (np.exp(-thisbarr / (KB * self.sett["Temp4Time"])) * thisSuperBasin.meanpref)
            thistime = -np.log(rnd[0]) * thissumtime
        #thistime = 1.0*thisSuperBasin.sumtime
        return thistime

    def get_probs(self, thisSuperBasin):
        #thisfreqs = self.freqs*np.concatenate(self.sumfreqs).flatten()/self.sumfreq
        thisSuperBasin.compute_mean_residence_time(self.idnow)

        #print("inside get_probs")
        #print(f"before freqs: {self.freqs}")

        thisfreqs = self.freqs * thisSuperBasin.mrtime[self.localiba] / thisSuperBasin.sumtime
        self.sumfreq = np.sum(thisfreqs)
        self.probs = np.hstack((np.array([0]), np.cumsum(thisfreqs / self.sumfreq)))

        #print(f"after freqs:{thisfreqs}")
        #print(f"probs: {self.probs}")
        #print("+++++++++++++++++++++++++")

    def select_event(self, thisSuperBasin):
        self.get_probs(thisSuperBasin)
        rnd = np.random.rand(1)
        sel = np.where(self.probs < rnd[0])
        if rnd[0] == 0.0:
            isel = 0
        else:
            isel = sel[0][sel[0].shape[0] - 1]
        return isel

    def update_information(self, isel, thisSuperBasin):
        iba = self.localiba[isel]
        iav = self.localiav[isel]
        isp = self.localisp[isel]
        self.iba_sels.append(iba)
        self.iav_sels.append(iav)
        self.isp_sels.append(isp)
        if iba == self.nBasin - 1: self.isels.append(isp)
        self.freqs = np.delete(self.freqs, isel)
        self.localiba = np.delete(self.localiba, isel)
        self.localiav = np.delete(self.localiav, isel)
        self.localisp = np.delete(self.localisp, isel)
        self.isps = np.delete(self.isps, isel)
        self.nSP -= 1

        thisSuperBasin.iba_sels.append(iba)
        thisSuperBasin.iav_sels.append(iav)
        thisSuperBasin.isp_sels.append(isp)
        fb = thisSuperBasin.Basin_list[iba].barrs[isp]
        bb = fb - thisSuperBasin.Basin_list[iba].bias[isp]
        thisSuperBasin.isRelease = True
        self.isValid = False
        if self.sett["AccStyle"][0:3].upper() == "MRM":
            if fb < self.sett["EnCut4Transient"] and bb < self.sett["EnCut4Transient"]:
                thisSuperBasin.isRelease = False
                thisSuperBasin.update_info(iba, isp, fb, bb)
                isVisit = thisSuperBasin.check_isVisit(iba, iav, isp,
                                                       Tol4Disp=self.sett["Tol4Disp"], Tol4Barr=self.sett["Tol4Barr"])
                if isVisit:
                    self.isValid = True
                    self.idnow = thisSuperBasin.now
                    self.reorganization(thisSuperBasin)
            else:
                pass
        else:
            pass

    def run_KMC(self, thisSuperBasin):
        while self.isValid:
            isel = self.select_event(thisSuperBasin)
            self.gsp_sels.append(isel)
            self.update_information(isel, thisSuperBasin)
            self.timeelapse += self.get_timestep(thisSuperBasin)
            self.iter += 1
        return self.timeelapse

    def update_last_defect_center(self, thisSuperBasin):
        iba = self.iba_sels[len(self.iba_sels) - 1]
        thisBasin = thisSuperBasin.Basin_list[iba]
        return thisBasin.data.de_center

    def update_coords4relaxation(self, thisSuperBasin):
        iba = self.iba_sels[len(self.iba_sels) - 1]
        iav = self.iav_sels[len(self.iav_sels) - 1]
        isp = self.isp_sels[len(self.isp_sels) - 1]
        thisBasin = thisSuperBasin.Basin_list[iba]
        idav = thisBasin.idavs[iav]
        thisdata = copy.deepcopy(thisBasin.data)
        if self.sett["DispStyle"][0:2].upper() == "FI":
            thisdata.update_coords_from_disps(idav, thisBasin.fdisps[isp].astype(float), thisBasin.AVitags,
                                              Reset_MolID=False)
        else:
            thisdata.update_coords_from_disps(idav, thisBasin.disps[isp].astype(float), thisBasin.AVitags,
                                              Reset_MolID=False)
        return thisdata

    def Prob_to_file(self, filename, DataSPs, DetailOut=False, fsel="SP_Details.csv", SPs4Detail="Auto",
                     VerySmallNumber=1.0e-20):
        float_format = '%.' + str(self.float_precision) + 'f'

        def init_disp_details():
            arraylist = []
            for i in range(NDISPARRAY * NENTRY_SELECTED_HEADER):
                a = np.array([])
                arraylist.append(a)
            return arraylist

        def get_disp_details(arraylist, isp, DataSPs, df_SPs, DispStyle="SP"):
            ind = SEQUENCE_DISPARRAY.index(DispStyle)
            istart = NENTRY_SELECTED_HEADER * ind
            if DispStyle == "FI":
                x = DataSPs.fdisps[isp].copy()
            elif DispStyle == "FS":
                x = DataSPs.fdisps[isp].copy() - DataSPs.disps[isp]
            else:
                x = DataSPs.disps[isp].copy()

            x2 = x * x
            dtot = np.sum(np.sqrt(np.sum(x2, axis=0)))
            dmags = np.sqrt(np.sum(x2, axis=1))
            xa = np.sum(x2, axis=0)
            idmax = np.argmax(xa)
            dmaxs = np.array([x[0][idmax], x[1][idmax], x[2][idmax]])
            xa = np.sqrt(x2)
            vmaxs = np.max(xa, axis=1)
            dsums = np.sum(x, axis=1)
            adsums = np.sum(np.absolute(x), axis=1)

            arraylist[istart] = np.append(arraylist[istart], dtot)
            arraylist[istart + 1] = np.append(arraylist[istart + 1], mat_mag(dmags))
            arraylist[istart + 2] = np.append(arraylist[istart + 2], mat_mag(dmaxs))
            arraylist[istart + 3] = np.append(arraylist[istart + 3], mat_mag(dsums))
            arraylist[istart + 4] = np.append(arraylist[istart + 4], mat_mag(adsums))

            arraylist[istart + 5] = np.append(arraylist[istart + 5], dmags[0])
            arraylist[istart + 6] = np.append(arraylist[istart + 6], dmags[1])
            arraylist[istart + 7] = np.append(arraylist[istart + 7], dmags[2])
            arraylist[istart + 8] = np.append(arraylist[istart + 8], dmaxs[0])
            arraylist[istart + 9] = np.append(arraylist[istart + 9], dmaxs[1])
            arraylist[istart + 10] = np.append(arraylist[istart + 10], dmaxs[2])
            arraylist[istart + 11] = np.append(arraylist[istart + 11], vmaxs[0])
            arraylist[istart + 12] = np.append(arraylist[istart + 12], vmaxs[1])
            arraylist[istart + 13] = np.append(arraylist[istart + 13], vmaxs[2])
            arraylist[istart + 14] = np.append(arraylist[istart + 14], dsums[0])
            arraylist[istart + 15] = np.append(arraylist[istart + 15], dsums[1])
            arraylist[istart + 16] = np.append(arraylist[istart + 16], dsums[2])
            arraylist[istart + 17] = np.append(arraylist[istart + 17], adsums[0])
            arraylist[istart + 18] = np.append(arraylist[istart + 18], adsums[1])
            arraylist[istart + 19] = np.append(arraylist[istart + 19], adsums[2])
            return arraylist

        df_SPs = pd.DataFrame(columns=SP_DATA_HEADER)
        for i in range(len(DataSPs.df_SPs)):
            if len(DataSPs.df_SPs[i]) > 0:
                if len(df_SPs) == 0:
                    df_SPs = DataSPs.df_SPs[i].copy(deep=True)
                    df_SPs = df_SPs.reset_index(drop=True)
                else:
                    df_SPs = pd.concat([df_SPs, DataSPs.df_SPs[i]], ignore_index=True)
        idavs = df_SPs["idav"].to_numpy().astype(int)
        barriers = df_SPs["barrier"].to_numpy()
        prefactors = df_SPs["prefactor"].to_numpy()
        biass = df_SPs["ebias"].to_numpy()
        dmags = df_SPs["dmag"].to_numpy()
        fdmags = df_SPs["dmagfi"].to_numpy()
        dmaxs = df_SPs["dmax"].to_numpy()
        fdmaxs = df_SPs["dmaxfi"].to_numpy()
        dsums = df_SPs["dsum"].to_numpy()
        fdsums = df_SPs["dsumfi"].to_numpy()
        rdsums = df_SPs["dsum"].to_numpy() / (df_SPs["adsum"].to_numpy() + VerySmallNumber)
        frdsums = df_SPs["dsumfi"].to_numpy() / (df_SPs["adsumfi"].to_numpy() + VerySmallNumber)
        isConns = df_SPs["isConnect"].to_numpy().astype(int)
        isSels = np.zeros(len(barriers), dtype=int)
        isSels[np.array(self.isels, dtype=int)] = 1

        marray = zip(idavs, barriers, prefactors, biass,
                     dmags, dmaxs, dsums, rdsums,
                     fdmags, fdmaxs, fdsums, frdsums,
                     isConns, self.thisprobs, isSels)

        array = list(tuple(marray))
        df = pd.DataFrame(array, columns=SP_KMC_HEADER)
        df.to_csv(filename, float_format=float_format)

        if DetailOut:
            if SPs4Detail[0:3].upper() == "ALL":
                isels = np.arange(len(df_SPs), dtype=int)
            else:
                isels = self.isels.copy()

            isps = np.array([], dtype=int)
            barriers = np.array([])
            prefactors = np.array([])
            biass = np.array([])
            arraylist = init_disp_details()
            for i in range(len(isels)):
                isp = isels[i]
                isps = np.append(isps, isp)
                barriers = np.append(barriers, df_SPs.at[isp, "barrier"])
                prefactors = np.append(prefactors, df_SPs.at[isp, "prefactor"])
                biass = np.append(biass, df_SPs.at[isp, "ebias"])

                arraylist = get_disp_details(arraylist, isp, DataSPs, df_SPs, DispStyle="SP")
                arraylist = get_disp_details(arraylist, isp, DataSPs, df_SPs, DispStyle="FS")
                arraylist = get_disp_details(arraylist, isp, DataSPs, df_SPs, DispStyle="FI")

            marray = zip(isps, barriers, prefactors, biass,
                         arraylist[0], arraylist[1], arraylist[2], arraylist[3], arraylist[4],
                         arraylist[5], arraylist[6], arraylist[7], arraylist[8], arraylist[9],
                         arraylist[10], arraylist[11], arraylist[12], arraylist[13], arraylist[14],
                         arraylist[15], arraylist[16], arraylist[17], arraylist[18], arraylist[19],
                         arraylist[20], arraylist[21], arraylist[22], arraylist[23], arraylist[24],
                         arraylist[25], arraylist[26], arraylist[27], arraylist[28], arraylist[29],
                         arraylist[30], arraylist[31], arraylist[32], arraylist[33], arraylist[34],
                         arraylist[35], arraylist[36], arraylist[37], arraylist[38], arraylist[39],
                         arraylist[40], arraylist[41], arraylist[42], arraylist[43], arraylist[44],
                         arraylist[45], arraylist[46], arraylist[47], arraylist[48], arraylist[49],
                         arraylist[50], arraylist[51], arraylist[52], arraylist[53], arraylist[54],
                         arraylist[55], arraylist[56], arraylist[57], arraylist[58], arraylist[59])

            array = list(tuple(marray))
            df = pd.DataFrame(array, columns=SP_KMC_SELECTED_HEADER)
            df.to_csv(fsel, float_format=float_format)
