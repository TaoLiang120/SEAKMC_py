import os
import time
import copy
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

import seakmc.process.DataDyn as mydatadyn
from seakmc.input.Input import SP_COMPACT_HEADER4Delete, SP_DATA_HEADER
from seakmc.kmc.KMC import SuperBasin
from seakmc.spsearch.SaddlePoints import Data_SPs
from seakmc.general.General import DFWriter
import seakmc.datasps.PreSPS as preSPS
import seakmc.datasps.DataSPS as dataSPS
from seakmc.kmc.KMC import Basin

KB = 8.617333262145e-5
class TrialDisp2Basin:
    def __init__(self, seakmcdata, displacement, itrial, Eground=0.0, key="displacement"):
        self.seakmcdata = seakmcdata
        self.displacement = displacement
        self.itrial = itrial
        self.key = self.key
        self.export = {self.key: self.displacement}
        self.thisdata = copy.deepcopy(self.seakmcdata)
        self.Eground = Eground
        self.timeelapse = 0.0
        self.one_over_freq = 0.0
        self.equi_barr = 0.0
        self.meanpref = 10
        self.multiply_factor = 1000000

    def relax_basin(self, force_evaluator, LogWriter, ntask_tot=1, nproc_task=1):
        ntask_tot = 1
        nproc_task = 1
        [Eground, relaxed_coords, isValid, errormsg] = mydatadyn.data_dynamics("DATATDB", force_evaluator,
                                                                               self.thisdata, ntask_tot,
                                                                               nactive=self.thisdata.natoms,
                                                                               nproc_task=nproc_task, thisExports=self.export)
        self.Eground = Eground
        if not isValid:
            LogWriter.write_data(errormsg)
            quit()

    def run_seakmc(self, istep, thissett, object_dict):
        out_paths = object_dict['out_paths']
        LogWriter = object_dict['LogWriter']
        thisSummary = object_dict['thisSummary']
        THIS_PATH = out_paths[-1]
        SPOutpath = os.path.join(THIS_PATH, "SPOut")
        thisDFWriter = DFWriter(OutPath=SPOutpath, WriteSPs=False)

        istep_this = (istep + 1) * self.multiply_factor + self.itrial + 1

        #initializing
        thisExports = thisSummary.export_dict
        simulation_time = 0.0
        thisSuperBasin = SuperBasin([], thissett.kinetic_MC["Temp"])
        thisSuperBasin.initialization()
        DefectBank_list = []
        if thissett.defect_bank["LoadDB"]: DefectBank_list = preSPS.load_DefectBanks(
            thissett.defect_bank, out_paths[2], significant_figures=thissett.system["significant_figures"])

        if isinstance(thissett.active_volume["DefectCenter4RT_SetMolID"], list):
            last_de_center = thissett.active_volume["DefectCenter4RT_SetMolID"]
        else:
            last_de_center = None


        #### start saddle point search
        tickmc = time.time()
        thisDFWriter.init_deleted_SPs(istep_this)
        thisDFWriter.init_SPs(istep_this)

        logstr = f"istep KMC: {istep} itrial: {self.itrial}"
        logstr += "\n" + f"pseudo_step KMC: {istep_this}"
        LogWriter.write_data(logstr)

        self.thisdata.get_defects(LogWriter, last_de_center=last_de_center)
        emptya = np.array([], dtype=int)
        AVitags = [emptya for i in range(self.thisdata.ndefects)]
        DataSPs = Data_SPs(istep_this, self.thisdata.ndefects)
        DataSPs.initialization()
        df_delete_SPs = pd.DataFrame(columns=SP_COMPACT_HEADER4Delete)
        undo_idavs = np.arange(self.thisdata.ndefects, dtype=int)
        finished_AVs = 0
        logstr = (f"The ground energy is {round(self.Eground, thissett.system['float_precision'])} eV at"
                  f" {istep_this} KMC pseudo_step !")
        logstr += "\n" + f"There are {self.thisdata.ndefects} defects (active volumes) in data at {istep_this} KMC pseudo_step!"
        logstr += "\n" + (f"The fractional coords of the defect center are "
                          f"{np.around(self.thisdata.de_center, decimals=thissett.system['float_precision'])} at"
                          f" {istep_this} KMC pseudo_step!")
        logstr += "\n"
        LogWriter.write_data(logstr)

        self.thisdata, DataSPs, AVitags = dataSPS.data_find_saddlepoints(istep_this, thissett, self.thisdata, DefectBank_list,
                                                                      thisSuperBasin, self.Eground,
                                                                      DataSPs, AVitags, df_delete_SPs, undo_idavs,
                                                                      finished_AVs, simulation_time,
                                                                      thisDFWriter, object_dict)

        self.thisdata.to_atom_style()
        self.thisdata.velocities = None
        self.thisdata.defects = None
        self.thisdata.def_atoms = []
        self.thisdata.atoms_ghost = None
        self.thisdata.natoms_ghost = 0

        os.chdir(THIS_PATH)


        thisBasin = Basin(len(thisSuperBasin.Basin_list), istep_this, self.thisdata, AVitags, DataSPs, thissett.kinetic_MC,
                          VerySmallNumber=thissett.system["VerySmallNumber"])
        self.timeelapse = thisBasin.timeelapse
        self.one_over_freq = thisBasin.one_over_freq
        self.meanpref = thisBasin.meanpref
        self.equi_barr = thisBasin.equi_barr

        tockmc = time.time()
        logstr += "\n" + "KMC " + str(istep_this) + "th pseudo_step is finished."
        logstr += "\n" + (f"one_over_freq: "
                          f"{round(self.one_over_freq, thissett.system['float_precision'])} ps")
        logstr += "\n" + (f"equivalent barrier: "
                          f"{round(self.equi_barr, thissett.system['float_precision'])} eV")
        logstr += "\n" + (f"Real time cost for {istep} KMC step: "
                          f"{round(tockmc - tickmc, thissett.system['float_precision'])} s")
        logstr += "\n" + "==================================================================="
        LogWriter.write_data(logstr)

def func1(x, a, b):
    return a * x + b


class TrialDisps:
    def __init__(self, displacements, ref_length, target_strainrate,
                 temp=300.0, mindisp=0.0001, maxdisp=0.01, straintype="tension"):
        self.displacements = np.array(displacements)
        self.ref_length = ref_length
        self.target_strainrate = target_strainrate
        self.temp = temp
        self.mindisp = mindisp
        self.maxdisp = maxdisp
        self.straintype = straintype
        self.ndisps = len(self.displacements)
        self.strains = self.displacements / self.ref_length
        self.barrs = np.zeros(self.ndisps)
        self.timeelapse = np.zeros(self.ndisps)
        self.one_over_freqs = np.zeros(self.ndisps)
        self.meanprefs = np.zeros(self.ndisps)
        self.target_displacement = 0.01
        self.KBT = KB * self.temp

    def Add_one_trialdisp(self, thisTrialDisp2Basin):
        id = np.where(self.displacements == thisTrialDisp2Basin.displacement)
        id = id[0][0]
        self.barrs[id] = thisTrialDisp2Basin.equi_barr
        self.timeelapse[id] = thisTrialDisp2Basin.timeelapse
        self.one_over_freqs[id] = thisTrialDisp2Basin.one_over_freq
        self.meanprefs[id] = thisTrialDisp2Basin.meanpref

    @staticmethod
    def fit(x, y):
        try:
            popt, pcov = curve_fit(func1, x, y)
            isValid = True
        except:
            popt = np.array([0.0, 0.0])
            isValid = False
        return isValid, popt

    @staticmethod
    def chop_x_y(x, y, popt):
        residuals = y - func1(x, *popt)
        absr = np.absolute(residuals)
        m = np.mean(absr)
        s = np.std(absr)
        if s > 0.0:
            r = np.absolute((absr - m) / s)
            rthres = 3
            inds = np.arange(len(x))
            inds = np.compress(r < rthres, inds)
            x = x[inds]
            y = y[inds]
        return x, y

    def apply_displacement(self):
        self.meanpref = np.mean(self.meanprefs)
        self.strainrates = np.divide(self.strains, self.one_over_freqs) * 1.0e12
        self.logstrainrates = np.log(np.absolute(self.strainrates))
        isValid, popt = TrialDisps.fit(self.strains, self.logstrainrates)
        if isValid and self.ndisps > 2:
            x, y = self.chop_x_y(self.strains, self.logstrainrates, popt)
            if len(x) >= 2:
                isValid, popt = TrialDisps.fit(x, y)
            self.target_strain = (np.log(self.target_strainrate) - popt[1]) / popt[0]
        else:
            self.target_strain = 0.0
        self.target_displacement = self.target_strain * self.ref_length

        if self.straintype[0:3].upper() == "COM":
            self.target_displacement = -abs(self.target_displacement)
        else:
            self.target_displacement = abs(self.target_displacement)

        if abs(self.target_displacement) < self.mindisp:
            if self.straintype[0:3].upper() == "COM":
                self.target_displacement = -self.mindisp
            else:
                self.target_displacement = self.mindisp
        if abs(self.target_displacement) > self.maxdisp:
            if self.straintype[0:3].upper() == "COM":
                self.target_displacement = -self.maxdisp
            else:
                self.target_displacement = self.maxdisp
        return self.target_displacement












