import os
import pandas as pd
from monty.io import zopen

from seakmc.runner.LammpsRunner import LammpsRunner
from seakmc.runner.PyLammpsRunner import PyLammpsRunner
from seakmc.runner.VaspRunner import VaspRunner
from seakmc.input.Input import export_Keys
from seakmc.input.Input import SP_COMPACT_HEADER, SP_COMPACT_HEADER4Delete

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


class LogWriter(object):
    def __init__(self, path, Screen=True, Log=True, RESTART=False):
        self.logfile = os.path.join(path, "Seakmc.log")
        self.Screen = Screen
        self.Log = Log
        logstr_input = "The data file uses the same format of LAMMPS."
        logstr_input += "The unit is LAMMPS metal unit."
        logstr_input += "\n" + "The default bond length is the bond length in pymatgen with bond order one."
        logstr_input += "\n" + "The default cutting distance for neighbors is 1.1 times of the bond lengths!"
        logstr_input += "\n" + "The default coordination number is 4."
        if self.Log:
            if RESTART in (None, False):
                with open(self.logfile, 'w') as data_file:
                    data_file.write(logstr_input)
                    data_file.write("\n" + "Seakmc start ...")
            else:
                if os.path.isfile(self.logfile):
                    with open(self.logfile, 'a') as data_file:
                        data_file.write(
                            "\n" + f"Restart istep: {RESTART.istep_this} and finished AVs: {RESTART.finished_AVs} ...")
                else:
                    with open(self.logfile, 'w') as data_file:
                        data_file.write(logstr_input)
                        data_file.write(
                            "\n" + f"Restart istep: {RESTART.istep_this} and finished AVs: {RESTART.finished_AVs} ...")

    def write_data(self, logstr):
        if self.Screen: print(logstr)
        if self.Log:
            # write the line to the file
            with open(self.logfile, 'a') as data_file:
                data_file.write("\n" + logstr)


class SeakmcSummary(object):
    def __init__(self, path, RESTART=False, significant_figures=6):
        self.summaryfile = os.path.join(path, "Seakmc_summary.csv")
        self.significant_figures = significant_figures
        if RESTART in (None, False):
            thisstr = export_Keys[0]
            for i in range(1, len(export_Keys)):
                thisstr += "," + export_Keys[i]
            with open(self.summaryfile, 'w') as f:
                f.write(thisstr)
        else:
            if os.path.isfile(self.summaryfile):
                pass
            else:
                thisstr = export_Keys[0]
                for i in range(1, len(export_Keys)):
                    thisstr += "," + export_Keys[i]
                with open(self.summaryfile, 'w') as f:
                    f.write(thisstr)

        export_dict = {}
        for key in export_Keys:
            export_dict[key] = 0.0
        self.export_dict = export_dict

    def update_data(self, thisdict):
        for key in thisdict:
            self.export_dict[key] = thisdict[key]

    def write_data(self):
        thisstr = str(int(self.export_dict[export_Keys[0]]))
        for i in range(1, len(export_Keys)):
            if export_Keys[i] == "nDefect" or export_Keys[i] == "nSP_thisbasin" or \
                    export_Keys[i] == "nSP_superbasin" or export_Keys[i] == "nBasin" or \
                    export_Keys[i] == "iSP_selected":
                thisstr += "," + str(int(self.export_dict[export_Keys[i]]))
            else:
                thisstr += "," + str(round(self.export_dict[export_Keys[i]], self.significant_figures))

        with open(self.summaryfile, 'a') as f:
            f.write("\n" + thisstr)


class DFWriter(object):
    def __init__(self, OutPath="SPOut", WriteSPs=True):
        self.OutPath = OutPath
        self.WriteSPs = WriteSPs

    def init_deleted_SPs(self, istep):
        self.deleted_SPs_file = "KMC_" + str(istep) + "_Deleted_SPs.csv"
        if self.WriteSPs:
            thisdf = pd.DataFrame(columns=SP_COMPACT_HEADER4Delete)
            self.dfs_to_file(self.deleted_SPs_file, thisdf, idstart=0, mode='w')

    def init_SPs(self, istep):
        self.SPs_file = "KMC_" + str(istep) + "_SPs.csv"
        if self.WriteSPs:
            thisdf = pd.DataFrame(columns=SP_COMPACT_HEADER)
            self.dfs_to_file(self.SPs_file, thisdf, idstart=0, mode='w')

    def write_deleted_SPs(self, thisdf, idstart=0, mode='a'):
        if self.WriteSPs:
            self.dfs_to_file(self.deleted_SPs_file,
                             thisdf, idstart=idstart, mode=mode)

    def write_SPs(self, thisdf, idstart=0, mode='a'):
        if self.WriteSPs:
            self.dfs_to_file(self.SPs_file,
                             thisdf, idstart=idstart, mode=mode)

    def dfs_to_file(self, filename, thisdf, idstart=0, mode="append"):
        filename = os.path.join(self.OutPath, filename)
        cols = thisdf.columns.tolist()
        if isinstance(thisdf, pd.DataFrame):
            lines = []
            if mode[0:1].upper() == "W":
                thisstr = ""
                for k in range(len(cols)):
                    thisstr += "," + cols[k]
                lines.append(thisstr)

            for i in range(len(thisdf)):
                thisstr = str(idstart + i)
                for k in range(len(cols)):
                    thisstr += "," + str(thisdf.at[i, cols[k]])
                lines.append(thisstr)

            if mode[0:1].upper() == "A":
                if len(lines) > 0:
                    with zopen(filename, "at", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")
            elif mode[0:1].upper() == "W":
                if len(lines) > 0:
                    with zopen(filename, "wt", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")


def object_maker(thissett, thisRestart):
    object_dict = {}
    THIS_PATH = os.getcwd()

    thisLogWriter = LogWriter(THIS_PATH, Screen=thissett.visual["Screen"], Log=thissett.visual["Log"],
                              RESTART=thisRestart)
    object_dict['LogWriter'] = thisLogWriter

    thisSummary = SeakmcSummary(THIS_PATH, RESTART=thisRestart,
                                significant_figures=thissett.system["significant_figures"])
    object_dict['thisSummary'] = thisSummary

    if thissett.force_evaluator['Style'].upper() == "VASP":
        thisRunner = VaspRunner(thissett)
    elif thissett.force_evaluator['Style'].upper() == "PYLAMMPS":
        thisRunner = PyLammpsRunner(thissett)
    elif thissett.force_evaluator['Style'].upper() == "LAMMPS":
        thisRunner = LammpsRunner(thissett)
    else:
        print("Unkown force_evaluator!")
        quit()
    object_dict['force_evaluator'] = thisRunner

    AVOutpath = os.path.join(THIS_PATH, "AVOut")
    os.makedirs(AVOutpath, exist_ok=True)

    DataOutpath = os.path.join(THIS_PATH, "DataOut")
    os.makedirs(DataOutpath, exist_ok=True)

    SPOutpath = os.path.join(THIS_PATH, "SPOut")
    os.makedirs(SPOutpath, exist_ok=True)

    if isinstance(thissett.defect_bank["LoadPath"], str):
        DBLoadpath = os.path.join(THIS_PATH, thissett.defect_bank["LoadPath"])
    else:
        DBLoadpath = THIS_PATH

    if isinstance(thissett.defect_bank["SavePath"], str):
        DBSavepath = os.path.join(THIS_PATH, thissett.defect_bank["SavePath"])
        os.makedirs(DBSavepath, exist_ok=True)
    else:
        DBSavepath = THIS_PATH

    DynMatOutpath = os.path.join(THIS_PATH, "DynMatOut")
    os.makedirs(DynMatOutpath, exist_ok=True)

    Paths = [AVOutpath, DataOutpath, DBLoadpath, DBSavepath, SPOutpath, DynMatOutpath, THIS_PATH]
    object_dict['out_paths'] = Paths


    thisDFWriter = DFWriter(OutPath=SPOutpath, WriteSPs=thissett.visual["Write_SP_Summary"])
    object_dict['DFWriter'] = thisDFWriter

    return object_dict
