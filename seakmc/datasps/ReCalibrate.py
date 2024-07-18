import time
import copy
import numpy as np
import pandas as pd

from seakmc.input.Input import SP_COMPACT_HEADER4Delete

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


def calibrate_energy_with_DataSPs(thissett, DataSPs, seakmcdata, AVitags, Eground, object_dict, ReBias=False,
                                  float_precision=3):
    out_paths = object_dict['out_paths']
    force_evaluator = object_dict['force_evaluator']
    LogWriter = object_dict['LogWriter']

    logstr = "\n" + "ReCalibrating the energy with data ..."
    LogWriter.write_data(logstr)

    df_delete_this = pd.DataFrame(columns=SP_COMPACT_HEADER4Delete)
    to_delete_localisp = np.array([], dtype=int)
    thistic = time.time()
    for iav in range(len(DataSPs.df_SPs)):
        nSP = len(DataSPs.df_SPs[iav])
        idav = DataSPs.df_SPs[iav].at[0, "idav"]
        newbarrs = []
        newbias = []
        ispstart = DataSPs.ispstart[iav]
        for i in range(nSP):
            thisid = ispstart + i
            thisRecal = True
            if isinstance(thissett.saddle_point["Thres4Recalib"], float) or isinstance(
                    thissett.saddle_point["Thres4Recalib"], int):
                if DataSPs.df_SPs[iav].at[i, "barrier"] - DataSPs.barriermin > thissett.saddle_point[
                    "Thres4Recalib"]: thisRecal = False
            if thisRecal:
                thisdata = copy.deepcopy(seakmcdata)
                thisdata.update_coords_from_disps(idav, DataSPs.disps[ispstart + i], AVitags)
                [thisesp, coords, isValid, errormsg] = force_evaluator.run_runner("MD0", thisdata, 0,
                                                                                  nactive=thisdata.natoms,
                                                                                  thisExports=None)
                if not isValid:
                    LogWriter.write_data(errormsg)
                    quit()
                thisbarr = round(thisesp - Eground, float_precision)
                newbarrs.append(thisbarr)
                DataSPs.df_SPs[iav].at[i, "barrier"] = thisbarr
                if ReBias:
                    thisdata = copy.deepcopy(seakmcdata)
                    thisdata.update_coords_from_disps(idav, DataSPs.fdisps[ispstart + i], AVitags)
                    [thisefin, coords, isValid, errormsg] = force_evaluator.run_runner("MD0", thisdata, 0,
                                                                                       nactive=thisdata.natoms,
                                                                                       thisExports=None)
                    if not isValid:
                        LogWriter.write_data(errormsg)
                        quit()
                    thisebias = round(thisefin - Eground, float_precision)
                    newbias.append(thisebias)
                    DataSPs.df_SPs[iav].at[i, "ebias"] = thisebias
                if thisbarr < thissett.saddle_point["BarrierMin"]:
                    to_delete_localisp = np.append(to_delete_localisp, [ispstart + i])
                    thisrow = DataSPs.df_SPs[iav].loc[i].to_dict()
                    thisrow["reason"] = "minB_Re"
                    df_delete_this.loc[len(df_delete_this)] = thisrow
                elif ReBias:
                    if thisbarr - thisebias < thissett.saddle_point["BackBarrierMin"]:
                        to_delete_localisp = np.append(to_delete_localisp, [ispstart + i])
                        thisrow = DataSPs.df_SPs[iav].loc[i].to_dict()
                        thisrow["reason"] = "minBB_Re"
                        df_delete_this.loc[len(df_delete_this)] = thisrow
    thisdata = None
    if len(to_delete_localisp) > 0: DataSPs.reorganization(to_delete_localisp)
    thistoc = time.time()
    logstr = f"Finished recalibration and time cost is {round(thistoc - thistic, thissett.system['float_precision'])}s."
    logstr += "\n" + "KMC_istep_SPs.csv has energies before the recalibration."
    logstr += "\n" + "KMC_istep_Prob.csv has energies after the recalibration."
    logstr += "\n" + "-----------------------------------------------------------------"
    LogWriter.write_data(logstr)

    return DataSPs, df_delete_this
