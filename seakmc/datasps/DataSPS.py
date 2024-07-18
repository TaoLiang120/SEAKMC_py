import time
import numpy as np

import seakmc.general.DataOut as Dataout
import seakmc.datasps.PreSPS as preSPS
import seakmc.datasps.SaddlePointSearch as mySPS
import seakmc.datasps.PostSPS as postSPS
from seakmc.restart.Restart import RESTART

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


def data_find_saddlepoints(istep, thissett, seakmcdata, DefectBank_list, thisSuperBasin, Eground,
                           DataSPs, AVitags, df_delete_SPs, undo_idavs, finished_AVs, simulation_time, object_dict):
    out_paths = object_dict['out_paths']
    LogWriter = object_dict['LogWriter']
    DFWriter = object_dict['DFWriter']

    float_precision = thissett.system['float_precision']
    iav = 0
    thiscolor = 0
    preSPS.initialization_thisdata(seakmcdata, thissett)
    for idav in undo_idavs:
        ticav = time.time()

        thisAV = preSPS.initialize_thisAV(seakmcdata, idav, Rebuild=False)
        if thisAV is not None:
            AVitags[idav] = thisAV.itags[0:(thisAV.nactive + thisAV.nbuffer)]
            x = seakmcdata.defects.at[idav, 'xsn']
            y = seakmcdata.defects.at[idav, 'ysn']
            z = seakmcdata.defects.at[idav, 'zsn']
            logstr = f"ActVol ID: {idav} nactive:{thisAV.nactive} nbuffer:{thisAV.nbuffer} nfixed:{thisAV.nfixed}"
            logstr += "\n" + (f"AV center fractional coords: "
                              f"{round(x, 5), round(y, 5), round(z, 5)}")
            LogWriter.write_data(logstr)

            local_coords, thisVNS = preSPS.initialize_AV_props(thisAV)
            #thisStrain = preSPS.get_AV_atom_strain(thisAV, thissett, thiscolor)
            thisSOPs, isPGSYMM = preSPS.get_SymmOperators(thissett, thisAV, idav,
                                                          PointGroup=thissett.active_volume["PointGroupSymm"])
            isRecycled, Pre_Disps = preSPS.get_Pre_Disps(idav, thisAV, thissett, thisSOPs, DefectBank_list, istep)

            thisnspsearch = thissett.spsearch['NSearch']
            thisSPS = preSPS.initialize_thisSPS(idav, local_coords, thisnspsearch, thissett)

            SNC, CalPref, errorlog = preSPS.initial_SNC_CalPref(idav, thisAV, thissett)
            if len(errorlog) > 0: LogWriter.write_data(logstr)

            dynmatAV = None
            if SNC or CalPref:
                SNC, CalPref, dynmatAV = preSPS.get_thisSNC4spsearch(idav, thissett, thisAV, SNC, CalPref, object_dict,
                                                                     thiscolor)

            thisSPS, df_delete_SPs = mySPS.saddlepoint_search(thiscolor, istep, thissett, idav, thisAV, local_coords,
                                                              thisSOPs, dynmatAV, SNC, CalPref,
                                                              thisSPS, Pre_Disps, thisnspsearch, thisVNS,
                                                              df_delete_SPs, object_dict)

            thisSPS, df_delete_SPs = postSPS.SPs_1postprocessing(thissett, thisSPS, df_delete_SPs, DFWriter,
                                                                 nSPstart=thisSPS.nSP)
            thisSPS, df_delete_this = DataSPs.check_dup_avSP(idav, thisSPS, seakmcdata.de_neighbors, AVitags,
                                                             thissett.saddle_point)
            df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)

            if thisSPS.nSP > 0:
                if thissett.defect_bank["Recycle"]:
                    DefectBank_list = postSPS.add_to_DefectBank(thissett, thisAV, thisSPS, isRecycled, isPGSYMM,
                                                                thisSOPs.sch_symbol, DefectBank_list, out_paths[3])
                DataSPs = postSPS.insert_AVSP2DataSPs(DataSPs, thisSPS, idav, DFWriter)

                logstr = f"Found {str(thisSPS.nSP)} saddle points in {str(idav)} active volume!"
                logstr += "\n" + "-----------------------------------------------------------------"
                LogWriter.write_data(logstr)

            Dataout.visualize_AV_SPs(thissett.visual, seakmcdata, AVitags, thisAV, thisSPS, istep, idav, out_paths[0])

        iav += 1
        finished_AVs += 1
        undo_idavs = np.delete(undo_idavs, np.argwhere(undo_idavs == idav))

        tocav = time.time()
        logstr = f"Total time for {idav} active volume: {round(tocav - ticav, float_precision)} s"
        logstr += "\n" + "-----------------------------------------------------------------"
        LogWriter.write_data(logstr)

        if (thissett.system["Restart"]["WriteRestart"] and
                finished_AVs % thissett.system["Restart"]["AVStep4Restart"] == 0):
            thisRestart = RESTART(istep, finished_AVs, DefectBank_list, thisSuperBasin, seakmcdata, Eground,
                                  DataSPs, AVitags, df_delete_SPs, undo_idavs, simulation_time)
            thisRestart.to_file()
            thisRestart = None

    thisAV = None
    thisSPS = None
    thisSOPs = None
    dynmatAV = None
    return seakmcdata, DataSPs, AVitags
