import copy
import numpy as np

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


def visualize_data_AVs(Vsett, seakmcdata, istep, DataOutpath):
    if Vsett["Write_Data_SPs"]["Write_Data_AVs"]:
        filename = DataOutpath + "/" + "KMC_" + str(istep) + "_Data_AVs" + ".dat"
        seakmcdata.Write_AVs(filename, idavs="all", Buffer=Vsett["ShowBuffer"], Fixed=Vsett["ShowFixed"],
                             Invisible=Vsett["Invisible"], Reset_Index=Vsett["Reset_Index"])


def visualize_AV_SPs(Vsett, seakmcdata, AVitags, thisAV, thisSPS, istep, idav, AVOutpath):
    if Vsett["Write_AV_SPs"]["Write_AV_SPs"]:
        if thisSPS.nSP > 0:
            fileheader = "KMC_" + str(istep) + "_AV_" + str(idav) + "_"
            thisAV.Write_Stack_avSPs(thisSPS.SPlist, fileheader, OutPath=AVOutpath,
                                   rdcut4vis=Vsett["RCut4Vis"], dcut4vis=Vsett["DCut4Vis"],
                                   DispStyle=Vsett["Write_AV_SPs"]["DispStyle4AVSP"],
                                   Invisible=Vsett["Invisible"], Reset_Index=Vsett["Reset_Index"])
    if Vsett["Write_AV_SPs"]["Write_Data_AV_SPs"]:
        if thisSPS.nSP > 0:
            fileheader = "KMC_" + str(istep) + "_Data_AV_" + str(idav) + "_"
            seakmcdata.Write_Stack_SPs(thisSPS.SPlist, AVitags, fileheader, OutPath=AVOutpath,
                                       rdcut4vis=Vsett["RCut4Vis"], dcut4vis=Vsett["DCut4Vis"],
                                       DispStyle=Vsett["Write_AV_SPs"]["DispStyle4AVSP"],
                                       Invisible=Vsett["Invisible"], Reset_Index=Vsett["Reset_Index"])
    if Vsett["Write_AV_SPs"]["Write_Local_AV"]:
        filename = AVOutpath + "/" + "KMC_" + str(istep) + "_Data_AV_" + str(idav) + ".dat"
        seakmcdata.Write_AVs(filename, idavs=idav, Buffer=Vsett["ShowBuffer"], Fixed=Vsett["ShowFixed"],
                             Invisible=Vsett["Invisible"], Reset_Index=Vsett["Reset_Index"])

        filename = AVOutpath + "/" + "KMC_" + str(istep) + "_AV_" + str(idav) + ".dat"
        thisAV.Write_AV(filename, Fixed=Vsett["ShowFixed"])

def write_prob_to_file(Vsett, thiskmc, DataSPs, istep, SPOutpath, VerySmallNumber=1.0e-20):
    if Vsett["Write_Data_SPs"]["Write_Prob"]:
        filename = SPOutpath + "/" + "KMC_" + str(istep) + "_Prob" + ".csv"
        fsel = SPOutpath + "/" + "KMC_" + str(istep) + "_DetailSPs" + ".csv"
        thiskmc.Prob_to_file(filename, DataSPs, DetailOut=Vsett["Write_Data_SPs"]["DetailOut"], fsel=fsel,
                             SPs4Detail=Vsett["Write_Data_SPs"]["SPs4Detail"], VerySmallNumber=VerySmallNumber)


def get_sel_SPs_for_out(Vsett, thiskmc, DataSPs):
    sel_SPs = []
    if Vsett["Write_Data_SPs"]["Write_Data_SPs"]:
        if isinstance(Vsett["Write_Data_SPs"]["Sel_iSPs"], list):
            for i in Vsett["Write_Data_SPs"]["Sel_iSPs"]:
                try:
                    sel_SPs.append(i)
                except:
                    pass
        elif Vsett["Write_Data_SPs"]["Sel_iSPs"].upper() == "ALL":
            sel_SPs = np.arange(DataSPs.nSP, dtype=int)
        else:
            sel_SPs = thiskmc.isels.copy()
    return sel_SPs


def visualize_data_SPs(Vsett, seakmcdata, AVitags, DataSPs, sel_SPs, istep, DataOutpath):
    if Vsett["Write_Data_SPs"]["OutputStyle"][0:3].upper() == "SEP":
        seakmcdata.Write_Separate_SPs_from_DataSPs(sel_SPs, DataSPs, AVitags, istep, OutPath=DataOutpath,
                                                   DispStyle=Vsett["Write_Data_SPs"]["DispStyle4DataSP"],
                                                   Invisible=Vsett["Invisible"],
                                                   offset=Vsett["Write_Data_SPs"]["Offset"])
    else:
        fileheader = "KMC_" + str(istep) + "_Data_" + "Selected_"
        seakmcdata.Write_Stack_SPs_from_DataSPs(sel_SPs, DataSPs, AVitags, fileheader, OutPath=DataOutpath,
                                                rdcut4vis=Vsett["RCut4Vis"], dcut4vis=Vsett["DCut4Vis"],
                                                DispStyle=Vsett["Write_Data_SPs"]["DispStyle4DataSP"],
                                                Invisible=Vsett["Invisible"], Reset_Index=Vsett["Reset_Index"])


def visualize_data_SPs_Superbasin(Vsett, thiskmc, thisSuperBasin, istep, DataOutpath):
    if Vsett["Write_Data_SPs"]["Write_Data_SPs"]:
        iba = thiskmc.iba_sels[len(thiskmc.iba_sels) - 1]
        iav = thiskmc.iav_sels[len(thiskmc.iav_sels) - 1]
        isp = thiskmc.isp_sels[len(thiskmc.isp_sels) - 1]
        thisBasin = thisSuperBasin.Basin_list[iba]
        idav = thisBasin.idavs[iav]
        thisdata = copy.deepcopy(thisBasin.data)
        thisdata.Write_SPs_from_Superbasin(isp, idav, thisBasin, istep, OutPath=DataOutpath,
                                           DispStyle=Vsett["Write_Data_SPs"]["DispStyle4DataSP"],
                                           Invisible=Vsett["Invisible"], offset=10000)
        thisdata = None
