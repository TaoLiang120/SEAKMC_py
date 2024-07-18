from seakmc.kmc.KMC import Basin, DataKMC

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


def run_KMC(istep, thisSuperBasin, seakmcdata, AVitags, DataSPs, thissett, simulation_time, thisExports, LogWriter):
    thiskmc = DataKMC(istep, thissett.kinetic_MC, float_precision=thissett.system["float_precision"])
    thisBasin = Basin(len(thisSuperBasin.Basin_list), istep, seakmcdata, AVitags, DataSPs, thissett.kinetic_MC,
                      VerySmallNumber=thissett.system["VerySmallNumber"])
    nSPthis = thisBasin.nSP
    thiskmc.initialization(thisSuperBasin, thisBasin)
    time_step = thiskmc.run_KMC(thisSuperBasin)
    liba = thiskmc.iba_sels[len(thiskmc.iba_sels) - 1]
    lisp = thiskmc.isp_sels[len(thiskmc.isp_sels) - 1]
    forward_barrier = thisSuperBasin.Basin_list[liba].barrs[lisp]
    backward_barrier = thisSuperBasin.Basin_list[liba].barrs[lisp] - thisSuperBasin.Basin_list[liba].bias[lisp]
    logstr = (f"Current basin info at {istep} KMC step - "
              f"ID:{thisBasin.id} Number of saddle points:{nSPthis} ID of selection:{thiskmc.isels}")
    logstr += "\n" + (f"                  Selected event - "
                      f"Barrier:{round(forward_barrier, thissett.system['float_precision'])} "
                      f"Backward barrier:{round(backward_barrier, thissett.system['float_precision'])}")
    if thissett.kinetic_MC["AccStyle"][0:3].upper() == "MRM":
        logstr += "\n" + (f"Superbasin info at {istep} KMC step - "
                          f"Number of basins:{thiskmc.nBasin} Number of saddle points: {thiskmc.nSP} "
                          f"ID of selection:{thiskmc.gsp_sels}")
        logstr += "\n" + (f"Local info of seleted events at {istep} KMC step - "
                          f"Basin ID:{thiskmc.iba_sels} KMC step:{thisSuperBasin.Basin_list[liba].istep} "
                          f"Event ID:{thiskmc.isp_sels}")
        logstr += "\n" + (f"All seleted events of the superbasin at {istep} KMC step - "
                          f"Basin ID:{thisSuperBasin.iba_sels} Event ID: {thisSuperBasin.isp_sels}")
    logstr += "\n" + "-----------------------------------------------------------------"
    LogWriter.write_data(logstr)
    thisBasin = None
    simulation_time += time_step

    thisExports["istep"] = istep
    thisExports["nDefect"] = seakmcdata.ndefects
    thisExports["defect_center_xs"] = seakmcdata.de_center[0]
    thisExports["defect_center_ys"] = seakmcdata.de_center[1]
    thisExports["defect_center_zs"] = seakmcdata.de_center[2]
    thisExports["nBasin"] = thiskmc.nBasin
    thisExports["nSP_thisbasin"] = DataSPs.nSP
    thisExports["nSP_superbasin"] = thiskmc.nSP
    thisExports["iSP_selected"] = thiskmc.gsp_sels[len(thiskmc.gsp_sels) - 1]
    thisExports["forward_barrier"] = forward_barrier
    thisExports["ebias"] = thisSuperBasin.Basin_list[liba].bias[lisp]
    thisExports["backward_barrier"] = backward_barrier
    thisExports["time_step"] = time_step
    thisExports["simulation_time"] = simulation_time

    return simulation_time, thiskmc, thisSuperBasin, thisExports
