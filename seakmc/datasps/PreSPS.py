import os
import pandas as pd
import numpy as np
import json

from seakmc.core.symmetry import PGSymmOps, SymmOP
from seakmc.dynmat.Dynmat import DynMat
from seakmc.spsearch.SaddlePoints import AV_SPs, DefectBank

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


def load_DefectBanks(DBsett, DBLoadpath, significant_figures=6):
    thisFileHeader = DBsett["FileHeader"] + "_basin_"
    DefectBank_list = []
    for file in os.listdir(DBLoadpath):
        if thisFileHeader in file and ".csv" in file:
            thisstr = file.replace(thisFileHeader, "")
            thisstr = thisstr.replace(".csv", "")
            thisstrs = thisstr.split("_")
            try:
                thisid = int(thisstrs[0])
                thissch_symbol = thisstrs[1]
                thisDB = DefectBank.from_files(thisid, DBsett["FileHeader"], sch_symbol=thissch_symbol,
                                               filepath=DBLoadpath,
                                               namax=DBsett["NMax4DB"], namin=DBsett["NMin4DB"],
                                               SortDisps=DBsett["SortDisps"], significant_figures=significant_figures)
                if isinstance(thisDB, DefectBank): DefectBank_list.append(thisDB)
            except:
                pass
    return DefectBank_list


#####################################################################
def update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter):
    if len(df_delete_this) > 0:
        idstart = len(df_delete_SPs)
        DFWriter.write_deleted_SPs(df_delete_this, idstart=idstart, mode='a')
        if len(df_delete_this) > 0:
            if len(df_delete_SPs) == 0:
                df_delete_SPs = df_delete_this.copy(deep=True)
                df_delete_SPs = df_delete_SPs.reset_index(drop=True)
            else:
                df_delete_SPs = pd.concat([df_delete_SPs, df_delete_this], ignore_index=True)
    return df_delete_SPs


##################################################################
def initialization_thisdata(seakmcdata, thissett):
    seakmcdata.atoms = seakmcdata.get_fractional_coords(seakmcdata.atoms)
    seakmcdata.atoms = seakmcdata.insert_tags(seakmcdata.atoms)
    seakmcdata.atoms = seakmcdata.insert_itags(seakmcdata.atoms)
    rcut = thissett.active_volume['DActive'] + thissett.active_volume['DBuffer'] + thissett.active_volume['DFixed']
    seakmcdata.insert_atoms_cell(cellcut=rcut * 1.2)


def initialize_thisAV(seakmcdata, idav, Rebuild=False):
    thisAV = seakmcdata.get_active_volume(idav, Rebuild=Rebuild)
    return thisAV


def initialize_AV_props(thisAV):
    if thisAV is not None:
        local_coords = thisAV.to_coords(Buffer=False, Fixed=False)
    else:
        local_coords = np.array([[], [], []])
    thisVNS = []
    return local_coords, thisVNS


def get_AV_atom_strain(thisAV, thissett, thiscolor):
    if thisAV is not None:
        if "LAS" in thissett.spsearch["HandleVN"]["RescaleValue"].upper() and thissett.spsearch["HandleVN"][
            "RescaleVN"]:
            thisStrain = thisAV.estimate_atom_strain(thissett, nactive=thisAV.nactive, nbuffer=thisAV.nbuffer)
        else:
            thisStrain = None
    else:
        thisStrain = None
    return thisStrain


def initialize_thisSPS(idav, local_coords, nspsearch, thissett):
    thisSPS = AV_SPs(idav, local_coords, [], nspsearch, thissett.saddle_point,
                     float_precision=thissett.system["float_precision"])
    return thisSPS


def get_SymmOperators(thissett, thisAV, id, PointGroup=True):
    isPGSYMM = False
    if PointGroup:
        if thisAV.nactive > thissett.active_volume["NMax4PG"]:
            idenOP = SymmOP(np.eye(3), np.zeros(3))
            thisSOPs = PGSymmOps(id, [idenOP], tol=thissett.system["Tolerance"])
        else:
            SOPs, sch_symbol = thisAV.get_PG_OPs(nout=thisAV.nactive)
            thisSOPs = PGSymmOps(id, SOPs, sch_symbol=sch_symbol, tol=thissett.system["Tolerance"])
            thisSOPs.validate_OPs()
            isPGSYMM = True
    else:
        idenOP = SymmOP(np.eye(3), np.zeros(3))
        thisSOPs = PGSymmOps(id, [idenOP], tol=thissett.system["Tolerance"])
    return thisSOPs, isPGSYMM


def check_SNC_natoms(nmaxSNC, nactive):
    isValid = True
    if nactive > nmaxSNC:
        isValid = False
    return isValid


def initial_SNC_CalPref(idav, thisAV, thissett):
    SNC = thissett.dynamic_matrix["SNC"]
    CalPref = thissett.dynamic_matrix["CalPrefactor"]
    errorlog = ""
    NMax4SNC = thissett.dynamic_matrix["NMax4SNC"]
    isValid = check_SNC_natoms(NMax4SNC, thisAV.nactive)
    if not isValid:
        SNC = False
        CalPref = False
        errorlog = f"Number of atoms {thisAV.nactive} of {idav} active volume is larger than NMax4SNC {NMax4SNC}!"
        errorlog += "\n" + "Program will swtich SNC = False for this active volume."
        errorlog += "\n" + "If want to use SNC, terminate program, change NMax4SNC and rerun it."
    return SNC, CalPref, errorlog


def get_dynmatAV(idav, thissett, thisAV, force_evaluator, thiscolor):
    [etotal, coords, isValid, errormsg] = force_evaluator.run_runner("SPSDYNMAT", thisAV, thiscolor,
                                                                     nactive=thisAV.nactive, thisExports=None)
    if not isValid:
        print(errormsg)
        quit()
    delimiter = thissett.dynamic_matrix["delimiter"]
    vibcut = thissett.dynamic_matrix["VibCut"]
    LowerHalfMat = thissett.dynamic_matrix["LowerHalfMat"]
    dynmatAV = DynMat.from_file("Runner_" + str(thiscolor) + "/dynmat.dat", id=idav,
                                delimiter=delimiter, vibcut=vibcut, LowerHalfMat=LowerHalfMat)
    return dynmatAV


def diagonize_dynmatAV(dynmatAV, isVib=False, Get_inv_luf=True):
    dynmatAV.diagonize_matrix()
    if not isVib:
        dynmatAV.is_SNCable()
        if dynmatAV.isValid:
            if Get_inv_luf:
                dynmatAV.get_inv_luf_eigvec()
                dynmatAV.sqrt_eig()
    return dynmatAV


def get_thisSNC4spsearch(idav, thissett, thisAV, thisSNC, thisCalPref, object_dict, thiscolor):
    force_evaluator = object_dict['force_evaluator']
    dynmatAV = get_dynmatAV(idav, thissett, thisAV, force_evaluator, thiscolor)
    dynmatAV = diagonize_dynmatAV(dynmatAV, isVib=False, Get_inv_luf=True)
    if not dynmatAV.isValid:
        thisSNC = False
        thisCalPref = False
        dynmatAV = None
    else:
        if not dynmatAV.isSNCable:
            thisSNC = False
            if not thisCalPref: dynmatAV = None
    return thisSNC, thisCalPref, dynmatAV


def get_disps_from_DefectBank(thissett, thisAV, thissch_symbol, DefectBank_list, istep):
    thisdisps = []
    if thisAV.nactive < thissett.defect_bank["NMin4DB"]:
        pass
    else:
        thisna = min(thissett.defect_bank["NMax4DB"], thisAV.nactive)
        atoms_in = thisAV.atoms.truncate(after=thisAV.atoms.index[thisna - 1])
        thisdisps = []
        for i in range(len(DefectBank_list)):
            isSame = DefectBank_list[i].is_same_structure(atoms_in, thissch_symbol,
                                                          scaling=thissett.defect_bank["Scaling"],
                                                          ignore_type=thissett.defect_bank["IgnoreType"],
                                                          tolerance=thissett.defect_bank["Tol4Disp"])
            if isSame:
                thisdisps = DefectBank_list[i].load_disps(scaling=thissett.defect_bank["Scaling"],
                                                          LoadRatio=thissett.defect_bank["Ratio4DispLoad"])
                break
    return thisdisps


def get_disps_from_spsearch(loadsett, thisAV, idav):
    def get_ai_atomsquence(loadsett, thisAV, idav):
        thispath = os.getcwd()
        thisna = thisAV.nactive
        if isinstance(loadsett["LoadPath"], str): thispath = os.path.join(thispath, loadsett["LoadPath"])
        thisfhead = loadsett["FileHeader4Data"] + str(idav)
        aiatoms = None
        for file in os.listdir(thispath):
            if thisfhead in file:
                thisdf = pd.read_csv(os.path.join(thispath, file))
                n = len(thisdf)
                aiatoms = np.arange(n, dtype=int)
                coords = np.vstack((thisAV.atoms["x"], thisAV.atoms['y'], thisAV.atoms['z']))
                nn = min(n, thisna)
                coords = coords[:, 0:nn]
                for i in range(nn):
                    thisxyz = np.array([thisdf.at[thisdf.index[i], 'x'], thisdf.at[thisdf.index[i], 'y'],
                                        thisdf.at[thisdf.index[i], 'z']])
                    thisd = coords - thisxyz.reshape([3, 1]) * loadsett["Scaling"]
                    thisdsq = np.sum(thisd * thisd, axis=0)
                    ai = np.argmin(thisdsq, axis=0)
                    aiatoms[i] = ai
                break
            else:
                aiatoms = None
        return aiatoms

    thispath = os.getcwd()
    thisna = thisAV.nactive
    if isinstance(loadsett["LoadPath"], str): thispath = os.path.join(thispath, loadsett["LoadPath"])
    disps = []
    if loadsett["Method"][0:4].upper() == "FILE":
        aiatoms = None
        if loadsett["CheckSequence"]:
            aiatoms = get_ai_atomsquence(loadsett, thisAV, idav)
        thisfhead = loadsett["FileHeader"] + str(idav) + "_disp_"
        fids = []
        for file in os.listdir(thispath):
            if thisfhead in file:
                thisdf = pd.read_csv(os.path.join(thispath, file))
                #thisdf = thisdf.truncate(after = thisdf.index[min(thisna,len(thisdf))-1])
                thisdisps = np.vstack((thisdf["dx"].to_numpy(), thisdf["dy"].to_numpy(), thisdf["dz"].to_numpy())) * \
                            loadsett["Ratio4DispLoad"] * loadsett["Scaling"]
                if aiatoms is not None:
                    #aiatoms = np.compress(aiatoms<thisdisps.shape[1], aiatoms)
                    thisdisps = thisdisps[:, aiatoms]
                disps.append(thisdisps)
                if loadsett["SortDisps"]:
                    fid = file.replace(thisfhead, "")
                    fid = fid.replace(".csv", "")
                    fids.append(int(fid))
        if loadsett["SortDisps"] and len(disps) > 0:
            disps = np.array(disps)
            ai = np.argsort(fids)
            ai = np.expand_dims(ai, axis=1)
            ai = np.expand_dims(ai, axis=2)
            disps = np.take_along_axis(disps, ai, axis=0)
            disps = list(disps)
    elif loadsett["Method"][0:4].upper() == "SETT":
        thisf = loadsett["FileHeader"] + str(idav) + ".json"
        thisf = os.path.join(thispath, thisf)
        if os.path.isfile(thisf):
            with open(thisf, "r") as f:
                dispsetts = json.load(f)
            for dispsett in dispsetts:
                thisdisps = thisAV.generate_displacements(dispsett)
                if thisdisps is not None: disps.append(thisdisps * loadsett["Ratio4DispLoad"])
    thisdf = None
    thisdisps = None
    return disps


def get_Pre_Disps(idav, thisAV, thissett, thisSOPs, DefectBank_list, istep):
    isRecycled = False
    Pre_Disps = []
    if thissett.defect_bank["Preload"]:
        dbdisps = get_disps_from_DefectBank(thissett, thisAV, thisSOPs.sch_symbol, DefectBank_list, istep)
        if len(dbdisps) > 0:
            Pre_Disps = Pre_Disps + dbdisps
            isRecycled = True
    if thissett.spsearch["Preloading"]["Preload"]:
        avdisps = get_disps_from_spsearch(thissett.spsearch["Preloading"], thisAV, idav)
        if len(avdisps) > 0: Pre_Disps = Pre_Disps + avdisps

    dbdisps = None
    avdisps = None
    return isRecycled, Pre_Disps
