import time
import numpy as np

from seakmc.core.data import ActiveVolumeSPS
from seakmc.core.util import mats_angle
from seakmc.spsearch.SPSearch import Dimer
from seakmc.spsearch.SaddlePoints import SaddlePoint
import seakmc.datasps.PreSPS as preSPS

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


def generate_VN(spsearchsett, thisVNS, nactive, SNC=False, dmAV=None):
    isvalid = False
    thisiter = 0
    while not isvalid:
        if SNC:
            VN = np.random.rand(3 * nactive) - 0.5
            isel = np.random.randint(0, high=3 * nactive)
            VN += dmAV.eigvec.T[isel]
            VN = VN.reshape([3, nactive])
        else:
            VN = np.random.rand(3, nactive) - 0.5
        if spsearchsett["HandleVN"]["CheckAng4Init"]:
            anglemin = 180.0
            for i in range(len(thisVNS)):
                angle = mats_angle(VN, thisVNS[i], Flatten=True)
                if angle < anglemin: anglemin = angle

            if (anglemin < spsearchsett["HandleVN"]["AngTol4Init"] and
                    thisiter <= spsearchsett["HandleVN"]["MaxIter4Init"]):
                isvalid = False
            else:
                isvalid = True
            thisiter += 1
        else:
            isvalid = True
    return VN


def saddlepoint_search(thiscolor, istep, thissett, idav, thisAV, local_coords, thisSOPs, dynmatAV, SNC, CalPref,
                       thisSPS, Pre_Disps, thisnspsearch, thisVNS,
                       df_delete_SPs, object_dict):
    out_paths = object_dict['out_paths']
    force_evaluator = object_dict['force_evaluator']
    LogWriter = object_dict['LogWriter']
    DFWriter = object_dict['DFWriter']

    float_precision = thissett.system['float_precision']
    for idsps in range(thisnspsearch):
        ticd = time.time()
        thisAVd = ActiveVolumeSPS.from_activevolume(idsps, thisAV)
        thisVN = generate_VN(thissett.spsearch, thisVNS, thisAV.nactive, SNC, dynmatAV)
        if len(thisVN) > 0 and len(thisVNS) <= thissett.spsearch["HandleVN"]["NMaxRandVN"]: thisVNS.append(thisVN)
        if thissett.spsearch["Method"].upper() == "DIMER":
            thispredisp = []
            if idsps < len(Pre_Disps):
                thispredisp = Pre_Disps[idsps]
                Pre_Disps[idsps] = np.array([])
            thisspsearch = Dimer(idav, idsps, thisAVd, thissett, thiscolor, force_evaluator,
                                 SNC=SNC, dmAV=dynmatAV, pre_disps=thispredisp,
                                 apply_mass=thissett.spsearch["ApplyMass"])
            thisspsearch.dimer_init(thisVN)
            thisspsearch.dimer_search(thisSPS)

            if thissett.spsearch["SearchBuffer"]:
                thisspsearch.dimer_re_search(thisSPS, nactive=thisAV.nactive + thisAV.nbuffer)
            thisspsearch.dimer_finalize()

            if CalPref and thisspsearch.ISVALID:
                toDel = thisspsearch.is_to_be_delete()
                if not toDel:
                    thisspsearch.calculate_prefactor()
            thisspsearch.force_evaluator.close()
            thisspsearch.dimer_finish()

        if thisspsearch.ISVALID:
            thissp = SaddlePoint(idav, idsps, idsps + 1, thisspsearch.BARR, thisspsearch.PREF, thisspsearch.EBIAS,
                                 thisspsearch.ISCONNECT,
                                 thisspsearch.XDISP, thisspsearch.DMAG, thisspsearch.DMAT, thisspsearch.DVEC,
                                 thisspsearch.FXDISP, thisspsearch.FDMAG, thisspsearch.ISVALID,
                                 iters=thisspsearch.iter, ntrans=thisspsearch.NTSITR, emax=thisspsearch.EDIFF_MAX,
                                 rdcut=thissett.spsearch["R2Dmax4SPAtom"], dcut=thissett.spsearch["DCut4SPAtom"],
                                 dyncut=thissett.spsearch["DynCut4SPAtom"],
                                 tol=thissett.system["Tolerance"])

            isDup, df_delete_this = thisSPS.check_this_duplicate(thissp)
            if isDup:
                df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)
                thissp.ISVALID = False
            else:
                if thissett.saddle_point["ValidSPs"]["RealtimeDelete"]:
                    df_delete_this, delete_this = thisSPS.realtime_validate_thisSP(thissp)
                    if delete_this:
                        df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)
                        thissp.ISVALID = False
            if thissp.ISVALID:
                if thisSOPs.nOP > 1:
                    thisSymmsps = thissp.get_SPs_from_symmetry(local_coords, thisSOPs)
                    thisSPS.insert_SP([thissp] + thisSymmsps)
                else:
                    thisSPS.insert_SP(thissp)
                if thissett.saddle_point["ValidSPs"]["RealtimeValid"]: thisSPS.validate_SPs(Delete=False)

        tocd = time.time()
        logstr = (f"idav:{idav}, idsps: {idsps},  ntrans: {thisspsearch.NTSITR},  "
                  f"barrier:{round(thisspsearch.BARR, float_precision)},  "
                  f"ebias: {round(thisspsearch.EBIAS, float_precision)}")
        logstr += "\n" + (f"      dmag:{round(thisspsearch.DMAG, float_precision)}, "
                          f"dmagFin:{round(thisspsearch.FDMAG, float_precision)}, isConnect: {thisspsearch.ISCONNECT}")
        logstr += "\n" + (f"      Valid SPs: {thisSPS.nvalid}, num of SPs:{thisSPS.nSP}, "
                          f"time: {round(tocd - ticd, float_precision)} s")
        logstr += "\n" + "-----------------------------------------------------------------"
        LogWriter.write_data(logstr)

    return thisSPS, df_delete_SPs
