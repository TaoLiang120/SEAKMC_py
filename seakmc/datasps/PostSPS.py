from seakmc.spsearch.SaddlePoints import DefectBank
import seakmc.datasps.PreSPS as preSPS

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


def SPs_1postprocessing(thissett, thisSPS, df_delete_SPs, DFWriter, nSPstart=0):
    df_delete_this = thisSPS.check_duplicate(nstart=nSPstart)
    df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)
    if thissett.saddle_point["ValidSPs"]["RealtimeDelete"]:
        pass
    else:
        df_delete_this = thisSPS.validate_SPs(Delete=True)
        df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)
    group_info, df_delete_this = thisSPS.get_GSPs(SPlist=thisSPS.SPlist)
    df_delete_SPs = preSPS.update_df_delete_SPs(df_delete_SPs, df_delete_this, DFWriter)
    if thissett.saddle_point["ValidSPs"]["FindSPType"]: thisSPS.get_SP_type()
    return thisSPS, df_delete_SPs


def save_DefectBanks(DBsett, thisDB, DBSavepath):
    thisDB.to_files(DBsett["FileHeader"], filepath=DBSavepath, OutIndex=DBsett["OutIndex"])


def add_to_DefectBank(thissett, thisAV, thisSPS, isRecycled, isPGSYMM, thissch_symbol, DefectBank_list, DBSavepath):
    if isRecycled:
        return DefectBank_list
    else:
        if isPGSYMM and thissett.defect_bank["UseSymm"]:
            if thisSPS.type_info is None: thisSPS.get_SP_type(SPlist=thisSPS.SPlist)
            thisDB = DefectBank.from_AV_SPs(thisAV, thisSPS, id=len(DefectBank_list), sch_symbol=thissch_symbol,
                                            scaling=thissett.defect_bank["Scaling"],
                                            namax=thissett.defect_bank["NMax4DB"],
                                            namin=thissett.defect_bank["NMin4DB"], Style='type',
                                            significant_figures=thissett.system["significant_figures"])
            if isinstance(thisDB, DefectBank):
                DefectBank_list.append(thisDB)
                if thissett.defect_bank["SaveDB"]:
                    save_DefectBanks(thissett.defect_bank, thisDB, DBSavepath)
        else:
            thisDB = DefectBank.from_AV_SPs(thisAV, thisSPS, id=len(DefectBank_list), sch_symbol=thissch_symbol,
                                            scaling=thissett.defect_bank["Scaling"],
                                            namax=thissett.defect_bank["NMax4DB"],
                                            namin=thissett.defect_bank["NMin4DB"], Style='All',
                                            significant_figures=thissett.system["significant_figures"])
            if isinstance(thisDB, DefectBank):
                DefectBank_list.append(thisDB)
                if thissett.defect_bank["SaveDB"]:
                    save_DefectBanks(thissett.defect_bank, thisDB, DBSavepath)
        return DefectBank_list


def insert_AVSP2DataSPs(DataSPs, thisSPS, idav, DFWriter):
    idstart = DataSPs.nSP
    DataSPs, thisdf_SPs = DataSPs.insert_AVSPs(thisSPS, idav)
    DFWriter.write_SPs(thisdf_SPs, idstart=idstart, mode="a")
    return DataSPs
