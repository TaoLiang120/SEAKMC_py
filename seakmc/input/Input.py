import copy

import numpy as np
# import json
import yaml
from numpy import pi
# from pymatgen.core.sites import Site
from pymatgen.core.bonds import obtain_all_bond_lengths
from pymatgen.core.periodic_table import Element

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"

NDISPARRAY = 3
SEQUENCE_DISPARRAY = ["SP", "FS", "FI"]

SP_COMPACT_HEADER = ["idav", "idsps", "type", "iters", "ntrans", "emax", "barrier", "prefactor", "ebias",
                     "dtot", "dmag", "dmax", "dsum", "adsum", "nad",
                     "dtotfs", "dmagfs", "dmaxfs", "dsumfs", "adsumfs", "nadfs",
                     "dtotfi", "dmagfi", "dmaxfi", "dsumfi", "adsumfi", "nadfi",
                     "isConnect"]

NENTRY_COMPACT_DISP = 6

SP_COMPACT_HEADER4Delete = SP_COMPACT_HEADER + ["reason"]
SP_DATA_HEADER = ["idav", "idsps", "barrier", "prefactor", "ebias", "dmag", "dmax", "dsum", "adsum", "dmagfi", "dmaxfi",
                  "dsumfi", "adsumfi", "isConnect"]
SP_KMC_HEADER = ["idav", "idsps", "barrier", "prefactor", "ebias", "dmag", "dmax", "dsum", "dsumr", "dmagfi", "dmaxfi",
                 "dsumfi", "dsumrfi", "isConnect", "probability", "isSel"]
SP_KMC_SELECTED_HEADER = ["isp", "barrier", "prefactor", "ebias",
                          "dtot", "dmag", "dmax", "dsum", "adsum",
                          "dmagx", "dmagy", "dmagz", "dmaxx", "dmaxy", "dmaxz", "vmaxx", "vmaxy", "vmaxz",
                          "dsumx", "dsumy", "dsumz", "adsumx", "adsumy", "adsumz",
                          "dtotfs", "dmagfs", "dmaxfs", "dsumfs", "adsumfs",
                          "dmagxfs", "dmagyfs", "dmagzfs", "dmaxxfs", "dmaxyfs", "dmaxzfs",
                          "vmaxxfs", "vmaxyfs", "vmaxzfs",
                          "dsumxfs", "dsumyfs", "dsumzfs", "adsumxfs", "adsumyfs", "adsumzfs",
                          "dtotfi", "dmagfi", "dmaxfi", "dsumfi", "adsumfi",
                          "dmagxfi", "dmagyfi", "dmagzfi", "dmaxxfi", "dmaxyfi", "dmaxzfi",
                          "vmaxxfi", "vmaxyfi", "vmaxzfi",
                          "dsumxfi", "dsumyfi", "dsumzfi", "adsumxfi", "adsumyfi", "adsumzfi"]
NENTRY_SELECTED_HEADER = 20
SORT_BY_KEYS = ["D", "DXY", "DXZ", "DYZ", "X", "Y", "Z"]

DEFECTBANK_ATOMS_HEADER = ["type", "x", "y", "z"]
DEFECTBANK_DISPS_HEADER = ["dx", "dy", "dz"]

export_Keys = ["istep", "ground_energy", "nDefect", "defect_center_xs", "defect_center_ys", "defect_center_zs",
               "nBasin", "nSP_thisbasin", "nSP_superbasin", "iSP_selected", "forward_barrier", "ebias",
               "backward_barrier", "time_step", "simulation_time"]

globals_dict = {"significant_figures": 6, "float_precision": 3, "VerySmallNumber": 1.0e-20, "Tolerance": 0.1,
                "angle_tolerance": 5.0, "float_format": "%.3f"}


class Globals:
    def __init__(
            self,
            system,
    ):
        self.system = system

    def reset_Global_Variables(self, indict):
        for key in globals_dict:
            if key in indict:
                self.system[key] = indict[key]
        float_format = '%.' + str(self.system["float_precision"]) + 'f'
        self.system["float_format"] = float_format


Global_Variables = Globals(globals_dict)


def get_atomic_mass(sp):
    sp = Element(sp) if isinstance(sp, str) else sp
    return sp.atomic_mass


def get_avg_bond_length(sp1, sp2):
    sp1 = Element(sp1) if isinstance(sp1, str) else sp1
    sp2 = Element(sp2) if isinstance(sp2, str) else sp2
    try:
        all_lengths = obtain_all_bond_lengths(sp1, sp2)
        n = 0
        bls = 0.0
        for bo, bl in all_lengths:
            bls += bl
            n += 1
        return bls / max(n, 1)
    except:
        r1 = sp1.atomic_radius
        r2 = sp2.atomic_radius
        if r1 is None: r1 = 2.0
        if r2 is None: r2 = 2.0
        return r1 + r2


def get_bo_one_length(sp1, sp2):
    sp1 = Element(sp1) if isinstance(sp1, str) else sp1
    sp2 = Element(sp2) if isinstance(sp2, str) else sp2
    try:
        all_lengths = obtain_all_bond_lengths(sp1, sp2)
        return all_lengths["1"]
    except:
        r1 = sp1.atomic_radius
        r2 = sp2.atomic_radius
        if r1 is None: r1 = 2.0
        if r2 is None: r2 = 2.0
        return r1 + r2


class Settings:
    def __init__(
            self,
            system,
            force_evaluator,
            potential,
            data,
            kinetic_MC,
            active_volume,
            defect_bank,
            dynamic_matrix,
            spsearch,
            saddle_point,
            visual,
    ):
        self.system = system
        self.force_evaluator = force_evaluator
        self.potential = potential
        self.data = data
        self.kinetic_MC = kinetic_MC
        self.active_volume = active_volume
        self.defect_bank = defect_bank
        self.dynamic_matrix = dynamic_matrix
        self.spsearch = spsearch
        self.saddle_point = saddle_point
        self.visual = visual

    def __str__(self):
        keys = ["system", "force_evaluator", "potential", "data", "kinetic_MC", "active_volume", "defect_bank",
                "dynamic_matrix", "spsearch", "saddle_point", "visual"]
        s = ""
        for key in keys:
            s += key + ":\n"
            if key == "system":
                s += str(self.system) + "\n"
            elif key == "force_evaluator":
                s += str(self.force_evaluator) + "\n"
            elif key == "potential":
                s += str(self.potential) + "\n"
            elif key == "data":
                s += str(self.data) + "\n"
            elif key == "kinetic_MC":
                s += str(self.kinetic_MC) + "\n"
            elif key == "active_volume":
                s += str(self.active_volume) + "\n"
            elif key == "defect_bank":
                s += str(self.defect_bank) + "\n"
            elif key == "dynamic_matrix":
                s += str(self.dynamic_matrix) + "\n"
            elif key == "spsearch":
                s += str(self.spsearch) + "\n"
            elif key == "saddle_point":
                s += str(self.saddle_point) + "\n"
            elif key == "visual":
                s += str(self.visual) + "\n"

        return s

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            parameters = yaml.safe_load(f)

        TempFiles = ["tmp0.dat", "tmp1.dat", "tmp2.dat"]
        Restart = {"LoadRestart": True, "LoadFile": None, "WriteRestart": True,
                   "AVStep4Restart": 1000, "KMCStep4Restart": 1, "Reset_Simulation_Time": False}
        thissystem = {"TempFiles": TempFiles, "Interval4ShowProgress": 10,
                      "significant_figures": 6, "float_precision": 3, "VerySmallNumber": 1.0e-20,
                      "angle_tolerance": 5.0, "Tolerance": 0.1,
                      "Restart": Restart}

        if "system" in parameters:
            tsystem = parameters["system"]
            for key in tsystem:
                if key == "TempFiles":
                    pass
                elif key == "Restart":
                    for subkey in tsystem[key]:
                        thissystem[key][subkey] = tsystem[key][subkey]
                else:
                    thissystem[key] = tsystem[key]
        #############################################################
        thisdata = {"units": "metal", "dimension": 3, "boundary": "p p p",
                    "Relaxed": True, "BoxRelax": False, "MoleDyn": False,
                    "RinputOpt": False, "RinputMD": False, "RinputMD0": False}
        data = parameters['data']
        if "FileName" not in data:
            raise ValueError("There must be a FileName for data!")
        if "atom_style" not in data:
            raise ValueError("There must be an atom_style for data!")

        for key in data:
            thisdata[key] = data[key]

        if "atom_style_after" not in thisdata:
            atom_style = thisdata["atom_style"]
            atom_style_after = atom_style
            if atom_style_after == "atomic":
                atom_style_after = "molecular"
            elif atom_style_after == "charge":
                atom_style_after = "full"
            thisdata["atom_style_after"] = atom_style_after
        tmpstrs = list(filter(None, list(map(lambda strings: strings.strip(), thisdata["boundary"].split(" ")))))
        PBC = [True, True, True]
        for i in range(min(3, len(tmpstrs))):
            if tmpstrs[i].upper() != "P":
                PBC[i] = False
        thisdata["PBC"] = PBC
        #########################################################
        Relaxation = {"BoxRelax": False, "InitTemp4Opt": 0.0, "TargetTemp4NVT": 5.0, "NVTSteps4Opt": 10000}
        thisfeval = {"Bin": "pylammps", "Path2Bin": False, "Style": "pylammps", "nproc": "auto", "processors": False,
                     "partition": False, "Screen": False, "LogFile": False, "NSteps4Relax": 10000, "timestep": 0.002,
                     "Master_Slave4ReCal": False, "RinputOpt": False, "RinputMD0": False, "nproc4ReCal": "auto",
                     "ImportValue4RinputOpt": False, "Keys4ImportValue4RinputOpt": [["Timestep", "time_step"]],
                     "OutFileHeaders": [], "Relaxation": Relaxation,
                     "GPU": None}

        if "force_evaluator" in parameters:
            force_evaluator = parameters["force_evaluator"]
            keys4IV = []
            for key in force_evaluator:
                if key == "Relaxation":
                    for subkey in force_evaluator[key]:
                        thisfeval[key][subkey] = force_evaluator[key][subkey]
                elif key == "Keys4ImportValue4RinputOpt":
                    thisstrs = force_evaluator["Keys4ImportValue4RinputOpt"]
                    for i in range(len(thisstrs)):
                        thislines = thisstrs[i].split(",")
                        try:
                            thiskeylist = [thislines[0].strip(), thislines[1].strip()]
                            keys4IV.append(thiskeylist)
                        except:
                            pass
                    thisfeval[key] = keys4IV
                else:
                    thisfeval[key] = force_evaluator[key]

        if isinstance(thisfeval["OutFileHeaders"], list):
            for i in range(len(thisfeval["OutFileHeaders"]) - 1, -1, -1):
                if not isinstance(thisfeval["OutFileHeaders"][i], str):
                    del thisfeval["OutFileHeaders"][i]
                else:
                    if thisfeval["OutFileHeaders"][i].upper() == "NONE":
                        del thisfeval["OutFileHeaders"][i]
        else:
            thisfeval["OutFileHeaders"] = []

        if not isinstance(thisfeval["nproc"], int):
            thisfeval["nproc"] = 1

        if not isinstance(thisfeval["nproc4ReCal"], int):
            thisfeval["nproc4ReCal"] = 1

        if thisfeval["Style"].lower() == "pylammps":
            thisfeval["Bin"] = "pylammps"
        elif thisfeval["Style"].lower() == "lammps":
            thisfeval["Bin"] = "calllammps"
        elif thisfeval["Style"].lower() == "vasp":
            thisfeval["Bin"] = "callvasp"
        else:
            pass

        if isinstance(thisfeval["GPU"], list):
            GPU_dict = {}
            for i in range(0, len(thisfeval["GPU"])):
                thisstr = thisfeval["GPU"][i]
                thisstr = thisstr.split(":")
                GPU_dict[thisstr[0]] = thisstr[1].strip()
            thisfeval["GPU"] = GPU_dict
        ############################################################
        potential = parameters['potential']
        symbols = []
        if 'species' not in potential:
            raise ValueError("Potential input must contain species.")
        else:
            for i in range(len(potential["species"])):
                thisstr = potential["species"][i]
                thisstr = thisstr.replace("-", "")
                symbols.append(thisstr)

        ntype = len(symbols)
        if ntype < 1:
            raise ValueError("No species has been found.")

        OpenKIM = {"OpenKIM": False, "kim_init": False, "kim_interaction": False, "kim_param": False}
        if "OpenKIM" in potential:
            for key in potential["OpenKIM"]:
                OpenKIM[key] = potential["OpenKIM"][key]
            if OpenKIM["OpenKIM"]:
                if not isinstance(OpenKIM["kim_init"], str):
                    raise ValueError("kim_init must be a string!")
                if not isinstance(OpenKIM["kim_interaction"], str):
                    raise ValueError("kim_interaction must be a string!")

        if "pair_coeff" not in potential: potential["pair_coeff"] = False
        if not OpenKIM["OpenKIM"]:
            if "FileName" not in potential:
                if "LAMMPS" in thisfeval["Style"].upper():
                    if isinstance(potential["pair_coeff"], str):
                        pass
                    elif isinstance(potential["pair_coeff"], list):
                        pass
                    else:
                        raise ValueError("Potential input must contain a FileName or pair_coeff.")
                else:
                    pass
            if "pair_style" not in potential:
                if "LAMMPS" in thisfeval["Style"].upper():
                    raise ValueError("Potential input must contain pair_style.")
        else:
            if "FileName" not in potential:
                potential["FileName"] = "FileName"
            if "pair_style" not in potential:
                potential["pair_style"] = "eam"
        if "Path2Pot" not in potential:
            potential["Path2Pot"] = False

        force_field = {}
        for i in range(ntype):
            force_field[str(i + 1)] = symbols[i]

        bondlengths = np.zeros((ntype, ntype), dtype=float)
        for i in range(ntype):
            for j in range(ntype):
                bondlengths[i][j] = get_bo_one_length(symbols[i], symbols[j])

        coordnums = np.ones(ntype, dtype=float) + 3.0
        charges = np.zeros(ntype, dtype=float)
        qtolerances = np.zeros(ntype, dtype=float) + 1.0e-6
        masses = np.ones(ntype, dtype=float)
        for i in range(ntype):
            masses[i] = get_atomic_mass(symbols[i])

        if 'bondlengths' in potential:
            for ii in range(len(potential['bondlengths'])):
                thisstr = potential['bondlengths'][ii]
                thisstr = thisstr.split(",")
                try:
                    iele = int(thisstr[0].strip()) - 1
                    jele = int(thisstr[1].strip()) - 1
                    bondlengths[iele][jele] = float(thisstr[2].strip())
                    bondlengths[jele][iele] = bondlengths[iele][jele]
                except:
                    pass

        cutneighs = bondlengths * 1.1
        if 'cutneighs' in potential:
            for ii in range(len(potential['cutneighs'])):
                thisstr = potential['cutneighs'][ii]
                thisstr = thisstr.split(",")
                try:
                    iele = int(thisstr[0].strip()) - 1
                    jele = int(thisstr[1].strip()) - 1
                    cutneighs[iele][jele] = float(thisstr[2].strip())
                    cutneighs[jele][iele] = cutneighs[iele][jele]
                except:
                    pass

        if 'coordnums' in potential:
            for ii in range(len(potential['coordnums'])):
                thisstr = potential['coordnums'][ii]
                thisstr = thisstr.split(",")
                try:
                    iele = int(thisstr[0].strip()) - 1
                    coordnums[iele] = float(thisstr[1].strip())
                except:
                    pass

        if 'charges' in potential:
            for ii in range(len(potential['charges'])):
                thisstr = potential['charges'][ii]
                thisstr = thisstr.split(",")
                try:
                    iele = int(thisstr[0].strip()) - 1
                    charges[iele] = float(thisstr[1].strip())
                except:
                    pass

        if 'qtolerances' in potential:
            for ii in range(len(potential['qtolerances'])):
                thisstr = potential['qtolerances'][ii]
                thisstr = thisstr.split(",")
                try:
                    iele = int(thisstr[0].strip()) - 1
                    qtolerances[iele] = float(thisstr[1].strip())
                except:
                    pass

        if 'masses' in potential:
            for ii in range(len(potential['masses'])):
                thisstr = potential['masses'][ii]
                thisstr = thisstr.split(",")
                #iele = symbols.index(thisstr[0].strip())
                try:
                    itype = int(thisstr[0]) - 1
                    if itype < ntype:
                        try:
                            masses[itype] = float(thisstr[1].strip())
                        except:
                            pass
                    else:
                        pass
                except:
                    pass

        cutneighmax = max(cutneighs.flatten())
        if 'cutneighmax' in potential:
            try:
                cutneighmax = float(potential['cutneighmax'])
            except:
                pass

        bondlengths4LAS = copy.deepcopy(bondlengths)
        if 'bondlengths4LAS' in potential:
            for ii in range(len(potential['bondlengths4LAS'])):
                thisstr = potential['bondlengths4LAS'][ii]
                thisstr = thisstr.split(",")
                try:
                    iele = int(thisstr[0].strip()) - 1
                    jele = int(thisstr[1].strip()) - 1
                    bondlengths4LAS[iele][jele] = float(thisstr[2].strip())
                    bondlengths4LAS[jele][iele] = bondlengths4LAS[iele][jele]
                except:
                    pass

        coordnums4LAS = copy.deepcopy(coordnums)
        if 'coordnums4LAS' in potential:
            for ii in range(len(potential['coordnums4LAS'])):
                thisstr = potential['coordnums4LAS'][ii]
                thisstr = thisstr.split(",")
                try:
                    iele = int(thisstr[0].strip()) - 1
                    coordnums4LAS[iele] = float(thisstr[1].strip())
                except:
                    pass

        potential["species"] = symbols
        potential["force_field"] = force_field
        potential['ntype'] = len(symbols)
        potential['masses'] = masses
        potential['bondlengths'] = bondlengths
        potential['coordnums'] = coordnums
        potential['cutneighs'] = cutneighs
        potential['charges'] = charges
        potential['qtolerances'] = qtolerances
        potential['cutneighmax'] = max(cutneighs.flatten())
        potential['bondlengths4LAS'] = bondlengths4LAS
        potential['cutneighs4LAS'] = bondlengths4LAS * 1.1
        potential['coordnums4LAS'] = coordnums4LAS
        if "dangercut" not in potential:
            potential['dangercut'] = 0.5 * min(bondlengths.flatten())
        potential['OpenKIM'] = OpenKIM
        ###################################################################
        cutdefectmax = max(cutneighs.flatten())
        cutdefectmin = min(cutneighs.flatten())
        FindDefects = {"Method": "BLCN", "MolIDCap": "NA", "DiscardType": "NA"}
        sort_by = ["D", "X", "Y", "Z"]
        thisav = {"NPredef": 0, "PredefOnly": False, "Style": "defects",
                  "FindDefects": FindDefects, "cutdefectmax": cutdefectmax,
                  "DActive": cutdefectmax * 4.1, "DBuffer": cutdefectmax * 0.4, "DFixed": cutdefectmax * 4.1,
                  "RT_SetMolID": False, "DefectCenter4RT_SetMolID": "AUTO", "R4RT_SetMolID": 30,
                  "FCT4RT_SetMolID": ["INF", "INF", "INF", "INF", "INF", "INF"],
                  "NMax4Def": False, "NMax4AV": False, "NMin4AV": 40,
                  "PDReduction": True, "SortD4PDR": False, "DCut4PDR": cutdefectmax * 1.4, "RecursiveRed": False,
                  "Order4Recursive4PDR": None,
                  "DCut4noOverlap": 9.0 * cutdefectmax, "Overlapping": True, "Order4Recursive4AV": None,
                  "Overlap4OrderRecursive": True,
                  "Stack4noOverlap": False, "PointGroupSymm": False, "NMax4PG": 1000,
                  "Sorting": True, "Sort_by": sort_by, "SortingSpacer": [0.3, 0.3, 0.3],
                  "SortingShift": [0.0, 0.0, 0.0],
                  "SortingBuffer": False, "SortingFixed": False,
                  "TurnoffPBC": [False, False, False], "NMin_perproc": 5}

        active_volume = parameters['active_volume']
        if "Style" not in active_volume:
            active_volume["Style"] = "defects"
        if active_volume["Style"].upper() == 'CUSTOM':
            if "NActive" not in active_volume:
                raise ValueError("Must input NActive for custom style of constructing AV!")
            else:
                if "NBuffer" not in active_volume:
                    active_volume["NBuffer"] = 0
                if "NFixed" not in active_volume:
                    active_volume["NFixed"] = 0

        defects = []
        if "FindDefects" in active_volume:
            if "Method" in active_volume["FindDefects"]:
                if "WS" in active_volume["FindDefects"]["Method"].upper():
                    if "ReferenceData" not in active_volume["FindDefects"]:
                        raise ValueError("Must have 'ReferenceData' for WS method of FindDefects!")
                    if 'atom_style4Ref' not in parameters['active_volume']['FindDefects']:
                        active_volume['FindDefects']['atom_style4Ref'] = thisdata['atom_style']
                    if "DCut4Def" not in active_volume["FindDefects"]:
                        active_volume["FindDefects"]["DCut4Def"] = 0.2
                elif active_volume["FindDefects"]["Method"].upper() == "CUSTOM":
                    if "Defects" in active_volume["FindDefects"]:
                        defectstr = active_volume["FindDefects"]["Defects"]
                        for i in range(len(defectstr)):
                            thisdefect = defectstr[i].split(",")
                            try:
                                xyz = [float(thisdefect[0]), float(thisdefect[1]), float(thisdefect[2])]
                                defects.append(xyz)
                            except:
                                pass
                    if len(defects) <= 0:
                        raise ValueError("Must input coordinates of defects for custom FindDefect!")
                else:
                    if "DCut4Def" not in active_volume["FindDefects"]:
                        active_volume["FindDefects"]["DCut4Def"] = 0.1
        active_volume["FindDefects"]["Defects"] = defects

        for key in active_volume:
            if key == "FindDefects":
                for subkey in active_volume[key]:
                    thisav[key][subkey] = active_volume[key][subkey]
            elif key == "DefectCenter4RT_SetMolID":
                isValid = True
                if isinstance(active_volume[key], list):
                    if len(active_volume[key]) == 3:
                        for i in range(len(active_volume[key])):
                            if isinstance(active_volume[key][i], float):
                                thisav[key][i] = active_volume[key][i] - int(active_volume[key][i])
                            else:
                                isValid = False
                    else:
                        isValid = False
                else:
                    isValid = False
                if not isValid:
                    thisav[key] = "AUTO"
            elif key == "FCT4RT_SetMolID":
                for i in range(min(6, len(active_volume[key]))):
                    thisav[key][i] = active_volume[key][i]
            elif key == "Sort_by":
                if isinstance(active_volume[key], list):
                    t = []
                    for i in range(len(active_volume[key])):
                        if active_volume[key][i] in SORT_BY_KEYS:
                            t.append(active_volume[key][i])
                    if len(t) > 0:
                        thisav[key] = t[0:len(t)]
                    else:
                        thisav[key] = sort_by
            elif key == "SortingSpacer":
                if isinstance(active_volume[key], float) or isinstance(active_volume[key], int):
                    thisav[key] = [active_volume[key]] * 3
                elif isinstance(active_volume[key], list):
                    for i in range(min(3, len(active_volume[key]))):
                        thisav[key][i] = active_volume[key][i]
            elif key == "SortingShift":
                if isinstance(active_volume[key], float) or isinstance(active_volume[key], int):
                    thisav[key] = [active_volume[key]] * 3
                elif isinstance(active_volume[key], list):
                    for i in range(min(3, len(active_volume[key]))):
                        thisav[key][i] = active_volume[key][i]
            elif key == "TurnoffPBC":
                for i in range(min(6, len(active_volume[key]))):
                    thisav[key][i] = active_volume[key][i]
            else:
                thisav[key] = active_volume[key]

        thisav["boundary"] = thisdata["boundary"]
        tmpstr = thisav["boundary"].strip()
        tmpstrs = list(filter(None, list(map(lambda strings: strings.strip(), tmpstr.split(" ")))))
        thisav["ResetBounds"] = False
        for i in range(3):
            if thisav["TurnoffPBC"][i]:
                tmpstrs[i] = "f"
                thisav["ResetBounds"] = True
        if thisav["ResetBounds"]:
            thisav["boundary"] = tmpstrs[0] + " " + tmpstrs[1] + " " + tmpstrs[2]
        #######################################################
        spsearch_force_evaluator = {}
        for key in thisfeval:
            if key != "Relaxation":
                spsearch_force_evaluator[key] = thisfeval[key]
        spsearch_force_evaluator["nproc"] = 1
        spsearch_force_evaluator["processors"] = False
        spsearch_force_evaluator["partition"] = False

        spsearch_force_evaluator["RinputOpt"] = False
        spsearch_force_evaluator["Rinput"] = False
        spsearch_force_evaluator["RinputDM"] = False

        LocalRelax = {"LocalRelax": True, "InitTemp4Opt": 0.0, "TargetTemp4NVT": 5.0, "NVTSteps4Opt": 1000}
        Preloading = {"Preload": False, "LoadPath": False, "Ratio4DispLoad": 0.8, "SortDisps": False, "Method": "Files",
                      "FileHeader": "SPS_AV_", "CheckSequence": False, "FileHeader4Data": "SPS_basin_", "Scaling": 1.0,
                      "IgnoreType": True}
        HandleVN = {"CheckAng4Init": True, "AngTol4Init": 5.0, "MaxIter4Init": 20, "NMaxRandVN": 20,
                    "CenterVN": False, "NSteps4CenterVN": 5, "IgnoreSteps": 4,
                    "RescaleVN": True, "RescaleValue": "LOGN", "Int4ComputeScale": 1, "TakeMin4MixedRescales": True,
                    "RescaleStyle4LOGV": "SIGMOID", "Period4MA": 1, "XRange4LOGV": 20.0, "PowerOnV": 4,
                    "Ratio4Zero4LOGV": 0.2, "MinValue4LOGV": -20.0,
                    "RescaleStyle4RAS": "SIGMOID", "XRange4RAS": 40.0, "Ratio4Zero4RAS": 0.3,
                    "MinSpan4LOGV": 4.0, "MinSpan4RAS": 40.0,
                    "ResetVN04Preload": True, "RatioVN04Preload": 0.2}
        thisspsearch = {"Method": "dimer", "NSearch": 10, "SearchBuffer": False, "NMax4Trans": 1000, "IgnoreSteps": 4,
                        "DimerSep": 0.005, "NMax4Rot": 3, "FThres4Rot": 0.1, "FMin4Rot": 0.001, "TrialStepsize": 0.015,
                        "MaxStepsize": 0.05,
                        "DRatio4Relax": 2.0, "FConv": 1e-6, "EnConv": 1e-5, "En4TransHorizon": 0.1,
                        "TransHorizon": True,
                        "CheckAng": True, "CheckAngSteps": 50, "AngCut": 2.0,
                        "DecayStyle": "Fixed", "MinStepsize": 0.003, "DecayRate": 0.71, "DecaySteps": 20,
                        "Tol4Connect": 0.1, "FixTypes": False, "FixAxesStr": "ALL",
                        "ActiveOnly4SPConfig": True, "R2Dmax4SPAtom": 0.04, "DCut4SPAtom": 0.01, "DynCut4SPAtom": False,
                        "ShowIterationResults": False, "Inteval4ShowIterationResults": 1,
                        "ShowVN4ShowIterationResults": False, "ShowCoords4ShowIterationResults": False,
                        "OutForces4IterationResults": False, "OutFix4IterationResults": False,
                        "ApplyMass": False, "TaskDist": "AV", "Master_Slave": True,
                        "LocalRelax": LocalRelax, "force_evaluator": spsearch_force_evaluator, "Preloading": Preloading,
                        "HandleVN": HandleVN}

        spsearch = parameters['spsearch']
        for key in spsearch:
            if key == "force_evaluator":
                for subkey in spsearch[key]:
                    thisspsearch[key][subkey] = spsearch[key][subkey]
            elif key == "Preloading":
                for subkey in spsearch[key]:
                    thisspsearch[key][subkey] = spsearch[key][subkey]
            elif key == "LocalRelax":
                for subkey in spsearch[key]:
                    thisspsearch[key][subkey] = spsearch[key][subkey]
            elif key == "HandleVN":
                for subkey in spsearch[key]:
                    thisspsearch[key][subkey] = spsearch[key][subkey]
            else:
                thisspsearch[key] = spsearch[key]
        thisspsearch["RatioStepsize"] = thisspsearch["MaxStepsize"] / thisspsearch["TrialStepsize"]
        ################################################################################
        thisKMC = {"NSteps": 1, "Temp": 800.0, "Temp4Time": 800.0, "AccStyle": "NoAcc", "NMaxBasin": "NA",
                   "Tol4Disp": 0.1, "Tol4Barr": 0.03,
                   "EnCut4Transient": 0.5, "Handle_no_Backward": "Out", "DispStyle": "FI", "Sorting": False}
        if "kinetic_MC" in parameters:
            KMC = parameters['kinetic_MC']
            for key in KMC:
                thisKMC[key] = KMC[key]
            if "Temp4Time" not in KMC:
                try:
                    thisKMC["Temp4Time"] = KMC["Temp"]
                except:
                    pass
        #################################################################################
        thisDynMat = {"SNC": False, "NMax4SNC": 1000, "displacement": 0.000001,
                      "delimiter": " ", "LowerHalfMat": False, "OutDynMat": False,
                      "CalPrefactor": False, "Method4Prefactor": "harmonic", "VibCut": 1.0e-8}
        if "dynamic_matrix" in parameters:
            DynMat = parameters['dynamic_matrix']
            for key in DynMat:
                thisDynMat[key] = DynMat[key]
        #################################################################################
        thisDefectBank = {"Preload": False, "NMax4DB": 100, "NMin4DB": 8,
                          "Scaling": 1.0, "Ratio4DispLoad": 0.8, "IgnoreType": True, "Tol4Disp": 0.1,
                          "FileHeader": "DB", "LoadDB": False, "LoadPath": "DefectBank", "SortDisps": False,
                          "Recycle": False, "UseSymm": False, "SaveDB": False, "SavePath": "DefectBank",
                          "OutIndex": True}
        if "defect_bank" in parameters:
            DefectBank = parameters['defect_bank']
            for key in DefectBank:
                thisDefectBank[key] = DefectBank[key]
        #################################################################################
        ScreenDisp = {"AND4ScreenD": [True], "Str4ScreenD": ["SP"], "Type4ScreenD": ["DMAG"], "AbsVal4ScreenD": [True],
                      "MinVal4ScreenD": ["NA"], "MaxVal4ScreenD": ["NA"]}
        ScreenEng = {"AND4ScreenE": [True], "Type4ScreenE": ["barrier"], "MinVal4ScreenE": ["NA"],
                     "MaxVal4ScreenE": ["NA"]}
        ValidSPs = {"RealtimeValid": False, "RealtimeDelete": False, "CheckConnectivity": True,
                    "toScreenDisp": "NotConn", "NScreenDisp": 0, "ScreenDisp": ScreenDisp,
                    "toScreenEng": "NotConn", "NScreenEng": 0, "ScreenEng": ScreenEng,
                    "AND4ScreenDE": True,
                    "GroupSP": False, "AngCut4GSP": 10.0, "MagCut4GSP": 0.1, "EnCut4GSP": 0.1,
                    "FindSPType": False, "EnCut4Type": 0.05, "MagCut4Type": 0.05, "LenCut4Type": 0.05,
                    "AngCut4Type": 5.0,
                    "MaxRatio4Dmag": "NA", "MaxRatio4Barr": "NA",
                    "EnTol4AVSP": 0.1, "Tol4AVSP": 0.1, "NMax4Dup": 600, "NCommonMin": 10, "R2Dmax4Tol": 0.1,
                    "Tol4Disp": 0.1}

        thissp = {"BarrierCut": 10.0, "BarrierMin": 0.0, "EbiasCut": "NA", "EbiasMin": "NA", "DAtomCut": cutneighmax,
                  "BackBarrierMin": 0.0, "DmagCut": "NA", "DmagMin": 0.0,
                  "DtotCut": "NA", "DtotMin": 0.0, "DmaxCut": "NA", "DmaxMin": 0.0,
                  "DsumCut": "NA", "DsumMin": 0.0, "DsumrCut": "NA", "DsumrMin": 0.0,
                  "DmagCut_FI": "NA", "DmagMin_FI": 0.0,
                  "DtotCut_FI": "NA", "DtotMin_FI": 0.0, "DmaxCut_FI": "NA", "DmaxMin_FI": 0.0,
                  "DsumCut_FI": "NA", "DsumMin_FI": 0.0, "DsumrCut_FI": "NA", "DsumrMin_FI": 0.0,
                  "DmagCut_FS": "NA", "DmagMin_FS": 0.0,
                  "DtotCut_FS": "NA", "DtotMin_FS": 0.0, "DmaxCut_FS": "NA", "DmaxMin_FS": 0.0,
                  "DsumCut_FS": "NA", "DsumMin_FS": 0.0, "DsumrCut_FS": "NA", "DsumrMin_FS": 0.0,
                  "Prefactor": 10.0, "CalBarrsInData": False, "CalEbiasInData": False, "Thres4Recalib": None,
                  "ValidSPs": ValidSPs}

        saddlepoint = parameters["saddle_point"]
        for key in saddlepoint:
            if key == "ValidSPs":
                for subkey in saddlepoint[key]:
                    if subkey == "ScreenDisp":
                        for thiskey in saddlepoint[key][subkey]:
                            thissp[key][subkey][thiskey] = saddlepoint[key][subkey][thiskey]
                    elif subkey == "ScreenEng":
                        for thiskey in saddlepoint[key][subkey]:
                            thissp[key][subkey][thiskey] = saddlepoint[key][subkey][thiskey]
                    else:
                        thissp[key][subkey] = saddlepoint[key][subkey]

            else:
                thissp[key] = saddlepoint[key]

        try:
            nScreenDisp = int(thissp["ValidSPs"]["NScreenDisp"])
        except:
            nScreenDisp = 0
        try:
            nScreenEng = int(thissp["ValidSPs"]["NScreenEng"])
        except:
            nScreenEng = 0
        if nScreenDisp > 0:
            thisScreenDisp = {"AND4ScreenD": [False] * nScreenDisp, "Str4ScreenD": ["SP"] * nScreenDisp,
                              "Type4ScreenD": ["DMAG"] * nScreenDisp, "AbsVal4ScreenD": [True] * nScreenDisp,
                              "MinVal4ScreenD": ["NA"] * nScreenDisp, "MaxVal4ScreenD": ["NA"] * nScreenDisp}
            for key in thisScreenDisp:
                if key in thissp["ValidSPs"]["ScreenDisp"]:
                    if len(thissp["ValidSPs"]["ScreenDisp"][key]) < nScreenDisp:
                        for i in range(len(thissp["ValidSPs"]["ScreenDisp"][key]), nScreenDisp):
                            thissp["ValidSPs"]["ScreenDisp"][key].append(thisScreenDisp[key][i])
                else:
                    thissp["ValidSPs"]["ScreenDisp"][key] = thisScreenDisp[key]

        if nScreenEng > 0:
            thisScreenEng = {"AND4ScreenE": [False] * nScreenEng, "Type4ScreenE": ["barrier"] * nScreenEng,
                             "MinVal4ScreenE": ["NA"] * nScreenEng, "MaxVal4ScreenE": ["NA"] * nScreenEng}
            for key in thisScreenEng:
                if key in thissp["ValidSPs"]["ScreenEng"]:
                    if len(thissp["ValidSPs"]["ScreenEng"][key]) < nScreenEng:
                        for i in range(len(thissp["ValidSPs"]["ScreenEng"][key]), nScreenEng):
                            thissp["ValidSPs"]["ScreenEng"][key].append(thisScreenEng[key][i])
                else:
                    thissp["ValidSPs"]["ScreenEng"][key] = thisScreenEng[key]

        thissp["ValidSPs"]["CosAngCut4GSP"] = 1.0 - np.cos(thissp["ValidSPs"]["AngCut4GSP"] * pi / 180.0)
        thissp["ValidSPs"]["CosAngCut4Type"] = 1.0 - np.cos(thissp["ValidSPs"]["AngCut4Type"] * pi / 180.0)
        ##############################################################################
        Write_Data_SPs = {"Write_Data_SPs": True, "DispStyle4DataSP": "Both", "OutputStyle": "SEP", "Sel_iSPs": "AUTO",
                          "Offset": 0,
                          "Write_Data_AVs": True, "Write_KMC_Data": True, "Write_Prob": True, "DetailOut": True,
                          "SPs4Detail": "AUTO"}
        Write_AV_SPs = {"Write_Local_AV": False, "Write_AV_SPs": False, "Write_Data_AV_SPs": False,
                        "DispStyle4AVSP": "SP"}
        thisvisual = {"Screen": True, "Log": True, "Write_SP_Summary": True,
                      "RCut4Vis": 0.04, "DCut4Vis": 0.01, "Invisible": True, "Reset_Index": False, "ShowBuffer": False,
                      "ShowFixed": False,
                      "Write_Data_SPs": Write_Data_SPs, "Write_AV_SPs": Write_AV_SPs}

        if "visual" in parameters:
            visual = parameters['visual']
            if "Write_Data_SPs" in visual and "Sel_iSPs" in visual["Write_Data_SPs"]:
                SelSPs = visual["Write_Data_SPs"]["Sel_iSPs"]
                if SelSPs.upper() == "AUTO":
                    pass
                elif SelSPs.upper() == "ALL":
                    pass
                elif SelSPs.upper() == "TRANSIENT":
                    pass
                elif SelSPs.upper() == "ABSORB":
                    pass
                else:
                    tmpl = []
                    tmps = SelSPs.split(",")
                    for i in range(len(tmps)):
                        try:
                            tmpl.append(int(tmps[i].strip()))
                        except:
                            pass
                    visual["Write_Data_SPs"]["Sel_iSPs"] = tmpl

            for key in visual:
                if key == "Write_Data_SPs":
                    for subkey in visual[key]:
                        thisvisual[key][subkey] = visual[key][subkey]
                elif key == "Write_AV_SPs":
                    for subkey in visual[key]:
                        thisvisual[key][subkey] = visual[key][subkey]
                else:
                    thisvisual[key] = visual[key]
        thissett = cls(thissystem, thisfeval, potential, thisdata, thisKMC, thisav, thisDefectBank, thisDynMat,
                       thisspsearch, thissp, thisvisual)

        return thissett
        ##return cls(**items)

    def validate_input(self):
        if self.force_evaluator["Style"].upper() == "VASP":
            logstr = f"The binary file for VASP style is 'callvasp', which is a submission script!"
            logstr += "\n" + f"Use the absolute path in 'Path2Pot' in potential!"
            logstr += "\n" + f"INCAR files should be provided for the correspond places with the header 'Rinput'!"
            logstr += "\n" + f"The force_evaluator-RinputOpt as 'INCAR_Opt' for relaxating between KMC steps."
            logstr += "\n" + f"The spsearch-force_evaluator-Rinput as 'INCAR_SPS' for SPS."
            logstr += "\n" + f"Two sets of KPOINTS_DATA, KPOINTS_SPS must be provided, "
            logstr += "\n" + f"where DATA is referred to the whole data and SPS is refrred to the AV for SPS."
            logstr += "\n" + f"The potential-FileName is not a string, provide the POTCAR_symbol in potential-Path2Pot!"
            logstr += "\n" + f"Code will generate POTCAR based on input structure!"
            print(logstr)

        if self.saddle_point["CalBarrsInData"] and self.saddle_point["CalEbiasInData"]:
            self.spsearch["LocalRelax"]["LocalRelax"] = True
        if self.force_evaluator["Relaxation"]["BoxRelax"] and self.kinetic_MC["AccStyle"][0:3].upper() == "MRM":
            self.kinetic_MC["AccStyle"] = "NOACC"
        if self.spsearch["LocalRelax"]["LocalRelax"] is False and self.kinetic_MC["AccStyle"][0:3].upper() == "MRM":
            self.kinetic_MC["AccStyle"] = "NOACC"
        if self.defect_bank["SaveDB"]:
            self.defect_bank["Recycle"] = True
        if self.data["dimension"] != 3:
            raise ValueError("The data must be 3 dimensional!")

        if self.active_volume["RT_SetMolID"] and self.active_volume["NPredef"] == 0:
            self.active_volume["NPredef"] = 1
        if isinstance(self.active_volume["Order4Recursive4PDR"], int):
            if self.active_volume["Order4Recursive4PDR"] < 1:
                errormsg = "The Order4Recursive4PDR must be >= 1!"
                raise ValueError(errormsg)
        if isinstance(self.active_volume["Order4Recursive4AV"], int):
            if self.active_volume["Order4Recursive4AV"] < 1:
                errormsg = "The Order4Recursive4AV must be >= 1!"
                raise ValueError(errormsg)
        if not self.active_volume["Overlapping"]:
            if self.spsearch["SearchBuffer"]:
                if (self.active_volume["DCut4noOverlap"] <=
                        self.active_volume["DActive"] + self.active_volume["DBuffer"]):
                    raise ValueError("DCut4noOverlap must be larger than DActive+DBuffer!")
            else:
                if self.active_volume["DCut4noOverlap"] <= self.active_volume["DActive"]:
                    raise ValueError("DCut4noOverlap must be larger than DActive!")

        if self.kinetic_MC["AccStyle"][0:3].upper() == "MRM":
            if isinstance(self.kinetic_MC["EnCut4Transient"], float) or isinstance(self.kinetic_MC["EnCut4Transient"],
                                                                                   int):
                pass
            else:
                raise ValueError("EnCut4Transient in kineticMC must be float number!")
        if self.saddle_point["CalEbiasInData"] and self.saddle_point["CalBarrsInData"] is False:
            raise ValueError("CalBarrsInData in saddle_point must be True if CalEbiasInData is True!")

        if self.spsearch["MaxStepsize"] <= self.spsearch["TrialStepsize"]:
            raise ValueError("The MaxStepsize must be larger than TrialStepsize!")

        if isinstance(self.spsearch["FixTypes"], list):
            errormsg = ""
            n = len(self.spsearch["FixTypes"])
            isValid = True
            FixAxes = []
            FixTypes_dict = None
            if isinstance(self.spsearch["FixAxesStr"], list):
                m = len(self.spsearch["FixAxesStr"])
                if n != m:
                    isValid = False
                if isValid:
                    for i in range(n):
                        fixtype = self.spsearch["FixTypes"][i]
                        if isinstance(fixtype, int):
                            thisstr = self.spsearch["FixAxesStr"][i]
                            thisstr = str(thisstr)
                            thisstr = thisstr.strip()
                            thisstr = thisstr.split(",")
                            thisFixAxes = []
                            for j in range(len(thisstr)):
                                try:
                                    axis = int(thisstr[j])
                                    if axis >= 0 and axis < 3:
                                        pass
                                    else:
                                        axis = -1
                                        errormsg += "\n" + "The axis index in FixAxesStr must be 0, 1 or 2."
                                        isValid = False
                                except:
                                    axis = -1
                                    errormsg += "\n" + ("FixAxesStr must be a string with axis index numbers"
                                                        " separating with commas.")
                                    isValid = False
                                thisFixAxes.append(axis)
                            thisFixAxes = list(dict.fromkeys(thisFixAxes))
                            FixAxes.append(thisFixAxes)
                            if not isValid:
                                break
                        else:
                            errormsg += "\n" + ("FixTypes must be an integer ranging from 1 to the number of"
                                                " types of the data.")
                            isValid = False
                            break
            else:
                a = [0, 1, 2]
                for i in range(n):
                    FixAxes.append(a)
            if isValid:
                FixTypes_dict = {}
                for i in range(n):
                    FixTypes_dict[self.spsearch["FixTypes"][i]] = FixAxes[i]
                self.spsearch["FixTypes_dict"] = FixTypes_dict
            else:
                raise ValueError(errormsg)
        else:
            self.spsearch["FixTypes_dict"] = None

        if self.saddle_point["BarrierCut"] > 10.0:
            logstr = f"Warning: The BarrierCut in saddle_point is larger than 10 eV!"
            logstr += "\n" + "This vibrational rate may be considered as 0.0 in code."
            logstr += "\n" + ("The probability distribution in KMC CANNOT differ the high barrier"
                              " event from its neighboring events.")
            logstr += "\n" + "This may cause unrealistic selection of the high barrier event."
            print(logstr)

    def reset_settings(self, key, value):
        if key == 'system':
            self.system = value
        elif key == 'potential':
            self.potential = value
        elif key == 'force_evaluator':
            self.force_evaluator = value
        elif key == 'data':
            self.data = value
        elif key == 'active_volume':
            self.active_volume = value
        elif key == 'spsearch':
            self.spsearch = value
        elif key == 'visual':
            self.visual = value
        elif key == 'kinetic_MC':
            self.kinetic_MC = value
        elif key == 'defect_bank':
            self.defect_bank = value
        elif key == 'dynamic_matrix':
            self.dynamic_matrix = value
        elif key == 'saddle_point':
            self.saddle_point = value
        else:
            pass

############################################
