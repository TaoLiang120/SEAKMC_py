import os
import numpy as np
import copy
import shutil

import subprocess
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Outcar

from seakmc.input.Input import export_Keys

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


class VaspRunner(object):
    def __init__(self, sett):
        self.name = 'vasp'
        self.sett = sett
        self.callscript = self.sett.force_evaluator['Bin']
        if isinstance(self.sett.force_evaluator['Path2Bin'], str):
            self.path_to_callscript = self.sett.force_evaluator['Path2Bin']
        else:
            self.path_to_callscript = os.getcwd()

        if isinstance(self.sett.potential['Path2Pot'], str):
            self.path_to_pot = self.sett.potential['Path2Pot']
        else:
            self.path_to_pot = os.getcwd()

        if not os.path.isfile(os.path.join(self.path_to_callscript, self.callscript)):
            print(f"Cannot find {os.path.join(self.path_to_callscript, self.callscript)} !")
            quit()

    def run_runner(self, purpose, data, thiscolor, nactive=None, thisExports=None):

        purpose = purpose.upper()
        #nproc_task = self.get_nproc_task(purpose)
        nproc_task = 1
        self.this_path = os.getcwd()
        self.relative_path = "Runner_" + str(thiscolor)

        #self.modify_callscript(os.path.join(self.path_to_callscript, self.callscript), nproc=nproc_task)
        if os.path.isfile(self.relative_path + "/POSCAR"): os.remove(self.relative_path + "/POSCAR")
        if os.path.isfile(self.relative_path + "/CONTCAR"): os.remove(self.relative_path + "/CONTCAR")
        if os.path.isfile(self.relative_path + "/OUTCAR"): os.remove(self.relative_path + "/OUTCAR")
        data.to_POSCAR(self.relative_path + "/POSCAR", Direct=False, SortElements=False)
        self.preparation(purpose, data, thiscolor, nactive=nactive, thisExports=thisExports, nproc=nproc_task)

        errormsg = ""
        isValid = True
        total_energy = 0.0
        relaxed_coords = []
        job_dir_path = os.path.join(self.this_path, self.relative_path)
        devnull = open(os.devnull, 'w')
        try:
            subprocess.call([os.path.join(self.path_to_callscript, self.callscript), job_dir_path], stdout=devnull,
                            stderr=devnull)
        except:
            isValid = False
            errormsg = f"Error on running VASP!"
            errormsg += ("\n" +
                         f"Job - purpose: {purpose} datatype:{type(data)} thiscolor: {thiscolor} nactive:{nactive}!")

        if isValid:
            total_energy = self.get_total_energy()
            if total_energy == 0:
                isValid = False
                errormsg = f"Error on getting energy in VASP!"
                errormsg += ("\n" +
                             f"Job - purpose: {purpose} datatype:{type(data)} thiscolor:{thiscolor} nactive:{nactive}!")

        if isValid:
            if purpose == "SPSOPT" or purpose == "SPSRELAX":
                relaxed_coords = self.get_relaxed_coords()
                if len(relaxed_coords) == 0:
                    isValid = False
                    errormsg = f"Error on getting coordinates in VASP!"
                    errormsg += ("\n" +
                                 f"Job - purpose:{purpose} datatype:{type(data)} "
                                 f"thiscolor{thiscolor} nactive:{nactive}!")

        return [total_energy, relaxed_coords, isValid, errormsg]

    def init_spsearch_runner(self, data, thiscolor, nactive):
        purpose = "SPS"
        #nproc_task = self.get_nproc_task(purpose)
        nproc_task = 1
        self.this_path = os.getcwd()
        self.relative_path = "Runner_" + str(thiscolor)

        #self.modify_callscript(os.path.join(self.path_to_callscript, self.callscript), nproc=nproc_task)
        if os.path.isfile(self.relative_path + "/POSCAR"): os.remove(self.relative_path + "/POSCAR")
        if os.path.isfile(self.relative_path + "/CONTCAR"): os.remove(self.relative_path + "/CONTCAR")
        if os.path.isfile(self.relative_path + "/OUTCAR"): os.remove(self.relative_path + "/OUTCAR")
        data.to_POSCAR(self.relative_path + "/POSCAR", Direct=False, SortElements=False)
        self.preparation(purpose, data, thiscolor, nactive=nactive, nproc=nproc_task)

        errormsg = ""
        isValid = True
        total_energy = 0.0
        relaxed_coords = []
        job_dir_path = os.path.join(self.this_path, self.relative_path)
        devnull = open(os.devnull, 'w')
        try:
            subprocess.call([os.path.join(self.path_to_callscript, self.callscript), job_dir_path], stdout=devnull,
                            stderr=devnull)
        except:
            isValid = False
            errormsg = f"Error on initializing VASP!"
            errormsg += ("\n" +
                         f"Job - purpose:{purpose} datatype:{type(data)} thiscolor:{thiscolor} nactive:{nactive}!")

        if isValid:
            total_energy = self.get_total_energy()
            if total_energy == 0:
                isValid = False
                errormsg = f"Error on getting energy in VASP!"
                errormsg += ("\n" +
                             f"Job - purpose: {purpose} datatype:{type(data)} thiscolor:{thiscolor} nactive:{nactive}!")

        return [total_energy, [], isValid, errormsg]

    def get_spsearch_forces(self, coords, data, thiscolor, nactive):
        purpose = 'SPS'
        if os.path.isfile(self.relative_path + "/POSCAR"): os.remove(self.relative_path + "/POSCAR")
        if os.path.isfile(self.relative_path + "/CONTCAR"): os.remove(self.relative_path + "/CONTCAR")
        if os.path.isfile(self.relative_path + "/OUTCAR"): os.remove(self.relative_path + "/OUTCAR")
        thisdata = copy.deepcopy(data)
        thisdata.update_coords(coords.T)
        thisdata.to_POSCAR(self.relative_path + "/POSCAR", Direct=False, SortElements=False)

        job_dir_path = os.path.join(self.this_path, self.relative_path)
        devnull = open(os.devnull, 'w')

        errormsg = ""
        isValid = True
        total_energy = 0.0
        forces = []
        try:
            subprocess.call([os.path.join(self.path_to_callscript, self.callscript), job_dir_path], stdout=devnull,
                            stderr=devnull)
        except:
            isValid = False
            errormsg = f"Error on running VASP!"
            errormsg += ("\n" +
                         f"Job - purpose:{purpose} datatype:{type(data)} thiscolor:{thiscolor} nactive:{nactive}!")

        if isValid:
            total_energy = self.get_total_energy()
            if total_energy == 0:
                isValid = False
                errormsg = f"Error on getting energy in VASP!"
                errormsg += ("\n" +
                             f"Job - purpose:{purpose} datatype:{type(data)} thiscolor:{thiscolor} nactive:{nactive}!")
        if isValid:
            forces = self.get_forces()
            if len(forces) == 0:
                isValid = False
                errormsg = f"Error on getting forces in VASP!"
                errormsg += ("\n" +
                             f"Job - purpose:{purpose} datatype:{type(data)} thiscolor:{thiscolor} nactive:{nactive}!")

        return [total_energy, forces, isValid, errormsg]

    def preparation(self, purpose, data, thiscolor, nactive=None, thisExports=None, nproc=1):
        if purpose == "DATAMD":
            if isinstance(self.sett.data["RinputMD"], str):
                rinputs = self.read_input_script(self.sett.data["RinputMD"])
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive=nactive)
            shutil.copy("KPOINTS_DATA", self.relative_path + "/KPOINTS")
        elif purpose == "DATAMD0":
            if isinstance(self.sett.data["RinputMD0"], str):
                rinputs = self.read_input_script(self.sett.data["RinputMD0"])
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive=nactive)
            shutil.copy("KPOINTS_DATA", self.relative_path + "/KPOINTS")
        elif purpose == "DATAOPT" or purpose == "DATARELAX":
            if isinstance(self.sett.data["RinputOpt"], str):
                rinputs = self.read_input_script(self.sett.data["RinputOpt"])
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive=nactive)
            shutil.copy("KPOINTS_DATA", self.relative_path + "/KPOINTS")
        elif purpose == "MD0" or "RECAL" in purpose:
            if isinstance(self.sett.force_evaluator["RinputMD0"], str):
                rinputs = self.read_input_script(self.sett.force_evaluator["RinputMD0"])
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive=nactive)
            shutil.copy("KPOINTS_DATA", self.relative_path + "/KPOINTS")
        elif purpose == "OPT" or purpose == "RELAX":
            if isinstance(self.sett.force_evaluator["RinputOpt"], str):
                rinputs = self.read_input_script(self.sett.force_evaluator["RinputOpt"])
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive=nactive)
            shutil.copy("KPOINTS_DATA", self.relative_path + "/KPOINTS")
        elif purpose == "SPSDYNMAT" or purpose == "DYNMAT":
            if isinstance(self.sett.spsearch["force_evaluator"]["RinputDM"], str):
                rinputs = self.read_input_script(self.sett.spsearch["force_evaluator"]["RinputDM"])
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive=nactive)
            shutil.copy("KPOINTS_SPS", self.relative_path + "/KPOINTS")
        elif purpose == "SPS":
            if isinstance(self.sett.spsearch["force_evaluator"]["Rinput"], str):
                rinputs = self.read_input_script(self.sett.spsearch["force_evaluator"]["Rinput"])
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive=nactive)
            shutil.copy("KPOINTS_SPS", self.relative_path + "/KPOINTS")
        elif purpose == "SPSOPT" or purpose == "SPSRELAX":
            if isinstance(self.sett.spsearch["force_evaluator"]["RinputOpt"], str):
                rinputs = self.read_input_script(self.sett.spsearch["force_evaluator"]["RinputOpt"])
            else:
                rinputs = self.get_default_inputs(purpose, data, thiscolor, nactive=nactive)
            shutil.copy("KPOINTS_SPS", self.relative_path + "/KPOINTS")

        if os.path.isfile(self.relative_path + "/INCAR"): os.remove(self.relative_path + "/INCAR")
        with open(self.relative_path + "/INCAR", "w") as f:
            for line in rinputs:
                f.write(line)

        if isinstance(self.sett.force_evaluator["POTCAR"], str):
            shutil.copy(self.sett.force_evaluator["POTCAR"], self.relative_path + "/POTCAR")
        else:
            self.get_default_potcar()

    def get_default_inputs(self, purpose, data, thiscolor, nactive=None):
        if nactive is None:
            try:
                nactive = data.nactive
            except:
                nactive = data.natoms
        lines = []
        ###add "\n" at the end of each line
        if purpose == "DATAMD":
            pass
        elif purpose == "DATAMD0":
            pass
        elif purpose == "DATAOPT" or purpose == "DATARELAX":
            pass
        elif purpose == "MD0" or "RECAL" in purpose:
            pass
        elif purpose == "OPT" or purpose == "RELAX":
            pass
        elif purpose == "SPSDYNMAT" or purpose == "DYNMAT":
            pass
        elif purpose == "SPS":
            pass
        elif purpose == "SPSOPT" or purpose == "SPSRELAX":
            pass

        return lines

    def read_input_script(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
        return lines

    def get_default_potcar(self):
        if os.path.isfile(self.relative_path + "/" + "POSCAR"):
            with open(os.path.isfile(self.relative_path + "/" + "POSCAR"), 'r') as f:
                lines = f.readlines()
            symbols = list(filter(None, list(map(lambda strings: strings.strip(), lines[5].split(" ")))))
            symbols = list(set(symbols))
            this_potcar = self.relative_path + "/" + "POTCAR"
            with open(this_potcar, 'w') as this_potcar_file:
                for symbol in symbols:
                    try:
                        with open(os.path.join(self.path_to_pot, "POTCAR" + "_" + symbol), 'r') as potcar_file:
                            for line in potcar_file:
                                this_potcar_file.write(line)
                    except:
                        print(f"Cannot find {os.path.join(self.path_to_pot, 'POTCAR' + '_' + symbol)} POTCAR!")
                        quit()
        else:
            print(f"Cannot find {self.relative_path + '/' + 'POSCAR'} to make POTCAR!")
            quit()
        return

    def get_nproc_task(self, purpose):
        if "SPS" in purpose or "DYNMAT" in purpose:
            nproc_task = self.sett.spsearch['force_evaluator']['nproc']
        elif purpose == "MD0" or "RECAL" in purpose:
            nproc_task = self.sett.force_evaluator['nproc4ReCal']
        else:
            nproc_task = self.sett.force_evaluator['nproc']
        return nproc_task

    def modify_callscript(self, filename, nproc=1):
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                lines = f.readlines()
            for i in range(len(lines)):
                if "-np" in lines[i]:
                    tmpstrs = list(filter(None, list(map(lambda strings: strings.strip(), lines[i].split(" ")))))
                    ind = tmpstrs.index("-np")
                    tmpstrs[ind + 1] = str(nproc)
                    line = ""
                    for j in range(len(tmpstrs)):
                        line += tmpstrs[j] + " "
                    lines[i] = line + "\n"
                elif "-n" in lines[i]:
                    tmpstrs = list(filter(None, list(map(lambda strings: strings.strip(), lines[i].split(" ")))))
                    ind = tmpstrs.index("-n")
                    tmpstrs[ind + 1] = str(nproc)
                    line = ""
                    for j in range(len(tmpstrs)):
                        line += tmpstrs[j] + " "
                    lines[i] = line + "\n"
            with open(filename, "w") as f:
                for line in lines:
                    f.write(line)
        else:
            print(f"Cannot find {filename} !")
            quit()

    def get_total_energy(self):
        if os.path.isfile(self.relative_path + "/" + "OUTCAR"):
            converged = False
            with open(self.relative_path + '/OUTCAR') as f:
                for line in f:
                    if 'reached' in line and 'required' in line and \
                            'accuracy' in line:
                        converged = True
            if not converged:
                return 0.0
            pv = 0
            with open(self.relative_path + "/" + "OUTCAR", 'r') as f:
                for line in f:
                    if 'energy(sigma->0)' in line:
                        u = float(line.split()[-1])
                    elif 'enthalpy' in line:
                        pv = float(line.split()[-1])
            return u + pv
        else:
            return 0.0

    def get_relaxed_coords(self):
        if os.path.isfile(self.relative_path + "/" + "CONTCAR"):
            thisstr = Structure.from_file(self.relative_path + "/" + "CONTCAR")
            coords = np.array(thisstr.cart_coords)
            return coords.flatten()
        else:
            return np.array([])

    def get_forces(self):
        if os.path.isfile(self.relative_path + "/" + "OUTCAR"):
            outcar = Outcar(self.relative_path + "/" + "OUTCAR")
            forces = outcar.read_table_pattern(
                header_pattern=r"\sPOSITION\s+TOTAL-FORCE \(eV/Angst\)\n\s-+",
                row_pattern=r"\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)",
                footer_pattern=r"\s--+",
                postprocess=lambda x: float(x),
                last_one_only=True)
            forces = np.array(forces)
            return forces.flatten()
        else:
            return np.array([])

    def ImportValue4RinputOpt(self, rinputs, thisExports=[]):
        isValid = True
        if not self.sett.force_evaluator['ImportValue4RinputOpt']: isValid = False
        if thisExports is None:
            isValid = False
        elif len(thisExports) == 0:
            isValid = False
        if isValid:
            KEYS = self.sett.force_evaluator['Keys4ImportValue4RinputOpt']
            InKeys = []
            InVals = []
            n = len(KEYS)
            for i in range(n):
                thiskeys = KEYS[i]
                m = len(thiskeys)
                if m == 2:
                    if thiskeys[1] in export_Keys:
                        InKeys.append(thiskeys[0])
                        InVals.append(thisExports[thiskeys[1]])

            n = len(InKeys)
            for i in range(len(rinputs)):
                line = rinputs[i]
                for ik in range(n):
                    key = InKeys[ik]
                    istart = line.find(key)
                    if istart != -1:
                        thislines = line.split(key)
                        afters = list(filter(None, list(map(lambda strings: strings.strip(), thislines[1].split("=")))))
                        na = len(afters)
                        if istart == 0:
                            newline = key + " = " + str(InVals[ik])
                        else:
                            newline = thislines[0] + "=" + key + " = " + str(InVals[ik])
                        if na > 1:
                            for j in range(1, na):
                                newline += "=" + afters[j]
                        rinputs[i] = newline
            return rinputs
        else:
            return rinputs

    def close(self):
        pass
