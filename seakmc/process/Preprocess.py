import os

from seakmc.restart.Restart import RESTART
from seakmc.core.data import SeakmcData
import seakmc.general.General as mygen


def load_RESTART(Restartsett):
    thisRestart = None
    if Restartsett["LoadRestart"]:
        filename = Restartsett["LoadFile"]
        if isinstance(filename, str):
            thisRestart = RESTART.from_file(filename)
        else:
            FileHeader = "RESTART_istep_"
            fapp = ".restart"
            files = []
            for file in os.listdir(os.getcwd()):
                if FileHeader in file and fapp in file:
                    thisstr = file.replace(FileHeader, "")
                    thisstr = thisstr.replace(fapp, "")
                    thisstrs = thisstr.split("_")
                    try:
                        istep_this = int(thisstrs[0])
                        finished_AVs = int(thisstrs[1])
                        files.append((istep_this, finished_AVs))
                    except:
                        pass
            if len(files) > 0:
                fsorted = sorted(files, key=lambda t: (t[0], t[1]), reverse=True)
                filename = FileHeader + str(fsorted[0][0]) + "_" + str(fsorted[0][1]) + fapp
                thisRestart = RESTART.from_file(filename)
            else:
                thisRestart = None
    return thisRestart


def initial_data_dynamics(thissett, seakmcdata, force_evaluator, LogWriter):
    thiscolor = 0
    if thissett.data["MoleDyn"]:
        logstr = "Molecular dynamics simulation of the initial structure ..."
        LogWriter.write_data(logstr)
        [Eground, relaxed_coords, isValid, errormsg] = force_evaluator.run_runner("DATAMD", seakmcdata, thiscolor,
                                                                                  nactive=seakmcdata.natoms,
                                                                                  thisExports=None)
        if not isValid:
            LogWriter.write_data(errormsg)
            quit()
        seakmcdata = SeakmcData.from_file("Runner_0/tmp1.dat", atom_style=thissett.data['atom_style_after'])
        seakmcdata.assert_settings(thissett)
        seakmcdata.to_atom_style()
        [Eground, relaxed_coords, isValid, errormsg] = force_evaluator.run_runner("DATAOPT", seakmcdata, thiscolor,
                                                                                  nactive=seakmcdata.natoms,
                                                                                  thisExports=None)
        if not isValid:
            LogWriter.write_data(errormsg)
            quit()
        seakmcdata = SeakmcData.from_file("Runner_0/tmp1.dat", atom_style=thissett.data['atom_style_after'])
        seakmcdata.assert_settings(thissett)
        seakmcdata.to_atom_style()
        seakmcdata.velocities = None
    elif not thissett.data["Relaxed"]:
        logstr = "Relaxing the initial structure ..."
        LogWriter.write_data(logstr)
        [Eground, relaxed_coords, isValid, errormsg] = force_evaluator.run_runner("DATAOPT", seakmcdata, thiscolor,
                                                                                  nactive=seakmcdata.natoms,
                                                                                  thisExports=None)
        if not isValid:
            LogWriter.write_data(errormsg)
            quit()
        seakmcdata = SeakmcData.from_file("Runner_0/tmp1.dat", atom_style=thissett.data['atom_style_after'])
        seakmcdata.assert_settings(thissett)
        seakmcdata.to_atom_style()
        seakmcdata.velocities = None
    else:
        [Eground, relaxed_coords, isValid, errormsg] = force_evaluator.run_runner("DATAMD0", seakmcdata, thiscolor,
                                                                                  nactive=seakmcdata.natoms,
                                                                                  thisExports=None)
        if not isValid:
            LogWriter.write_data(errormsg)
            quit()
    return seakmcdata, Eground


def preprocess(thissett):
    Eground = 0.0
    os.makedirs("Runner_0", exist_ok=True)
    thisRestart = load_RESTART(thissett.system["Restart"])
    object_dict = mygen.object_maker(thissett, thisRestart)
    out_paths = object_dict['out_paths']
    force_evaluator = object_dict['force_evaluator']
    LogWriter = object_dict['LogWriter']

    thiscolor = 0
    if thisRestart is None:
        seakmcdata = SeakmcData.from_file(thissett.data['FileName'], atom_style=thissett.data['atom_style'])
        seakmcdata.assert_settings(thissett)
        seakmcdata.to_atom_style()
        seakmcdata.velocities = None
        logstr = "Successfully loading input and structure ..."
        LogWriter.write_data(logstr)
        seakmcdata, Eground = initial_data_dynamics(thissett, seakmcdata, force_evaluator, LogWriter)
        if thissett.visual["Write_Data_SPs"]["Write_KMC_Data"]:  seakmcdata.to_lammps_data(
            out_paths[1] + "/" + "KMC_" + str(0) + ".dat", to_atom_style=True)

    else:
        seakmcdata = thisRestart.seakmcdata
        seakmcdata.assert_settings(thissett)
        istep_this = thisRestart.istep_this
        Eground = thisRestart.Eground
        if Eground is None or Eground == 0.0:
            [Eground, relaxed_coords, isValid, errormsg] = force_evaluator.run_runner("DATAMD0", seakmcdata, thiscolor,
                                                                                      nactive=seakmcdata.natoms,
                                                                                      thisExports=None)
            if not isValid:
                LogWriter.write_data(errormsg)
                quit()
        seakmcdata.to_lammps_data(out_paths[1] + "/" + "KMC_" + str(istep_this) + ".dat", to_atom_style=True)

    return seakmcdata, object_dict, Eground, thisRestart
