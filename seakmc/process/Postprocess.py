import os
import time
import shutil


def postprocess(tic, thissett, object_dict, simulation_time):
    LogWriter = object_dict['LogWriter']
    folds = os.listdir()
    for fold in folds:
        if "Runner_" in fold: shutil.rmtree(fold)
    for f in thissett.system["TempFiles"]:
        if os.path.isfile(f): os.remove(f)

    toc = time.time()
    logstr = f"Total KMC time steps for this simulation:{round(simulation_time, thissett.system['float_precision'])} ps"
    logstr += "\n" + "Real time cost for this simulation:" + str(
        round(toc - tic, thissett.system['float_precision'])) + " s"
    logstr += "\n" + "==================================================================="
    LogWriter.write_data(logstr)
