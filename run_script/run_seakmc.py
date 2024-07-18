import time

from seakmc.input.Input import Settings
import seakmc.process.Preprocess as preseakmc
import seakmc.process.Process as runseakmc
import seakmc.process.Postprocess as postseakmc

def main():

    tic = time.time()
    inputf = "input.yaml"
    thissett = Settings.from_file(inputf)
    thissett.validate_input()

    seakmcdata, object_dict, Eground, thisRestart = preseakmc.preprocess(thissett)
    simulation_time = runseakmc.run_seakmc(thissett, seakmcdata, object_dict, Eground, thisRestart)
    postseakmc.postprocess(tic, thissett, object_dict, simulation_time)

if __name__ == '__main__':
    main()
