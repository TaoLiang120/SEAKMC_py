import os

def data_dynamics(purpose, force_evaluator, data, ntask_tot, nactive=None, nproc_task=1, thisExports=None):
    if nactive is None:
        try:
            nactive = data.nactive
        except:
            nactive = data.natoms

    ntask_tot = 1
    thiscolor = 1
    [Eground, relaxed_coords, isValid, errormsg] = force_evaluator.run_runner(purpose, data, thiscolor,
                                                                                      nactive=nactive,
                                                                                      thisExports=thisExports,
                                                                                      comm=None)

    return [Eground, relaxed_coords, isValid, errormsg]
