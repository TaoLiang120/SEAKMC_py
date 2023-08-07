import sys, os, warnings
import time
import numpy as np
import shutil

from seakmc.core.data import SeakmcData

infile = "in.data"
outfile = "out.data"
thisdata = SeakmcData.from_file(infile, atom_style="molecular")
by = "dxyz" ## by
range = [10, 20] ## [value_min, value_max]
to_val = -2 ## assign to_val to molecule-ID of selected atoms
Selection = False 

'''
  if Selection is True, the selected atoms will be by >= value_min and by < value_max
  if Selection is False, the selected atoms will be by < value_min or by >= value_max 
'''

'''
   available by are:
   DXYZ, DXY, DXZ, DYZ, X, Y, Z, XSN, YSN, ZSN, TYPE
   XSN, YSN, ZSN are the fractional coordinates ranging from [0, 1.0)
   all are case insensitive
'''

thisdata.modify_molecule_id(by, range=range, to_val=to_val, Selection=Selection)
## you can apply modify_molecule_id multiply times with different by and to_val
thisdata.to_lammps_data(outfile, to_atom_style=True)

