import numpy as np
a = {"force": np.array([1,2,3]), "energy": 1.5}

force = a["force"]
force[1] = -2
print(a)

energy = a["energy"]
energy = 20
print(a)