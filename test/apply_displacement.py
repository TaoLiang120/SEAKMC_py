import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func1(x, a, b):
    return a * x + b

def fit(x, y):
        if len(x) == 2:
            popt = np.linalg.solve(np.array([[x[0], 1.0], [x[1], 1.0]]), y)
            #popt, pcov = curve_fit(func1, x, y)
            isValid = True
        else:
            try:
                popt, pcov = curve_fit(func1, x, y)
                isValid = True
            except:
                popt = np.array([0.0, 0.0])
                isValid = False
        return isValid, popt


def chop_x_y(x, y, popt):
        residuals = y - func1(x, *popt)
        absr = np.absolute(residuals)
        m = np.mean(absr)
        s = np.std(absr)
        if s > 0.0:
            r = np.absolute((absr - m) / s)
            print(f"absr:{absr} m:{m} s:{s} r:{r}")
            rthres = 1.3
            inds = np.arange(len(x))
            inds = np.compress(r < rthres, inds)
            x = x[inds]
            y = y[inds]
        return x, y


mindisp = 0.00001
maxdisp = 0.01
KB = 8.617333262145e-5
temp = 600.0
KBT = KB * temp
meanpref = 10


ref_length = 100.0
target_strainrate = 1.0e3

strains = np.array([1.0e-6, 2.0e-6])
barrs = np.array([0.5, 0.48])
freqs = meanpref * np.exp(-barrs / KBT)
timesteps = np.divide(1.0, freqs)
strainrates = np.divide(strains, timesteps) * 1.0e12
logstrainrates = np.log(np.absolute(strainrates))
print(meanpref * np.exp(-0.5 / KBT))
print(f"freqs: {freqs} timesteps: {timesteps}")
print(f"strainrates: {strainrates} logstrainrates: {logstrainrates}")

isValid, popt = fit(strains, logstrainrates)
print(f"isValid: {isValid} popt: {popt}")

if isValid:
    if len(strains) > 2:
        x, y = chop_x_y(strains, logstrainrates, popt)
        if len(x) >= 2:
            isValid, popt = fit(x, y)
    target_strain = (np.log(target_strainrate) - popt[1]) / popt[0]
    target_displacement = target_strain * ref_length
else:
    target_strain = 0.0
    target_displacement = mindisp

straintype = 1
print(f"target_strain: {target_strain} target_displacement: {target_displacement}")

if straintype == 0:
    target_displacement = -abs(target_displacement)
else:
    target_displacement = abs(target_displacement)

if abs(target_displacement) < mindisp:
    if straintype == 0:
        target_displacement = -mindisp
    else:
        target_displacement = mindisp
if abs(target_displacement) > maxdisp:
    if straintype == 0:
        target_displacement = -maxdisp
if target_displacement < mindisp:
        target_displacement = mindisp
if target_displacement > maxdisp:
        target_displacement = maxdisp
print(f"target_displacement: {target_displacement}")
