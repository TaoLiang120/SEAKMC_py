#!/usr/bin/env python

import os
from setuptools import setup, find_packages
import versioneer
#module_dir = os.path.dirname(os.path.abspath(__file__))
##with open('README.rst') as f:
##    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name = 'seakmc',
    description = 'Self Evolution Adaptive Kinetic Monte Carlo',
    long_description="SEAKMC samples the potential energy landscape of a system at its local minimum.",
    url = 'https://github.com/TaoLiang120/SEAKMC_py',
    version = versioneer.get_version(),
    license = license,
    author = 'Tao Liang',
    author_email = 'xhtliang120@gmail.com',
    classifiers=[
         'Development Status :: 2 - pylammps, lammps/Stable, VASP/Test',
         'Topic :: Scientific/Engineering :: Physics',
         'License :: MIT License',
         'Intended Audience :: Science/Research',
         'Operating System :: Linux, macOS, Windows(not tested)',
         'Programming Language :: Python :: 3.0',
    ],
    keywords='seakmc',
    packages = find_packages(exclude=('tests','docs')),
    include_package_data = True,
    package_data={
        "seakmc.input": ["*.yaml"],
    },
    entry_points={
        'console_scripts': ['seakmc = seakmc.script.seakmc:main']
        },
    install_requires = ['anaconda>=3.0', 'pymatgen>=2020',
                        'lammps>=2020', 'mpi4py>=3.0'],
##  extras_require = {'doc': ['codecov>=2.0', 'sphinx>=1.3.1']},
    cmdclass=versioneer.get_cmdclass(),
)
