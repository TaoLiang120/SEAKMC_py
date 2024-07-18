#!/usr/bin/env python

import os
from setuptools import setup, find_packages


setup(
    name='seakmc',
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
    version='2.0.0',
    package_data={
        "seakmc.input": ["*.yaml"],
    },
    entry_points={
        'console_scripts': ['seakmc = seakmc.script.seakmc:main']
    },
    description='Self Evolution Adaptive Kinetic Monte Carlo',
    author='Tao Liang',
    author_email='xhtliang120@gmail.com',
    url='https://github.com/TaoLiang120/SEAKMC_py',
    download_url='https://github.com/TaoLiang120/SEAKMC_py/archive/refs/heads/main.zip',
)
