#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Command line tool to run simulations."""

from setuptools import find_packages, setup

# Get the version from src/sdrun/version.py without importing the package
def get_version():
    g = {}
    exec(open("src/sdrun/version.py").read(), g)
    return g["__version__"]


setup(
    name="sdrun",
    version=get_version(),
    python_requires=">=3.6",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    entry_points="""
        [console_scripts]
        sdrun=sdrun.main:sdrun
    """,
    url="https://github.com/malramsay64/statdyn-simulation",
    author="Malcolm Ramsay",
    author_email="malramsay64@gmail.com",
    description="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
