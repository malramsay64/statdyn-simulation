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


setup_requires = []
install_requires = ["numpy", "rowan", "attrs>=19.2", "click"]
test_requires = [
    "pytest==5.3.2",
    "pylint==2.4.4",
    "hypothesis>=4.43.1,<6.0",
    "coverage==5.0.3",
    "black==19.10b0",
    "mypy==0.761",
    "pytest-mypy==0.4.2",
    "pytest-pylint==0.15.0",
    "pytest-cov==2.8.1",
]
docs_requires = ["sphinx", "sphinx_rtd_theme", "sphinx_autodoc_typehints"]
dev_requires = docs_requires + test_requires


setup(
    name="sdrun",
    version=get_version(),
    python_requires=">=3.6",
    setup_requires=setup_requires,
    install_requires=install_requires,
    extras_require={
        "doc": docs_requires,
        "test": test_requires,
        "dev": dev_requires,
        "mpi": ["mpi4py"],
    },
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
