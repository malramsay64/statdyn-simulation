MD-Molecules-Hoomd
==================

[![Build Status](https://travis-ci.org/malramsay64/statdyn-simulation.svg?branch=master)](https://travis-ci.org/malramsay64/statdyn-simulation)
[![codecov](https://codecov.io/gh/malramsay64/statdyn-simulation/branch/master/graph/badge.svg)](https://codecov.io/gh/malramsay64/statdyn-simulation)
[![Anaconda-Server Badge](https://anaconda.org/malramsay/sdrun/badges/installer/conda.svg)](https://conda.anaconda.org/malramsay)
[![Anaconda-Server Badge](https://anaconda.org/malramsay/sdrun/badges/version.svg)](https://anaconda.org/malramsay/sdrun)


This is a set of scripts that use
[Hoomd](https://bitbucket.org/glotzer/hoomd-blue) to perform the Molecular
dynamics simulations of a glass forming molecular liquid. There is a particular
focus on understanding the dynamic properties of these molecules.

Note that this is still very early alpha software and there are likely to be
large breaking changes that occur.

Installation
------------

The simplest method of installation is using `conda`. To install

    conda install -c malramsay sdrun

It is also possible to set the repository up as a development environment,
in which case cloning the repository and installing is possible by running

    git clone https://github.com/malramsay64/statdyn-simulation
    cd statdyn-simulation
    conda env create
    source activate sdrun-dev
    pip install -e .

Once the environment is setup the tests can be run with

    pytest

Running Simulations
-------------------

Interaction with the program is currently through the command line, using the
command line arguments to specify the various parameters.

To create a crystal structure for a simulation run

    sdrun create --space-group p2 -s 1000 test.gsd

which will generate a file which has a trimer molecule with a p2 crystal
structure. The simulation will be run for 1000 steps at a default low
temperature to relax any stress.

For other options see

    sdrun create --help

This output file we created can then be equilibrated using

    sdrun equil -t 1.2 -s 1000 test.gsd test-1.2.gsd

which will gradually bring the temperature from the default to 1.2 over 1000
steps with the final configuration output to `test-1.2.gsd`. This is unlikely
to actually equilibrate this configuration, but it will run fast.

A production run can be run with the `prod` sub-command

    sdrun prod -t 1.2 -s 1000 test-1.2.gsd

This has a different series of options including outputting a series of
timesteps optimised for the analysis of dynamics quantities in the file
prefixed with `trajectory-`. 

For the analysis of the resulting trajectories the [sdanalysis][sdanalyis] tool
can be used.

[sdanalysis]: https://github.com/malramsay64/statdyn-analysis
