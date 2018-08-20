# statdyn-simulation

[![Build Status](https://travis-ci.org/malramsay64/statdyn-simulation.svg?branch=master)](https://travis-ci.org/malramsay64/statdyn-simulation)
[![codecov](https://codecov.io/gh/malramsay64/statdyn-simulation/branch/master/graph/badge.svg)](https://codecov.io/gh/malramsay64/statdyn-simulation)
[![Anaconda-Server Badge](https://anaconda.org/malramsay/sdrun/badges/installer/conda.svg)](https://conda.anaconda.org/malramsay)
[![Anaconda-Server Badge](https://anaconda.org/malramsay/sdrun/badges/version.svg)](https://anaconda.org/malramsay/sdrun)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

This is a set of scripts that use
[Hoomd](https://bitbucket.org/glotzer/hoomd-blue) to perform the Molecular
dynamics simulations of a glass forming molecular liquid. There is a particular
focus on understanding the dynamic properties of these molecules.

Note that this is still very early alpha software and there are likely to be
large breaking changes that occur.

## Installation

The simplest method of installation is using `conda`. To install

    conda install malramsay::sdrun

It is also possible to set the repository up as a development environment,
in which case cloning the repository and installing is possible by running

    git clone https://github.com/malramsay64/statdyn-simulation
    cd statdyn-simulation
    conda env create
    source activate sdrun-dev
    pip install -e .

Once the environment is setup the tests can be run with

    pytest

## Running Simulations

Interaction with the program is currently through the command line, using the
command line arguments to specify the various parameters. There are two types
of parameters that can be specified.

The simulation options, which map to the internal SimulationParams class, which
are the general properties of the simulation. These idea is that these options
will be shared across all simulation types. The simulation options are
specified after the `sdrun` portion of the command.

```bash
$ sdrun --temperature 0.4 --pressure 13.5 --num-steps 1000 --molecule trimer ...
```

The other option are the simulation specific options and arguments. These are
specific to each simulation type. The simulation specific options are specified
after the simulation type.

```bash
$ sdrun --temperature 0.4 --pressure 13.5 --num-steps 1000 --molecule trimer \
  create --interface interface.gsd
```

There is documentation on each of the options and arguments that can be
specified in the help of the command.

To create a crystal structure for a simulation, the command

```bash
$ sdrun --space-group p2 --num-steps 1000 --temperature 0.4 create test.gsd
```

will generate a file which has a trimer molecule with a p2 crystal structure.
The simulation will be run for 1000 steps at a temperature of 0.4.

The output file we created be equilibrated at a higher temperature using the command

```bash
$ sdrun --space-group p2 --num-steps 1000 --temperature 1.2 --init-temp 0.4 \
  equil --equil-type crystal test.gsd test-equil.gsd
```

which will gradually increase the temperature from 0.4 to 1.2 over 1000 steps
with the final configuration output to `test-1.2.gsd`. This equilibration will
use the equilibration type of crystal, allowing the simulation cell to tilt and
adjusting the length of each side of the box independently. This is unlikely to
equilibrate this configuration; however, it runs in a reasonable time.

A production simulation can be run with the `prod` sub-command

```bash
$ sdrun --space-group p2 --num-steps 1000 --temperature 1.2 --init-temp 0.4 \
  prod test-equil.gsd
```

This has a different series of options including outputting a series of
timesteps optimised for the analysis of dynamics quantities in a file
prefixed with `trajectory-`.

Another tool I have written, [sdanalysis][sdanalyis] can be used to easily
analyse the resulting trajectories.

For running simulations with many different parameters, [experi][experi]
provides an easy to read yaml interface for running a series of command line
scripts with a complex set of variables.

[sdanalysis]: https://github.com/malramsay64/statdyn-analysis
[experi]: https://github.com/malramsay64/experi
