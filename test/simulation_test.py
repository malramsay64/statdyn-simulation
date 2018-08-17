#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Test the simulation module."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import hoomd
import pytest

from sdrun.crystals import CRYSTAL_FUNCS
from sdrun.initialise import init_from_crystal, init_from_none
from sdrun.molecules import MOLECULE_DICT
from sdrun.params import SimulationParams
from sdrun.simulation import equilibrate, make_orthorhombic, production


@pytest.mark.simulation
def test_run_npt(mol_params):
    """Test an npt run."""
    snapshot = init_from_none(mol_params)
    context = hoomd.context.initialize(mol_params.hoomd_args)
    production(snapshot, context, mol_params, dynamics=False)


@pytest.mark.simulation
@pytest.mark.parametrize("cell_dimensions", range(1, 5))
def test_orthorhombic_sims(cell_dimensions, crystal_params):
    """Test the initialisation from a crystal unit cell.

    This also ensures there is no unusual things going on with the calculation
    of the orthorhombic unit cell.
    """
    # Multiple of 6 works nicely with the p2 crystal
    cell_dimensions = cell_dimensions * 6
    with mol_params.temp_context(cell_dimensions=cell_dimensions):
        snapshot = init_from_crystal(crystal_params)
        snapshot = equilibrate(snapshot, crystal_params, equil_type="crystal")
    snapshot = make_orthorhombic(snapshot)
    temp_context = hoomd.context.initialize(crystal_params.hoomd_args)
    production(snapshot, temp_context, crystal_params, dynamics=False)


@pytest.mark.simulation
def test_file_placement(mol_params):
    """Ensure files are located in the correct directory when created."""
    snapshot = init_from_none(mol_params)
    context = hoomd.context.initialize(mol_params.hoomd_args)
    production(snapshot, context, mol_params, dynamics=True)

    params = {
        "molecule": mol_params.molecule,
        "pressure": mol_params.pressure,
        "temperature": mol_params.temperature,
    }
    outdir = Path(mol_params.output)
    print(list(outdir.glob("*")))
    assert (
        outdir / "{molecule}-P{pressure:.2f}-T{temperature:.2f}.gsd".format(**params)
    ).is_file()
    assert (
        outdir
        / "dump-{molecule}-P{pressure:.2f}-T{temperature:.2f}.gsd".format(**params)
    ).is_file()
    assert (
        outdir
        / "thermo-{molecule}-P{pressure:.2f}-T{temperature:.2f}.log".format(**params)
    ).is_file()
    assert (
        outdir
        / "trajectory-{molecule}-P{pressure:.2f}-T{temperature:.2f}.gsd".format(
            **params
        )
    ).is_file()


@pytest.mark.simulation
@pytest.mark.parametrize("pressure, temperature", [(1.0, 1.8), (13.5, 3.00)])
def test_interface(mol_params, pressure, temperature):
    init_temp = 0.4
    outdir = str(mol_params.output)
    create_command = [
        "sdrun",
        "--pressure",
        "{}".format(pressure),
        "--space-group",
        "p2",
        "--lattice-lengths",
        "48",
        "42",
        "--temperature",
        "{}".format(init_temp),
        "--num-steps",
        "1000",
        "--output",
        outdir,
        "-vvv",
        "--hoomd-args",
        '"--mode=cpu"',
        "create",
        str(Path(outdir) / "P{:.2f}-T{:.2f}.gsd".format(pressure, init_temp)),
    ]
    melt_command = [
        "sdrun",
        "--pressure",
        "{}".format(pressure),
        "--space-group",
        "p2",
        "--temperature",
        "{}".format(temperature),
        "--output",
        outdir,
        "--num-steps",
        "1000",
        "-vvv",
        "--hoomd-args",
        '"--mode=cpu"',
        "equil",
        "--equil-type",
        "interface",
        str(Path(outdir) / "P{:.2f}-T{:.2f}.gsd".format(pressure, init_temp)),
        str(Path(outdir) / "P{:.2f}-T{:.2f}.gsd".format(pressure, temperature)),
    ]
    # Run creation simulation
    create = subprocess.run(create_command)
    assert create.returncode == 0

    # Ensure input file for melting simulation is present
    assert Path(outdir).exists()
    assert Path(create_command[-1]).exists()

    # Run melting simulation
    melt = subprocess.run(melt_command)
    assert melt.returncode == 0
