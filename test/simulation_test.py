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
from sdrun.equilibrate import equil_crystal, make_orthorhombic
from sdrun.initialise import init_from_crystal, init_from_none
from sdrun.molecules import MOLECULE_DICT
from sdrun.params import SimulationParams, paramsContext
from sdrun.simrun import run_npt

HOOMD_ARGS = "--mode=cpu"


@pytest.fixture(params=MOLECULE_DICT.values(), ids=MOLECULE_DICT.keys())
def sim_params(request):
    with TemporaryDirectory() as tmp_dir:
        yield SimulationParams(
            temperature=0.4,
            num_steps=100,
            molecule=request.param(),
            output=tmp_dir,
            outfile=Path(tmp_dir) / "testout",
            dynamics=False,
            hoomd_args=HOOMD_ARGS,
            minimize=True,
        )


@pytest.fixture(params=CRYSTAL_FUNCS.values(), ids=CRYSTAL_FUNCS.keys())
def sim_params_crystal(request):
    with TemporaryDirectory() as tmp_dir:
        yield SimulationParams(
            temperature=0.4,
            num_steps=100,
            crystal=request.param(),
            output=tmp_dir,
            outfile=Path(tmp_dir) / "testout",
            dynamics=False,
            hoomd_args=HOOMD_ARGS,
            minimize=True,
        )


@pytest.mark.simulation
def test_run_npt(sim_params):
    """Test an npt run."""
    run_npt(
        snapshot=init_from_none(sim_params),
        context=hoomd.context.initialize(HOOMD_ARGS),
        sim_params=sim_params,
    )


@pytest.mark.simulation
@pytest.mark.parametrize("cell_dimensions", range(1, 5))
def test_orthorhombic_sims(cell_dimensions, sim_params_crystal):
    """Test the initialisation from a crystal unit cell.

    This also ensures there is no unusual things going on with the calculation
    of the orthorhombic unit cell.
    """
    sim_params = sim_params_crystal
    cell_dimensions = cell_dimensions * 6, cell_dimensions * 6
    with paramsContext(sim_params, cell_dimensions=cell_dimensions):
        snap = init_from_crystal(sim_params)
    snap = make_orthorhombic(equil_crystal(snap, sim_params=sim_params))
    run_npt(snap, context=hoomd.context.initialize(HOOMD_ARGS), sim_params=sim_params)


@pytest.mark.simulation
def test_file_placement(sim_params):
    """Ensure files are located in the correct directory when created."""
    with paramsContext(sim_params, dynamics=True):
        run_npt(
            init_from_none(sim_params),
            hoomd.context.initialize(HOOMD_ARGS),
            sim_params=sim_params,
        )
    params = {
        "molecule": sim_params.molecule,
        "pressure": sim_params.pressure,
        "temperature": sim_params.temperature,
    }
    outdir = Path(sim_params.output)
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
def test_interface(sim_params, pressure, temperature):
    init_temp = 0.4
    outdir = str(sim_params.output)
    create_command = [
        "sdrun",
        "create",
        "--pressure",
        "{}".format(pressure),
        "--space-group",
        "p2",
        "--lattice-lengths",
        "48",
        "42",
        "--temperature",
        "{}".format(init_temp),
        "--steps",
        "1000",
        "--output",
        outdir,
        "-vvv",
        "--hoomd-args",
        '"--mode=cpu"',
        str(
            Path(outdir)
            / "create_interface-P{:.2f}-T{:.2f}.gsd".format(pressure, init_temp)
        ),
    ]
    melt_command = [
        "sdrun",
        "equil",
        "--equil-type",
        "interface",
        "--pressure",
        "{}".format(pressure),
        "--space-group",
        "p2",
        "--temperature",
        "{}".format(temperature),
        "--output",
        outdir,
        "--steps",
        "1000",
        "-vvv",
        "--hoomd-args",
        '"--mode=cpu"',
        str(
            Path(outdir)
            / "create_interface-P{:.2f}-T{:.2f}.gsd".format(pressure, init_temp)
        ),
        str(
            Path(outdir)
            / "melt_interface-P{:.2f}-T{:.2f}.gsd".format(pressure, temperature)
        ),
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
