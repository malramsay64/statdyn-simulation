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

import hoomd
import numpy as np
import pytest

from sdrun.initialise import (
    init_from_crystal,
    init_from_file,
    init_from_none,
    initialise_snapshot,
)
from sdrun.simulation import (
    create_interface,
    equilibrate,
    make_orthorhombic,
    production,
)
from sdrun.util import dump_frame, get_num_mols


@pytest.mark.simulation
def test_run_npt(mol_params):
    """Test an npt run."""
    snapshot = init_from_none(mol_params)
    context = hoomd.context.initialize(mol_params.hoomd_args)
    production(snapshot, context, mol_params, dynamics=False)


def test_run_zero_equil_steps(snapshot_params):
    snapshot = snapshot_params["snapshot"]
    sim_params = snapshot_params["sim_params"]
    equilibrate(snapshot, sim_params)


def test_run_zero_prod_steps(snapshot_params):
    snapshot = snapshot_params["snapshot"]
    sim_params = snapshot_params["sim_params"]
    context = hoomd.context.initialize(sim_params.hoomd_args)
    production(snapshot, context, sim_params)


@pytest.mark.simulation
@pytest.mark.parametrize("cell_dimensions", range(1, 5))
def test_orthorhombic_sims(cell_dimensions, crystal_params):
    """Test the initialisation from a crystal unit cell.

    This also ensures there is no unusual things going on with the calculation
    of the orthorhombic unit cell.
    """
    # Multiple of 6 works nicely with the p2 crystal
    cell_dimensions = cell_dimensions * 6
    with crystal_params.temp_context(cell_dimensions=cell_dimensions):
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


def test_simulation_from_file(snapshot_params):
    sim_params = snapshot_params["sim_params"]
    snapshot = snapshot_params["snapshot"]
    filename = sim_params.output / "testfile.gsd"
    num_mols = get_num_mols(snapshot)

    context = hoomd.context.initialize(args=sim_params.hoomd_args)
    initialise_snapshot(snapshot, context, sim_params)
    with context:
        if sim_params.molecule.rigid:
            group = hoomd.group.rigid_center()
        else:
            group = hoomd.group.all()
        dump_frame(group, filename)

    snap_file = init_from_file(filename, sim_params.molecule, sim_params.hoomd_args)
    snap_equil = equilibrate(snap_file, sim_params)

    assert snap_equil.particles.types == sim_params.molecule.get_types()
    assert snap_equil.particles.N == snapshot.particles.N
    assert np.all(snap_equil.particles.mass[:num_mols] == sim_params.molecule.mass)
