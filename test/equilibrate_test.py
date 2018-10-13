#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Test the equilibrate module."""

import hoomd
import numpy as np
import pytest

from sdrun.initialise import init_from_crystal, init_from_file, make_orthorhombic
from sdrun.simulation import create_interface, equilibrate, get_group
from sdrun.util import get_num_mols


def test_orthorhombic_equil(crystal_params):
    """Ensure the equilibration is close to initialisation."""
    snap_min = init_from_crystal(crystal_params)
    snap_ortho = make_orthorhombic(snap_min)
    snap_equil = equilibrate(snap_ortho, crystal_params, equil_type="crystal")

    # Simulation box within 20% of initialisation
    for attribute in ["Lx", "Ly", "Lz", "xy", "xz", "yz"]:
        assert np.isclose(
            getattr(snap_ortho.box, attribute),
            getattr(snap_equil.box, attribute),
            rtol=0.2,
        )


def test_create_interface(crystal_params):
    with crystal_params.temp_context(init_temp=0.4, temperature=3.0):
        snapshot = create_interface(crystal_params)

    assert snapshot.box.xy == 0
    assert snapshot.box.xz == 0
    assert snapshot.box.yz == 0


@pytest.mark.parametrize("equil_type", ["liquid", "crystal", "interface"])
def test_equilibrate(crystal_params, equil_type):
    """Ensure the equilibration is close to initialisation."""
    # Initialisation of snapshot
    snap_min = init_from_crystal(crystal_params)

    # Equilibration
    snapshot = equilibrate(snap_min, crystal_params, equil_type)

    # Simulation box within 10% of initialisation
    for attribute in ["Lx", "Ly", "Lz", "xy", "xz", "yz"]:
        assert np.isclose(
            getattr(snap_min.box, attribute),
            getattr(snapshot.box, attribute),
            rtol=0.20,
        )
    if equil_type in ["liquid", "interface"]:
        assert snapshot.box.xy == 0
        assert snapshot.box.xz == 0
        assert snapshot.box.yz == 0


def test_get_group(mol_params):
    with hoomd.context.initialize(mol_params.hoomd_args):
        sys = hoomd.init.create_lattice(hoomd.lattice.sq(1), 5)
        group = get_group(sys, mol_params)
        assert group is not None


@pytest.mark.parametrize("equil_type", ["liquid", "crystal", "interface"])
def test_dump_mols(crystal_params, equil_type):
    """Ensure the equilibration is close to initialisation."""
    # Initialisation of snapshot
    snap_init = init_from_crystal(crystal_params)
    snap_init_mols = get_num_mols(snap_init)

    # Equilibration
    snap_equil = equilibrate(snap_init, crystal_params, equil_type)
    snap_equil_mols = get_num_mols(snap_equil)

    snap_out = init_from_file(
        crystal_params.outfile, crystal_params.molecule, crystal_params.hoomd_args
    )
    snap_out_mols = get_num_mols(snap_out)

    assert snap_init_mols == snap_equil_mols
    assert snap_init_mols == snap_out_mols
