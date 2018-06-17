#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Test the equilibrate module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from sdrun.crystals import CRYSTAL_FUNCS
from sdrun.equilibrate import (
    create_interface,
    equil_crystal,
    equil_interface,
    equil_liquid,
    equilibrate,
)
from sdrun.initialise import init_from_crystal, make_orthorhombic
from sdrun.params import SimulationParams


@pytest.fixture(params=CRYSTAL_FUNCS.values(), ids=CRYSTAL_FUNCS.keys())
def sim_params(request):
    with TemporaryDirectory() as tmp_dir:
        yield SimulationParams(
            temperature=0.4,
            num_steps=1000,
            crystal=request.param(),
            output=Path(tmp_dir),
            cell_dimensions=(10, 12, 10),
            outfile=Path(tmp_dir) / "out.gsd",
        )


def test_equil_crystal(sim_params):
    """Ensure the equilibration is close to initialisation."""
    snap_min = init_from_crystal(sim_params)
    snap_equil = equil_crystal(snap_min, sim_params)

    # Simulation box within 10% of initialisation
    for attribute in ["Lx", "Ly", "Lz", "xy", "xz", "yz"]:
        assert np.isclose(
            getattr(snap_min.box, attribute),
            getattr(snap_equil.box, attribute),
            rtol=0.1,
        )


def test_orthorhombic_equil(sim_params):
    """Ensure the equilibration is close to initialisation."""
    snap_min = init_from_crystal(sim_params)
    snap_ortho = make_orthorhombic(snap_min)
    snap_equil = equil_crystal(snap_ortho, sim_params)

    # Simulation box within 10% of initialisation
    for attribute in ["Lx", "Ly", "Lz", "xy", "xz", "yz"]:
        assert np.isclose(
            getattr(snap_ortho.box, attribute),
            getattr(snap_equil.box, attribute),
            rtol=0.1,
        )


def test_create_interface(sim_params):
    with sim_params.temp_context(init_temp=0.4, temperature=3.0):
        snapshot = create_interface(sim_params)

    assert snapshot.box.xy == 0
    assert snapshot.box.xz == 0
    assert snapshot.box.yz == 0


def test_equil_interface(sim_params):
    snap_min = init_from_crystal(sim_params)
    snap_equil = equil_crystal(snap_min, sim_params)
    snap_int = equil_interface(snap_equil, sim_params)
    assert snap_int.box.xy == 0
    assert snap_int.box.xz == 0
    assert snap_int.box.yz == 0


def test_equil_liquid(sim_params):
    equil_liquid(init_from_crystal(sim_params), sim_params)
    assert True


@pytest.mark.parametrize("equil_type", ["liquid", "crystal", "interface", "harmonic"])
def test_equilibrate(sim_params, equil_type):
    """Ensure the equilibration is close to initialisation."""
    # Initialisation of snapshot
    snap_min = init_from_crystal(sim_params)

    # Equilibration
    with sim_params.temp_context(harmonic_force=1):
        snapshot = equilibrate(snap_min, sim_params, equil_type)

    # Simulation box within 10% of initialisation
    for attribute in ["Lx", "Ly", "Lz", "xy", "xz", "yz"]:
        assert np.isclose(
            getattr(snap_min.box, attribute), getattr(snapshot.box, attribute), rtol=0.1
        )
    if equil_type in ["liquid", "interface"]:
        assert snapshot.box.xy == 0
        assert snapshot.box.xz == 0
        assert snapshot.box.yz == 0
