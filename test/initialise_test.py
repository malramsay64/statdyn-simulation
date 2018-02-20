#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Module for testing the initialisation."""

from pathlib import Path

import hoomd
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats, integers, tuples
from sdrun import crystals, molecules
from sdrun.simulation import initialise
from sdrun.simulation.params import SimulationParams, paramsContext

from .crystal_test import get_distance


PARAMETERS = SimulationParams(
    temperature=0.4,
    pressure=1.0,
    num_steps=100,
    crystal=crystals.TrimerPg(),
    outfile_path=Path('test/tmp'),
    cell_dimensions=(10, 10),
)

def create_snapshot(molecule):
    """Easily create a snapshot for later use in testing."""
    with paramsContext(PARAMETERS, molecule=molecule) as sim_params:
        return initialise.init_from_none(sim_params)

INIT_TEST_PARAMS = [
    (initialise.init_from_none, [PARAMETERS]),
    (initialise.init_from_crystal, [PARAMETERS]),
]


@pytest.mark.parametrize('molecule', molecules.MOLECULE_LIST)
def test_init_from_none(molecule):
    """Ensure init_from_none has the correct type and number of particles."""
    snap = create_snapshot(molecule)
    assert snap.particles.N == 100 * molecule.num_particles


@pytest.mark.parametrize('molecule', molecules.MOLECULE_LIST)
def test_initialise_snapshot(molecule):
    """Test initialisation from a snapshot works."""
    initialise.initialise_snapshot(
        create_snapshot(molecule),
        hoomd.context.initialize(''),
        molecule,
    )
    assert True


@pytest.mark.parametrize("init_func, args", INIT_TEST_PARAMS)
def test_init_all(init_func, args):
    """Test the initialisation of all init functions."""
    if args:
        init_func(*args)
    else:
        init_func()
    assert True


@pytest.mark.parametrize("init_func, args", INIT_TEST_PARAMS)
def test_2d(init_func, args):
    """Test box is 2d when initialised."""
    if args:
        sys = init_func(*args)
    else:
        sys = init_func()
        assert sys.box.dimensions == 2


def test_orthorhombic_null():
    """Ensure null operation with orthorhombic function.

    In the case where the unit cell is already orthorhombic,
    check that nothing has changed unexpectedly.
    """
    with hoomd.context.initialize():
        snap = create_snapshot(molecules.Trimer())
        assert np.all(
            initialise.make_orthorhombic(snap).particles.position ==
            snap.particles.position)
        assert snap.box.xy == 0
        assert snap.box.xz == 0
        assert snap.box.yz == 0


@given(tuples(integers(max_value=30, min_value=5),
              integers(max_value=30, min_value=5)))
@settings(max_examples=10, deadline=None)
def test_make_orthorhombic(cell_dimensions):
    """Ensure that a conversion to an orthorhombic cell goes smoothly.

    This tests a number of modes of operation
        - nothing changes in an already orthorhombic cell
        - no particles are outside the box when moved
        - the box is actually orthorhombic
    """
    with hoomd.context.initialize():
        snap_crys = hoomd.init.create_lattice(
            unitcell=crystals.TrimerP2().get_unitcell(),
            n=cell_dimensions
        ).take_snapshot()
        snap_ortho = initialise.make_orthorhombic(snap_crys)
        assert np.all(
            snap_ortho.particles.position[:, 0] < snap_ortho.box.Lx / 2.)
        assert np.all(
            snap_ortho.particles.position[:, 0] > -snap_ortho.box.Lx / 2.)
        assert np.all(
            snap_ortho.particles.position[:, 1] < snap_ortho.box.Ly / 2.)
        assert np.all(
            snap_ortho.particles.position[:, 1] > -snap_ortho.box.Ly / 2.)
        assert snap_ortho.box.xy == 0
        assert snap_ortho.box.xz == 0
        assert snap_ortho.box.yz == 0


@given(tuples(integers(max_value=30, min_value=5),
              integers(max_value=30, min_value=5)))
@settings(max_examples=10, deadline=None)
def test_orthorhombic_init(cell_dimensions):
    """Ensure orthorhombic cell initialises correctly."""
    snap = initialise.init_from_crystal(PARAMETERS)
    snap_ortho = initialise.make_orthorhombic(snap)
    assert np.all(snap_ortho.particles.position ==
                  snap.particles.position)
    assert np.all(
        snap_ortho.particles.position[:, 0] < snap_ortho.box.Lx / 2.)
    assert np.all(
        snap_ortho.particles.position[:, 0] > -snap_ortho.box.Lx / 2.)
    assert np.all(
        snap_ortho.particles.position[:, 1] < snap_ortho.box.Ly / 2.)
    assert np.all(
        snap_ortho.particles.position[:, 1] > -snap_ortho.box.Ly / 2.)
    assert snap_ortho.box.xy == 0
    assert snap_ortho.box.xz == 0
    assert snap_ortho.box.yz == 0
    for i in snap.particles.position:
        assert np.sum(get_distance(i, snap.particles.position, snap.box) < 1.1) <= 3


@given(floats(min_value=0.1, allow_infinity=False, allow_nan=False))
@settings(deadline=None)
def test_moment_inertia(scaling_factor):
    """Ensure moment of intertia is set correctly in setup."""
    init_mol = molecules.Trimer(moment_inertia_scale=scaling_factor)
    snapshot = initialise.initialise_snapshot(
        create_snapshot(molecules.Trimer()),
        hoomd.context.initialize(''),
        init_mol,
    ).take_snapshot()
    nmols = max(snapshot.particles.body) + 1
    assert np.allclose(snapshot.particles.moment_inertia[:nmols],
                       np.array(init_mol.moment_inertia).astype(np.float32))
