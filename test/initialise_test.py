#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Module for testing the initialisation."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import hoomd
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers, tuples

from sdrun.crystals import CRYSTAL_FUNCS, TrimerP2
from sdrun.initialise import (
    init_from_crystal,
    init_from_none,
    initialise_snapshot,
    make_orthorhombic,
)
from sdrun.molecules import MOLECULE_DICT
from sdrun.params import SimulationParams
from sdrun.util import get_num_mols

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture(params=MOLECULE_DICT.values(), ids=MOLECULE_DICT.keys())
def sim_params(request):
    with TemporaryDirectory() as tmp_dir:
        yield SimulationParams(
            temperature=0.4,
            pressure=1.0,
            num_steps=100,
            molecule=request.param(),
            output=Path(tmp_dir),
            cell_dimensions=10,
        )


@pytest.fixture(params=CRYSTAL_FUNCS.values(), ids=CRYSTAL_FUNCS.keys())
def sim_params_crystal(request):
    with TemporaryDirectory() as tmp_dir:
        yield SimulationParams(
            temperature=0.4,
            pressure=1.0,
            num_steps=100,
            crystal=request.param(),
            output=Path(tmp_dir),
            cell_dimensions=10,
        )


def test_init_from_none(sim_params):
    """Ensure init_from_none has the correct type and number of particles."""
    snap = init_from_none(sim_params)
    # Each unit cell should create a single particle
    num_particles = np.prod(np.array(sim_params.cell_dimensions))
    assert snap.particles.N == num_particles * sim_params.molecule.num_particles


def test_initialise_snapshot(sim_params):
    """Test initialisation from a snapshot works."""
    snap = init_from_none(sim_params)
    context = hoomd.context.initialize(sim_params.hoomd_args)
    sys = initialise_snapshot(snap, context, sim_params)
    assert isinstance(sys, hoomd.data.system_data)
    snap_init = sys.take_snapshot()
    # Ensure bodies are initialised
    assert np.any(snap.particles.body != 2 ** 32 - 1)
    # Ensure all particles in molecules are created
    num_particles = (
        np.prod(np.array(sim_params.cell_dimensions))
        * sim_params.molecule.num_particles
    )
    assert snap_init.particles.N == num_particles


def test_initialise_randomise(sim_params):
    snapshot = hoomd.data.gsd_snapshot("test/data/Trimer-13.50-3.00.gsd")
    context = hoomd.context.initialize(sim_params.hoomd_args)
    num_mols = get_num_mols(snapshot)
    with sim_params.temp_context(iteration_id=0):
        sys = initialise_snapshot(snapshot, context, sim_params)
        assert isinstance(sys, hoomd.data.system_data)
        snap = sys.take_snapshot()
    angmom_similarity = np.sum(
        snap.particles.angmom[:num_mols] != snapshot.particles.angmom[:num_mols]
    )
    velocity_similarity = np.sum(
        snap.particles.velocity[:num_mols] != snapshot.particles.velocity[:num_mols]
    )
    assert angmom_similarity < 5
    assert velocity_similarity < 5


def test_init_crystal(sim_params_crystal):
    """Test the initialisation of all crystals."""
    snap = init_from_crystal(sim_params_crystal)
    Nx, Ny, _ = sim_params_crystal.cell_dimensions
    unitcell = sim_params_crystal.crystal.get_matrix()
    logger.debug("Unitcell: %s", unitcell)
    Lx = np.linalg.norm(np.dot(np.array([1, 0, 0]), unitcell))
    Ly = np.linalg.norm(np.dot(np.array([0, 1, 0]), unitcell))

    assert Lx > 0
    assert Ly > 0
    assert Nx > 0
    assert Ny > 0

    # Simulation box within 20% of initialisation
    assert np.allclose(snap.box.Lx, Lx * Nx, rtol=0.2)
    assert np.allclose(snap.box.Ly, Ly * Ny, rtol=0.2)


def test_orthorhombic_null(sim_params):
    """Ensure null operation with orthorhombic function.

    In the case where the unit cell is already orthorhombic,
    check that nothing has changed unexpectedly.
    """
    with hoomd.context.initialize(sim_params.hoomd_args):
        snap = init_from_none(sim_params)
        assert np.all(
            make_orthorhombic(snap).particles.position == snap.particles.position
        )
        assert snap.box.xy == 0
        assert snap.box.xz == 0
        assert snap.box.yz == 0


@given(tuples(integers(max_value=30, min_value=5), integers(max_value=30, min_value=5)))
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
            unitcell=TrimerP2().get_unitcell(), n=cell_dimensions
        ).take_snapshot()
        snap_ortho = make_orthorhombic(snap_crys)
        assert np.all(snap_ortho.particles.position[:, 0] < snap_ortho.box.Lx / 2.)
        assert np.all(snap_ortho.particles.position[:, 0] > -snap_ortho.box.Lx / 2.)
        assert np.all(snap_ortho.particles.position[:, 1] < snap_ortho.box.Ly / 2.)
        assert np.all(snap_ortho.particles.position[:, 1] > -snap_ortho.box.Ly / 2.)
        assert snap_ortho.box.xy == 0
        assert snap_ortho.box.xz == 0
        assert snap_ortho.box.yz == 0


@pytest.mark.parametrize("cell_dimensions", [[5, 10, 15, 20]])
def test_orthorhombic_init(sim_params_crystal, cell_dimensions):
    """Ensure orthorhombic cell initialises correctly."""
    snap = init_from_crystal(sim_params_crystal)
    snap_ortho = make_orthorhombic(snap)
    assert np.all(snap_ortho.particles.position == snap.particles.position)
    assert np.all(snap_ortho.particles.position[:, 0] < snap_ortho.box.Lx / 2.)
    assert np.all(snap_ortho.particles.position[:, 0] > -snap_ortho.box.Lx / 2.)
    assert np.all(snap_ortho.particles.position[:, 1] < snap_ortho.box.Ly / 2.)
    assert np.all(snap_ortho.particles.position[:, 1] > -snap_ortho.box.Ly / 2.)
    assert snap_ortho.box.xy == 0
    assert snap_ortho.box.xz == 0
    assert snap_ortho.box.yz == 0


@pytest.mark.parametrize("scaling_factor", [0.1, 1, 10, 100])
def test_moment_inertia(sim_params, scaling_factor):
    """Ensure moment of intertia is set correctly in setup."""
    init_mol = np.array(sim_params.molecule.moment_inertia)
    logger.debug("Moment Intertia before scaling: %s", init_mol)
    init_mol *= scaling_factor
    logger.debug("Moment Intertia after scaling: %s", init_mol)
    with sim_params.temp_context(moment_inertia_scale=scaling_factor):
        snapshot = init_from_none(sim_params)
        context = hoomd.context.initialize(sim_params.hoomd_args)
        snapshot = initialise_snapshot(snapshot, context, sim_params).take_snapshot()
        num_mols = get_num_mols(snapshot)
        logger.debug(
            "Simulation Moment Intertia: %s", snapshot.particles.moment_inertia[0]
        )
        logger.debug("Intended Moment Intertia: %s", init_mol)
        diff = snapshot.particles.moment_inertia[:num_mols] - init_mol
        assert np.allclose(diff, 0, atol=1e-1)
