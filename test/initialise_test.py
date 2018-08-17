#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Module for testing the initialisation."""

import logging

import hoomd
import numpy as np
import pytest
import rowan
from hypothesis import given, settings
from hypothesis.strategies import integers, tuples
from numpy.testing import assert_allclose

from sdrun import SimulationParams
from sdrun.crystals import TrimerP2, TrimerPg
from sdrun.initialise import (
    init_from_crystal,
    init_from_none,
    initialise_snapshot,
    make_orthorhombic,
    minimize_snapshot,
)
from sdrun.util import get_num_mols

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_init_from_none(mol_params):
    """Ensure init_from_none has the correct type and number of particles."""
    snap = init_from_none(mol_params)
    # Each unit cell should create a single particle
    num_mols = np.prod(np.array(mol_params.cell_dimensions))
    assert snap.particles.N == num_mols * mol_params.molecule.num_particles


def test_initialise_snapshot(mol_params):
    """Test initialisation from a snapshot works."""
    snap = init_from_none(mol_params)
    context = hoomd.context.initialize(mol_params.hoomd_args)
    sys = initialise_snapshot(snap, context, mol_params)
    assert isinstance(sys, hoomd.data.system_data)
    snap_init = sys.take_snapshot()
    # Ensure bodies are initialised
    if mol_params.molecule.rigid:
        assert np.any(snap.particles.body != 2 ** 32 - 1)
    # Ensure all particles in molecules are created
    num_mols = np.prod(np.array(mol_params.cell_dimensions))
    num_particles = num_mols * mol_params.molecule.num_particles
    assert snap_init.particles.N == num_particles


def test_initialise_randomise(mol_params):
    snapshot = hoomd.data.gsd_snapshot("test/data/Trimer-13.50-3.00.gsd")
    context = hoomd.context.initialize(mol_params.hoomd_args)
    num_mols = get_num_mols(snapshot)
    with mol_params.temp_context(iteration_id=0):
        sys = initialise_snapshot(snapshot, context, mol_params)
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


def test_trimerP2_init_position():
    sim_params = SimulationParams(crystal=TrimerP2(), cell_dimensions=1)
    snap = init_from_crystal(sim_params, equilibration=False, minimize=False)
    manual = np.array(
        [
            [-1.0885514, 1.0257734, -0.5],
            [1.0885514, -1.0257734, -0.5],
            [-1.3476, 0.81600004, -0.5],
            [-0.8740344, -0.74631166, -0.5],
            [-0.4140196, 0.45763204, -0.5],
            [1.3476, -0.8159999, -0.5],
            [0.87403464, 0.7463118, -0.5],
            [0.41401944, -0.45763224, -0.5],
        ],
        dtype=np.float32,
    )
    assert_allclose(snap.particles.position, manual)


@pytest.mark.xfail()
def test_trimerPg_init_position():
    sim_params = SimulationParams(crystal=TrimerPg(), cell_dimensions=1)
    snap = init_from_crystal(sim_params, equilibration=False, minimize=False)
    manual = np.array(
        [
            [-1.0885514, 1.0257734, -0.5],
            [1.0885514, -1.0257734, -0.5],
            [-1.3476, 0.81600004, -0.5],
            [-0.8740344, -0.74631166, -0.5],
            [-0.4140196, 0.45763204, -0.5],
            [1.3476, -0.8159999, -0.5],
            [0.87403464, 0.7463118, -0.5],
            [0.41401944, -0.45763224, -0.5],
        ],
        dtype=np.float32,
    )
    assert_allclose(snap.particles.position, manual)


def test_init_crystal_position(crystal_params):
    if isinstance(crystal_params.crystal, (TrimerP2)):
        return
    with crystal_params.temp_context(cell_dimensions=1):
        snap = init_from_crystal(crystal_params, equilibration=False, minimize=False)

        crys = crystal_params.crystal

        num_mols = get_num_mols(snap)
        mol_positions = crystal_params.molecule.get_relative_positions()
        positions = np.concatenate(
            [
                pos + rowan.rotate(orient, mol_positions)
                for pos, orient in zip(crys.positions, crys.get_orientations())
            ]
        )
        box = np.array([snap.box.Lx, snap.box.Ly, snap.box.Lz])
        if crys.molecule.rigid:
            sim_pos = snap.particles.position[num_mols:] % box
        else:
            sim_pos = snap.particles.position % box
        init_pos = positions % box
        assert_allclose(sim_pos, init_pos)


@pytest.mark.parametrize("ensemble", ["NVE", "NPH"])
def test_minimize_snapshot(crystal_params, ensemble):
    snap = init_from_crystal(crystal_params, equilibration=False, minimize=False)
    min_snap = minimize_snapshot(snap, crystal_params, ensemble)
    if ensemble == "NPH":
        rtol = 0.2
    else:
        rtol = 0
    assert_allclose(snap.box.Lx, min_snap.box.Lx, rtol=rtol)
    assert_allclose(snap.box.Ly, min_snap.box.Ly, rtol=rtol)
    assert_allclose(snap.box.Lz, min_snap.box.Lz, rtol=rtol)
    assert_allclose(snap.box.xy, min_snap.box.xy, rtol=rtol)
    assert_allclose(snap.box.xz, min_snap.box.xz, rtol=rtol)
    assert_allclose(snap.box.yz, min_snap.box.yz, rtol=rtol)


def test_init_crystal(crystal_params):
    """Test the initialisation of all crystals."""
    snap = init_from_crystal(crystal_params)
    Nx, Ny, _ = crystal_params.cell_dimensions
    unitcell = crystal_params.crystal.cell_matrix
    logger.debug("Unitcell: %s", unitcell)
    Lx = np.linalg.norm(np.dot(unitcell, np.array([1, 0, 0])))
    Ly = np.linalg.norm(np.dot(unitcell, np.array([0, 1, 0])))

    assert Lx > 0
    assert Ly > 0
    assert Nx > 0
    assert Ny > 0

    # Simulation box within 20% of initialisation
    assert_allclose(snap.box.Lx, Lx * Nx, rtol=0.2)
    assert_allclose(snap.box.Ly, Ly * Ny, rtol=0.2)


def test_orthorhombic_null(mol_params):
    """Ensure null operation with orthorhombic function.

    In the case where the unit cell is already orthorhombic,
    check that nothing has changed unexpectedly.
    """
    with hoomd.context.initialize(mol_params.hoomd_args):
        snap = init_from_none(mol_params)
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
def test_orthorhombic_init(crystal_params, cell_dimensions):
    """Ensure orthorhombic cell initialises correctly."""
    snap = init_from_crystal(crystal_params)
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
def test_moment_inertia(mol_params, scaling_factor):
    """Ensure moment of inertia is set correctly in setup."""
    init_mol = np.array(mol_params.molecule.moment_inertia)
    logger.debug("Moment Inertia before scaling: %s", init_mol)
    init_mol *= scaling_factor
    logger.debug("Moment Inertia after scaling: %s", init_mol)
    with mol_params.temp_context(moment_inertia_scale=scaling_factor):
        snapshot = init_from_none(mol_params)
        context = hoomd.context.initialize(mol_params.hoomd_args)
        snapshot = initialise_snapshot(snapshot, context, mol_params).take_snapshot()
        num_mols = get_num_mols(snapshot)
        logger.debug(
            "Simulation Moment Inertia: %s", snapshot.particles.moment_inertia[0]
        )
        logger.debug("Intended Moment Inertia: %s", init_mol)
        diff = snapshot.particles.moment_inertia[:num_mols] - init_mol
        assert_allclose(diff, 0, atol=1e-1)


def test_thermalise(snapshot):
    pass
