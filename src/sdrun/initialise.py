#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Module for initialisation of a hoomd simulation environment.

This module allows the initialisation from a number of different starting
configurations, whether that is a file, a crystal lattice, or no predefined
config.
"""
import logging
from pathlib import Path

import hoomd
import hoomd.md
import numpy as np
from hoomd.context import SimulationContext as Context
from hoomd.data import SnapshotParticleData as Snapshot, system_data as System

from .crystals import Crystal
from .molecules import Molecule
from .params import SimulationParams
from .util import get_num_mols, get_num_particles, randomise_momenta

logger = logging.getLogger(__name__)


def init_from_file(fname: Path, molecule: Molecule, hoomd_args: str = "") -> Snapshot:
    """Initialise a hoomd simulation from an input file."""
    logger.debug("Initialising from file %s", fname)
    # Hoomd context needs to be initialised before calling gsd_snapshot
    logger.debug("Hoomd Arguments: %s", hoomd_args)
    temp_context = hoomd.context.initialize(hoomd_args)
    with temp_context:
        logger.debug("Reading snapshot: %s", fname)
        snapshot = hoomd.data.gsd_snapshot(str(fname), frame=0)
        sys = hoomd.init.read_snapshot(snapshot)
        if molecule.rigid:
            rigid = molecule.define_rigid()
            rigid.create_bodies()
        init_snapshot = sys.take_snapshot(all=True)
    return init_snapshot


def init_from_none(
    sim_params: SimulationParams, equilibration: bool = False
) -> Snapshot:
    """Initialise a system from no inputs.

    This creates a simulation with a large unit cell lattice such that there
    is no chance of molecules overlapping and places molecules on the lattice.

    """
    logger.debug("Hoomd Arguments: %s", sim_params.hoomd_args)
    mol_size = sim_params.molecule.compute_size()

    crystal = Crystal(
        cell_matrix=mol_size * np.identity(3), molecule=sim_params.molecule
    )
    with sim_params.temp_context(crystal=crystal):
        return init_from_crystal(sim_params, equilibration)


def initialise_snapshot(
    snapshot: Snapshot,
    context: Context,
    sim_params: SimulationParams,
    minimize: bool = False,
) -> System:
    """Initialise the configuration from a snapshot.

    In this function it is checked that the data in the snapshot and the
    passed arguments are in agreement with each other, and rectified if not.
    """

    # Only use the master process to check the snapshot
    if hoomd.comm.get_rank() == 0:
        num_molecules = get_num_mols(snapshot)
        num_particles = get_num_particles(snapshot)
        logger.debug(
            "Number of particles: %d , Number of molecules: %d",
            num_particles,
            num_molecules,
        )
        snapshot = _check_properties(snapshot, sim_params.molecule)

    if minimize:
        snapshot = minimize_snapshot(snapshot, sim_params, ensemble="NVE")

    if sim_params.iteration_id is not None:
        interface = False
        # Interface simulations require the space_group paramter to be set
        if sim_params.space_group is not None:
            interface = True
        snapshot = randomise_momenta(
            snapshot, interface, random_seed=sim_params.iteration_id
        )

    with context:
        sys = hoomd.init.read_snapshot(snapshot)
        sim_params.molecule.define_potential()
        sim_params.molecule.define_dimensions()
        if sim_params.molecule.rigid:
            rigid = sim_params.molecule.define_rigid()
            rigid.check_initialization()
        return sys


def minimize_snapshot(
    snapshot: Snapshot, sim_params: SimulationParams, ensemble: str = "NVE"
) -> Snapshot:
    assert ensemble in ["NVE", "NPH"]
    temp_context = hoomd.context.initialize(sim_params.hoomd_args)
    with temp_context:
        sys = hoomd.init.read_snapshot(snapshot)
        sim_params.molecule.define_potential()
        sim_params.molecule.define_dimensions()
        if sim_params.molecule.rigid:
            rigid = sim_params.molecule.define_rigid()
            rigid.check_initialization()
            group = hoomd.group.rigid_center()
        else:
            group = hoomd.group.all()

        logger.debug("Minimizing energy")
        fire = hoomd.md.integrate.mode_minimize_fire(0.001)

        if ensemble == "NVE":
            ensemble_integrator = hoomd.md.integrate.nve(group=group)
        elif ensemble == "NPH":
            ensemble_integrator = hoomd.md.integrate.nph(
                group=group, P=sim_params.pressure, tauP=sim_params.tauP
            )

        num_steps = 100
        while not fire.has_converged():
            hoomd.run(num_steps)
            num_steps *= 2
            if num_steps > sim_params.num_steps:
                break

        ensemble_integrator.disable()
        logger.debug("Energy Minimized in %s steps", num_steps)
        equil_snapshot = sys.take_snapshot(all=True)
    return equil_snapshot


def init_from_crystal(
    sim_params: SimulationParams, equilibration: bool = False, minimize: bool = True
) -> Snapshot:
    """Initialise a hoomd simulation from a crystal lattice.

    Args:
        crystal (class:`statdyn.crystals.Crystal`): The crystal lattice to
            generate the simulation from.
    """
    logger.info("Hoomd Arguments: %s", sim_params.hoomd_args)
    assert sim_params.cell_dimensions is not None
    assert sim_params.crystal is not None
    assert sim_params.molecule is not None
    temp_context = hoomd.context.initialize(sim_params.hoomd_args)
    with temp_context:
        logger.debug(
            "Creating %s cell of size %s",
            sim_params.crystal,
            sim_params.cell_dimensions,
        )
        cell_dimensions = sim_params.cell_dimensions[: sim_params.molecule.dimensions]
        sys = hoomd.init.create_lattice(
            unitcell=sim_params.crystal.get_unitcell(), n=cell_dimensions
        )
        for p_type in sim_params.molecule.get_types():
            sys.particles.pdata.addType(p_type)
        logger.debug("Particle Types: %s", sys.particles.types)
        if sim_params.molecule.rigid:
            rigid = sim_params.molecule.define_rigid()
            rigid.create_bodies()
        snap = sys.take_snapshot(all=True)
        logger.debug("Particle Types: %s", snap.particles.types)

    if minimize:
        snap = minimize_snapshot(snap, sim_params, ensemble="NPH")

    if equilibration:
        from .simulation import equilibrate

        snap = equilibrate(snap, sim_params, equil_type="crystal")
    return snap


def make_orthorhombic(snapshot: Snapshot) -> Snapshot:
    """Create orthorhombic unit cell from snapshot.

    This uses the periodic boundary conditions of the cell to generate an
    orthorhombic simulation cell from the input simulation environment. This
    is to ensure consistency within simulations and because it is simpler to
    use an orthorhombic simulation cell in calculations.

    Todo:
        This function doesn't yet account for particles within a molecule
        which are accross a simulation boundary. This needs to be fixed before
        this function is truly general, otherwise it only works with special
        cells.

    """
    logger.debug("Snapshot type: %s", snapshot)
    len_x = snapshot.box.Lx
    len_y = snapshot.box.Ly
    len_z = snapshot.box.Lz
    dimensions = snapshot.box.dimensions
    xlen = len_x + snapshot.box.xy * len_y
    snapshot.particles.position[:, 0] += xlen / 2.
    snapshot.particles.position[:, 0] %= len_x
    snapshot.particles.position[:, 0] -= len_x / 2.
    box = hoomd.data.boxdim(len_x, len_y, len_z, 0, 0, 0, dimensions=dimensions)
    hoomd.data.set_snapshot_box(snapshot, box)
    return snapshot


def _check_properties(snapshot: Snapshot, molecule: Molecule) -> Snapshot:
    num_molecules = get_num_mols(snapshot)
    num_particles = get_num_particles(snapshot)

    if num_molecules < num_particles:
        logger.debug("number of rigid bodies: %d", num_molecules)
        snapshot.particles.types = molecule.get_types()
        snapshot.particles.moment_inertia[:num_molecules] = np.array(
            [molecule.moment_inertia] * num_molecules
        )
    else:
        logger.debug("num_atoms: %d", num_particles)
        assert num_particles > 0
        snapshot.particles.types = molecule.get_types()
        snapshot.particles.moment_inertia[:] = np.array(
            [molecule.moment_inertia] * num_particles
        )
    return snapshot
