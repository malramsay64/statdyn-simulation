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
from typing import Tuple, Union

import hoomd
import hoomd.md as md
import numpy as np

from .helper import dump_frame
from .molecules import Molecule
from .params import SimulationParams

logger = logging.getLogger(__name__)
UnitCellLengths = Union[Tuple[int, int, int], Tuple[int, int]]


def init_from_file(
    fname: Path, molecule: Molecule, hoomd_args: str = ""
) -> hoomd.data.SnapshotParticleData:
    """Initialise a hoomd simulation from an input file."""
    logger.debug("Initialising from file %s", fname)
    # Hoomd context needs to be initialised before calling gsd_snapshot
    logger.debug("Hoomd Arguments: %s", hoomd_args)
    temp_context = hoomd.context.initialize(hoomd_args)
    with temp_context:
        snapshot = hoomd.data.gsd_snapshot(str(fname), frame=0)
        sys = hoomd.init.read_snapshot(snapshot)
        rigid = molecule.define_rigid()
        if rigid:
            rigid.create_bodies()
        init_snapshot = sys.take_snapshot(all=True)
    return init_snapshot


def init_from_none(sim_params: SimulationParams) -> hoomd.data.SnapshotParticleData:
    """Initialise a system from no inputs.

    This creates a simulation with a large unit cell lattice such that there
    is no chance of molecules overlapping and places molecules on the lattice.

    """
    logger.debug("Hoomd Arguments: %s", sim_params.hoomd_args)
    try:
        num_x, num_y, num_z = sim_params.cell_dimensions
    except ValueError:
        num_x, num_y = sim_params.cell_dimensions
        num_z = 1
    molecule = sim_params.molecule
    mol_size = molecule.compute_size()
    num_molecules = num_x * num_y * num_z
    len_x = mol_size * num_x
    len_y = mol_size * num_y
    len_z = mol_size * num_z
    box = hoomd.data.boxdim(
        Lx=len_x, Ly=len_y, Lz=len_z, dimensions=molecule.dimensions
    )
    with hoomd.context.initialize(sim_params.hoomd_args):
        snapshot = hoomd.data.make_snapshot(
            N=molecule.num_particles * num_molecules,
            box=box,
            particle_types=molecule.get_types(),
        )
        # Generate list of positions on grid
        xpos, ypos, zpos = np.mgrid[
            -len_x / 2 : len_x / 2 : mol_size,
            -len_y / 2 : len_y / 2 : mol_size,
            -len_z / 2 : len_z / 2 : mol_size,
        ]
        cell_positions = np.array([xpos.flatten(), ypos.flatten(), zpos.flatten()]).T
        positions = np.concatenate(
            [cell_positions + mol_pos for mol_pos in molecule.positions], axis=0
        )
        positions += np.array([mol_size / 2, mol_size / 2, mol_size / 2])
        # Check we are using the master process to update snapshot
        if hoomd.comm.get_rank() == 0:
            # Set values in snapshot
            snapshot.particles.position[:] = positions
            snapshot.particles.typeid[:] = molecule.identify_particles(num_molecules)
            snapshot.particles.body[:] = molecule.identify_bodies(num_molecules)
            snapshot.particles.moment_inertia[:] = np.array(
                [molecule.moment_inertia] * num_molecules * molecule.num_particles
            )
    return minimize_snapshot(snapshot, sim_params, ensemble="NPH")


def initialise_snapshot(
    snapshot: hoomd.data.SnapshotParticleData,
    context: hoomd.context.SimulationContext,
    sim_params: SimulationParams,
    minimize: bool = False,
) -> hoomd.data.system_data:
    """Initialise the configuration from a snapshot.

    In this function it is checked that the data in the snapshot and the
    passed arguments are in agreement with each other, and rectified if not.
    """
    with context:
        try:
            num_particles = snapshot.particles.N
            num_mols = max(snapshot.particles.body) + 1
        except (AttributeError, ValueError):
            num_particles = len(snapshot.particles.position)
            num_mols = num_particles
        logger.debug(
            "Number of particles: %d , Number of molecules: %d", num_particles, num_mols
        )
        snapshot = _check_properties(snapshot, molecule)
        sys = hoomd.init.read_snapshot(snapshot)
        sim_params.molecule.define_potential()
        sim_params.molecule.define_dimensions()
        rigid = sim_params.molecule.define_rigid()
        if rigid:
            rigid.check_initialization()
        return sys


def minimize_snapshot(snapshot, molecule, hoomd_args: str = ""):
    temp_context = hoomd.context.initialize(hoomd_args)
    with temp_context:
        sys = hoomd.init.read_snapshot(snapshot)
        molecule.define_potential()
        molecule.define_dimensions()
        rigid = molecule.define_rigid()
        if rigid:
            rigid.check_initialization()
            group = hoomd.group.rigid_center()
        else:
            group = hoomd.group.all()
        logger.debug("Minimizing energy")
        fire = hoomd.md.integrate.mode_minimize_fire(0.001)
        nph = hoomd.md.integrate.nph(group=group, P=1.0, tauP=5)
        num_steps = 0
        while not fire.has_converged():
            hoomd.run(100)
            num_steps += 100
            if num_steps > 10_000:
                break

        nph.disable()
        logger.debug("Energy Minimized in %s steps", num_steps)
        equil_snapshot = sys.take_snapshot(all=True)
    return equil_snapshot


def init_from_crystal(sim_params: SimulationParams,) -> hoomd.data.SnapshotParticleData:
    """Initialise a hoomd simulation from a crystal lattice.

    Args:
        crystal (class:`statdyn.crystals.Crystal`): The crystal lattice to
            generate the simulation from.
    """
    logger.info("Hoomd Arguments: %s", sim_params.hoomd_args)
    assert hasattr(sim_params, "cell_dimensions")
    assert hasattr(sim_params, "crystal")
    assert hasattr(sim_params, "molecule")
    temp_context = hoomd.context.initialize(sim_params.hoomd_args)
    with temp_context:
        logger.debug(
            "Creating %s cell of size %s",
            sim_params.crystal,
            sim_params.cell_dimensions,
        )
        sys = hoomd.init.create_lattice(
            unitcell=sim_params.crystal.get_unitcell(), n=sim_params.cell_dimensions
        )
        for p_type in sim_params.molecule.get_types()[1:]:
            sys.particles.pdata.addType(p_type)
        logger.debug("Particle Types: %s", sys.particles.types)
        rigid = sim_params.molecule.define_rigid()
        if rigid:
            rigid.create_bodies()
        snap = sys.take_snapshot(all=True)
        logger.debug("Particle Types: %s", snap.particles.types)
    temp_context = hoomd.context.initialize(sim_params.hoomd_args)
    with temp_context:
        sys = initialise_snapshot(snap, temp_context, sim_params)
        md.integrate.mode_standard(dt=sim_params.step_size)
        md.integrate.npt(
            group=sim_params.group,
            kT=sim_params.temperature,
            xy=True,
            couple="none",
            P=sim_params.pressure,
            tau=sim_params.tau,
            tauP=sim_params.tauP,
        )
        equil_snap = sys.take_snapshot(all=True)
        dump_frame(sim_params.filename(), group=sim_params.group)
    return make_orthorhombic(equil_snap)


def make_orthorhombic(
    snapshot: hoomd.data.SnapshotParticleData
) -> hoomd.data.SnapshotParticleData:
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
    xlen = len_x + snapshot.box.xy * len_y
    snapshot.particles.position[:, 0] += xlen / 2.
    snapshot.particles.position[:, 0] %= len_x
    snapshot.particles.position[:, 0] -= len_x / 2.
    logger.debug("Updated positions: \n%s", snapshot.particles.position)
    box = hoomd.data.boxdim(len_x, len_y, len_z, 0, 0, 0, dimensions=2)
    hoomd.data.set_snapshot_box(snapshot, box)
    return snapshot


def _check_properties(
    snapshot: hoomd.data.SnapshotParticleData, molecule: Molecule
) -> hoomd.data.SnapshotParticleData:
    try:
        nbodies = min(len(snapshot.particles.body), max(snapshot.particles.body) + 1)
        logger.debug("number of rigid bodies: %d", nbodies)
        snapshot.particles.types = molecule.get_types()
        snapshot.particles.moment_inertia[:nbodies] = np.array(
            [molecule.moment_inertia] * nbodies
        )
    except (AttributeError, ValueError):
        num_atoms = len(snapshot.particles.position)
        logger.debug("num_atoms: %d", num_atoms)
        if num_atoms > 0:
            snapshot.particles.types = molecule.get_types()
            snapshot.particles.moment_inertia[:] = np.array(
                [molecule.moment_inertia] * num_atoms
            )
    return snapshot
