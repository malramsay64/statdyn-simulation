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
from typing import Tuple

import hoomd
import hoomd.md
import numpy as np
import rowan
from hoomd.context import SimulationContext as Context
from hoomd.data import SnapshotParticleData as Snapshot, system_data as System
from scipy.stats import maxwell, norm

from .molecules import Molecule
from .params import SimulationParams
from .util import get_num_mols, get_num_particles, randomise_momenta, z2quaternion

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
        rigid = molecule.define_rigid()
        if rigid:
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
    num_x, num_y, num_z = sim_params.cell_dimensions
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
    snapshot = minimize_snapshot(snapshot, sim_params, ensemble="NPH")
    if equilibration:
        from .simulation import equilibrate

        equilibrate(snapshot, sim_params, equil_type="liquid")
    return snapshot


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
        rigid = sim_params.molecule.define_rigid()
        if rigid:
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
        rigid = sim_params.molecule.define_rigid()
        if rigid:
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
    sim_params: SimulationParams, equilibration: bool = False
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
        for p_type in sim_params.molecule.get_types()[1:]:
            sys.particles.pdata.addType(p_type)
        logger.debug("Particle Types: %s", sys.particles.types)
        rigid = sim_params.molecule.define_rigid()
        if rigid:
            rigid.create_bodies()
        snap = sys.take_snapshot(all=True)
        logger.debug("Particle Types: %s", snap.particles.types)
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


def thermalise(snapshot: Snapshot, sim_params: SimulationParams) -> Snapshot:
    """Set the momentum of the particles to the temperature distribution."""
    size = get_num_mols(snapshot)

    snapshot.particles.velocity[:size] = _scale_velocity(size, sim_params)
    snapshot.particles.angmom[:size] = _scale_angmom(size, sim_params, snapshot)

    return snapshot


def _calc_orient_energy(snapshot: Snapshot):
    num_mols = get_num_mols(snapshot)
    momentum = rowan.multiply(
        0.5 * rowan.conjugate(snapshot.particles.orientation[:num_mols]),
        snapshot.particles.angmom[:num_mols],
    )
    return np.nansum(
        0.5 * np.square(momentum)[:, 1:] / snapshot.particles.moment_inertia[:num_mols]
    )


def _scale_angmom(
    size: int, sim_params: SimulationParams, snapshot: Snapshot
) -> np.ndarray:
    energy_distribution = _generate_energies(size)
    if sim_params.molecule.dimensions == 2:
        direction_distribution = _generate_vectors(size, 1)[:, 0].reshape((-1, 1))
    elif sim_params.molecule.dimensions == 3:
        raise NotImplementedError(
            "Scaling angular momentum for 3d molecules is not yet implemented"
        )

    target_energy = 0.5 * sim_params.temperature

    def required_scaling(E):
        """Curve obtained from experiment."""
        return (np.sqrt(E) + -5) / 4.9

    num_mols = get_num_mols(snapshot)
    angmom = np.ones_like(snapshot.particles.orientation[:num_mols])
    angmom[:, 3] = _generate_energies(num_mols).reshape(-1) * required_scaling(
        target_energy
    )

    return angmom


def _scale_velocity(size: int, sim_params: SimulationParams) -> np.ndarray:
    energy_distribution = _generate_energies(size)
    direction_distribution = _generate_vectors(size, sim_params.molecule.dimensions)

    temperature = sim_params.temperature
    mass = 1
    velocity_magnitude = np.sqrt(2 * temperature / mass)
    logger.debug("Velocity: %s", velocity_magnitude)

    velocities = energy_distribution * direction_distribution * velocity_magnitude
    # zero net velocity
    velocities -= np.mean(velocities, axis=0)

    return velocities


def _generate_energies(size: int) -> np.ndarray:
    """Create a boltzmann distribution of energies.

    A function for generating a random distribution of values in a boltzmann distribution, which
    is the distribution for energies in a molecular simulation. This function is concerned with
    the shape of the distribution, with the values intended to be rescaled for the required
    temperature.

    """
    energy = maxwell.rvs(1, 0.2, size=size) / maxwell.mean(1, 0.2)
    return energy.reshape((-1, 1))


def _generate_vectors(size: int, dimensions: int = 2) -> np.ndarray:
    """Create a random distribution of unit vectors."""
    # The minimum number of dimensions for a position vector is 3, this would also allow for 4D
    # values like quaternions.
    vec_dim = max(3, dimensions)

    vec = norm.rvs(size=(size, vec_dim))

    # Set values of dimensions larger than shpae to 0
    if vec.shape[1] > dimensions:
        vec[:, dimensions:] = 0
    vec /= np.linalg.norm(vec, axis=1).reshape((-1, 1))
    return vec


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
