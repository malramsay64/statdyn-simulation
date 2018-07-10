#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Series of helper functions for the initialisation of parameters."""

import logging
from pathlib import Path
from typing import List, NamedTuple

import hoomd
import hoomd.md as md
import numpy as np
from hoomd.data import SnapshotParticleData
from hoomd.harmonic_force import HarmonicForceCompute

from .params import SimulationParams

logger = logging.getLogger(__name__)


def set_integrator(
    sim_params: SimulationParams,
    prime_interval: int = 33533,
    simulation_type: str = "liquid",
    integration_method: str = "NPT",
) -> hoomd.md.integrate.npt:
    """Hoomd integrate method.

    Args:
        sim_params: The general parameters of the simulation.
        prime_interval: The number of steps between zeroing the momentum of the simulation.
            A prime number is chosen to have this change be less systematic.
        simulation_type: The type of simulation, which influences the parameters for the
            integration. Most simulations are fine with the default, "liq". The alternatives
            "crys" allows for simulation cell to tilt and decoupling of cell parameters, and
            "interface" ensures all particles are rescaled instead of just those being
            integrated.
        integration_method: The type of thermodynamic integration.

    """
    assert integration_method in ["NPT", "NVT"]
    assert simulation_type in ["liquid", "crystal", "interface"]

    md.integrate.mode_standard(sim_params.step_size)
    if sim_params.molecule.dimensions == 2:
        md.update.enforce2d()

    if prime_interval:
        md.update.zero_momentum(period=prime_interval, phase=-1)

    if integration_method == "NPT":
        integrator = md.integrate.npt(
            group=sim_params.group,
            kT=sim_params.temperature,
            tau=sim_params.tau,
            P=sim_params.pressure,
            tauP=sim_params.tauP,
        )
        if simulation_type == "crystal":
            integrator.couple = "none"
            integrator.xy = True
            integrator.xz = True
            integrator.yz = True
        elif simulation_type == "interface":
            integrator.set_params(rescale_all=True)

    elif integration_method == "NVT":
        integrator = md.integrate.nvt(
            group=sim_params.group, kT=sim_params.temperature, tau=sim_params.tau
        )

    return integrator


def set_dump(
    outfile: Path,
    dump_period: int = 10000,
    timestep: int = 0,
    group: hoomd.group.group = None,
    extension: bool = True,
) -> hoomd.dump.gsd:
    """Initialise dumping configuration to a file."""
    if group is None:
        group = hoomd.group.rigid_center()
    if extension:
        outfile = outfile.with_suffix(".gsd")
    return hoomd.dump.gsd(
        str(outfile),
        time_step=timestep,
        period=dump_period,
        group=group,
        overwrite=True,
    )


def dump_frame(
    outfile: Path,
    timestep: int = 0,
    group: hoomd.group.group = None,
    extension: bool = True,
) -> hoomd.dump.gsd:
    return set_dump(
        outfile=outfile,
        dump_period=None,
        group=group,
        extension=extension,
        timestep=timestep,
    )


def set_thermo(outfile: Path, thermo_period: int = 10000, rigid=True) -> None:
    """Set the thermodynamic quantities for a simulation."""
    default = [
        "N",
        "volume",
        "momentum",
        "temperature",
        "pressure",
        "potential_energy",
        "kinetic_energy",
        "translational_kinetic_energy",
        "rotational_kinetic_energy",
        "npt_thermostat_energy",
        "lx",
        "ly",
        "lz",
        "xy",
        "xz",
        "yz",
    ]
    rigid_thermo: List[str] = []
    if rigid:
        rigid_thermo = [
            "temperature_rigid_center",
            "pressure_rigid_center",
            "potential_energy_rigid_center",
            "kinetic_energy_rigid_center",
            "translational_kinetic_energy_rigid_center",
            "translational_ndof_rigid_center",
            "rotational_ndof_rigid_center",
        ]
    # TODO Set logger to hdf5 file
    outfile = outfile.with_suffix(".log")
    hoomd.analyze.log(
        str(outfile), quantities=default + rigid_thermo, period=thermo_period
    )


def set_harmonic_force(
    snapshot: SnapshotParticleData, sim_params: SimulationParams
) -> None:
    assert sim_params.harmonic_force is not None
    if sim_params.harmonic_force == 0:
        return
    num_mols = get_num_mols(snapshot)
    HarmonicForceCompute(
        sim_params.group,
        snapshot.particles.position[:num_mols],
        snapshot.particles.orientation[:num_mols],
        sim_params.harmonic_force,
        sim_params.harmonic_force,
    )


def randomise_momenta(
    snapshot: SnapshotParticleData, interface: bool = False, random_seed=None
) -> SnapshotParticleData:
    """Randomise the momenta of particles in a snapshot."""
    num_mols = get_num_mols(snapshot)

    if random_seed is not None:
        np.random.RandomState(seed=random_seed)

    if interface:
        velocity_dist = snapshot.particles.velocity[:num_mols]
        velocity_dist = velocity_dist[np.linalg.norm(velocity_dist, axis=1) > 1e-5]
        # Flatten array to choose from distribution of all values
        #  velocity_dist = velocity_dist.flatten()
        assert velocity_dist.shape[0] > 0

        angmom_dist = snapshot.particles.angmom[:num_mols]
        # Only choose values that are non-zero
        angmom_dist = angmom_dist[np.linalg.norm(angmom_dist, axis=1) > 0]
        assert angmom_dist.shape[0] > 0

        for i in range(num_mols):
            snapshot.particles.velocity[i] = velocity_dist[
                np.random.choice(velocity_dist.shape[0], 1)
            ]
            snapshot.particles.angmom[i] = angmom_dist[
                np.random.choice(angmom_dist.shape[0], 1)
            ]

        return snapshot

    np.random.shuffle(snapshot.particles.velocity[:num_mols])
    np.random.shuffle(snapshot.particles.angmom[:num_mols])
    return snapshot


class NumBodies(NamedTuple):
    particles: int
    molecules: int


def _get_num_bodies(snapshot: SnapshotParticleData) -> NumBodies:
    try:
        num_particles = snapshot.particles.N
        num_mols = max(snapshot.particles.body) + 1
    except (AttributeError, ValueError):
        num_particles = snapshot.particles.N
        num_mols = num_particles
    if num_mols > num_particles:
        num_mols = num_particles

    assert (
        num_mols <= num_particles
    ), f"Num molecule: {num_mols}, Num particles {num_particles}"
    assert num_particles == len(snapshot.particles.position)

    return NumBodies(num_particles, num_mols)


def get_num_mols(snapshot: SnapshotParticleData) -> int:
    num_bodies = _get_num_bodies(snapshot)
    return num_bodies.molecules


def get_num_particles(snapshot: SnapshotParticleData) -> int:
    num_bodies = _get_num_bodies(snapshot)
    return num_bodies.particles
