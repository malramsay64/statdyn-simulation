#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Series of helper functions for the initialisation of parameters."""

import logging
from typing import List, NamedTuple

import hoomd
import hoomd.md as md
from hoomd.harmonic_force import HarmonicForceCompute

from .params import SimulationParams

logger = logging.getLogger(__name__)


def set_integrator(
    sim_params: SimulationParams,
    prime_interval: int = 33533,
    crystal: bool = False,
    create: bool = True,
    integration_method: str = "NPT",
) -> hoomd.md.integrate.npt:
    """Hoomd integrate method."""
    assert integration_method in ["NPT", "NVT"]

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
        if crystal:
            integrator.set_params(rescale_all=True)
            integrator.couple = "none"
    elif integration_method == "NVT":
        integrator = md.integrate.nvt(
            group=sim_params.group, kT=sim_params.temperature, tau=sim_params.tau
        )

    return integrator


def set_dump(
    outfile: str,
    dump_period: int = 10000,
    timestep: int = 0,
    group: hoomd.group.group = None,
    extension: bool = True,
) -> hoomd.dump.gsd:
    """Initialise dumping configuration to a file."""
    if group is None:
        group = hoomd.group.rigid_center()
    if extension:
        outfile += ".gsd"
    return hoomd.dump.gsd(
        outfile, time_step=timestep, period=dump_period, group=group, overwrite=True
    )


def dump_frame(
    outfile: str,
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


def set_thermo(outfile: str, thermo_period: int = 10000, rigid=True) -> None:
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
    hoomd.analyze.log(
        outfile + ".log", quantities=default + rigid_thermo, period=thermo_period
    )


class NumBodies(NamedTuple):
    particles: int
    molecules: int


def _get_num_bodies(snapshot: hoomd.data.SnapshotParticleData):
    try:
        num_particles = snapshot.particles.N
        num_mols = max(snapshot.particles.body) + 1
    except (AttributeError, ValueError):
        num_particles = len(snapshot.particles.N)
        num_mols = num_particles
    if num_mols > num_particles:
        num_mols = num_particles

    assert (
        num_mols <= num_particles
    ), f"Num molecule: {num_mols}, Num particles {num_particles}"
    assert num_particles == len(snapshot.particles.position)

    return NumBodies(num_particles, num_mols)


def get_num_mols(snapshot: hoomd.data.SnapshotParticleData) -> int:
    num_bodies = _get_num_bodies(snapshot)
    return num_bodies.molecules


def get_num_particles(snapshot: hoomd.data.SnapshotParticleData) -> int:
    num_bodies = _get_num_bodies(snapshot)
    return num_bodies.particles


def set_harmonic_force(
    snapshot: hoomd.data.SnapshotParticleData, sim_params: SimulationParams
) -> None:
    assert sim_params.parameters.get("harmonic_force") is not None
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
