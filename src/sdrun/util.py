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
from typing import List, Optional

import attr
import hoomd
import numpy as np
import rowan
from hoomd import md
from hoomd.data import SnapshotParticleData as Snapshot, system_data as System
from hoomd.group import group as Group

from .params import SimulationParams

logger = logging.getLogger(__name__)


def set_integrator(
    sim_params: SimulationParams,
    group: Group,
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
    assert group is not None

    md.integrate.mode_standard(sim_params.step_size)
    if sim_params.molecule.dimensions == 2:
        md.update.enforce2d()

    if prime_interval:
        md.update.zero_momentum(period=prime_interval, phase=-1)

    if integration_method == "NPT":
        integrator = md.integrate.npt(
            group=group,
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
            group=group, kT=sim_params.temperature, tau=sim_params.tau
        )

    return integrator


def set_dump(
    group: Group,
    outfile: Path,
    dump_period: Optional[int] = 10000,
    timestep: int = 0,
    extension: bool = True,
) -> hoomd.dump.gsd:
    """Initialise dumping configuration to a file."""
    assert group is not None
    # Ensure outfile is a Path object
    outfile = Path(outfile)
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
    group: Group, outfile: Path, timestep: int = 0, extension: bool = True
) -> hoomd.dump.gsd:
    """Dump a single frame to an output frame now."""
    return set_dump(
        group, outfile=outfile, dump_period=None, extension=extension, timestep=timestep
    )


def set_thermo(
    outfile: Path, thermo_period: int = 10000, rigid=True
) -> hoomd.analyze.log:
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
    outfile = Path(outfile).with_suffix(".log")
    return hoomd.analyze.log(
        str(outfile), quantities=default + rigid_thermo, period=thermo_period
    )


def get_group(
    sys: System, sim_params: SimulationParams, interface: bool = False
) -> Group:
    if sim_params.molecule.rigid:
        group = hoomd.group.rigid_center()
    else:
        group = hoomd.group.all()
    if interface is True:
        return _interface_group(sys, group)
    return group


def _interface_group(sys: System, base_group: Group, stationary: bool = False):
    assert base_group is not None
    stationary_group = hoomd.group.cuboid(
        name="stationary",
        xmin=-sys.box.Lx / 3,
        xmax=sys.box.Lx / 3,
        ymin=-sys.box.Ly / 3,
        ymax=sys.box.Ly / 3,
    )
    if stationary:
        return hoomd.group.intersection(
            "rigid_stationary", stationary_group, base_group
        )

    return hoomd.group.intersection(
        "rigid_mobile",
        hoomd.group.difference("mobile", hoomd.group.all(), stationary_group),
        base_group,
    )


@attr.s(auto_attribs=True)
class NumBodies:
    particles: int = attr.ib(converter=int)
    molecules: int = attr.ib(converter=int)


def _get_num_bodies(snapshot: Snapshot) -> NumBodies:
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


def get_num_mols(snapshot: Snapshot) -> int:
    """The number of molecules (rigid bodies) in the snapshot."""
    num_bodies = _get_num_bodies(snapshot)
    return num_bodies.molecules


def get_num_particles(snapshot: Snapshot) -> int:
    """The number of particles in the snapshot."""
    num_bodies = _get_num_bodies(snapshot)
    return num_bodies.particles


def z2quaternion(theta: np.ndarray) -> np.ndarray:
    """Convert a rotation about the z axis to a quaternion.

    This is a helper for 2D simulations, taking the rotation of a particle about the z axis and
    converting it to a quaternion. The input angle `theta` is assumed to be in radians.

    """
    return rowan.from_euler(theta, 0, 0).astype(np.float32)


def compute_translational_ke(snapshot: Snapshot) -> float:
    """Compute the kinetic energy of the translational degrees of freedom.

    Args:
        snapshot: (Snapshot): Simulation snapshot from which to compute the kinetic energy

    Returns: The total translational kinetic energy of the snapshot.

    """
    num_mols = get_num_mols(snapshot)
    return 0.5 * np.sum(
        snapshot.particles.mass[:num_mols].reshape((-1, 1))
        * np.square(snapshot.particles.velocity[:num_mols])
    )


def compute_rotational_ke(snapshot: Snapshot) -> float:
    """Compute the kinetic energy of the rotational degrees of freedom.

    Args:
        snapshot: (Snapshot): Simulation snapshot from which to compute the kinetic energy

    Returns: The total rotational kinetic energy of the snapshot.

    """
    num_mols = get_num_mols(snapshot)
    angmom = snapshot.particles.angmom[:num_mols]
    moment_inertia = snapshot.particles.moment_inertia[:num_mols]
    momentum = rowan.multiply(
        0.5 * rowan.conjugate(snapshot.particles.orientation[:num_mols]), angmom
    )[:, 1:]
    mask = moment_inertia > 0
    return np.sum(0.5 * np.square(momentum[mask]) / moment_inertia[mask])
