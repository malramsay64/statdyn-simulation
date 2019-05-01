#! /usr/bin/env python3
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
        group: The group of particles over which the integration is to take place.
        prime_interval: The number of steps between zeroing the momentum of the simulation.
            A prime number is chosen to have this change be less systematic.
        simulation_type: The type of simulation, which influences the parameters for the
            integration. Most simulations are fine with the default, "liq". The alternatives
            "crys" allows for simulation cell to tilt and decoupling of cell parameters, and
            "interface" ensures all particles are rescaled instead of just those being
            integrated.
        integration_method: The type of thermodynamic integration.

    """
    if integration_method not in ["NPT", "NVT"]:
        raise ValueError(
            f"Integration method must be in (NPT|NVT), found {integration_method}"
        )

    if simulation_type not in ["liquid", "crystal", "interface"]:
        raise ValueError(
            f"simulation_type must be one out (liquid|crystal|interface), found {simulation_type}"
        )

    if group is None:
        raise ValueError("group must not be none")

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
    """Initialise the dumping of configurations to a file.

    This initialises the simulation configuration to be output every
    `dump_period` timesteps.

    Args:
        group: The group of particles to dump.
        outfile: The file to store the output configurations.
        dump_period: How often the particles should be output. When
            this is None just the current frame is output. Defaults
            to 10000.
        timestep: The timestep to be associated with the current configuration, default is 0
        extension: Whether to manually set the file extension, default True

    """
    if group is None:
        raise ValueError("group must not be None")
    # Ensure outfile is a Path object
    outfile = Path(outfile)
    if extension:
        outfile = outfile.with_suffix(".gsd")
    return hoomd.dump.gsd(
        str(outfile),
        time_step=timestep,
        period=dump_period,
        group=group,
        dynamic=["property", "momentum"],
        overwrite=True,
    )


def dump_frame(
    group: Group, outfile: Path, timestep: int = 0, extension: bool = True
) -> hoomd.dump.gsd:
    """Dump a single frame to an output frame now.

    Args:
        group: The group of particles to dump.
        outfile: The file to store the output configurations.
        timestep: The timestep to be associated with the current configuration, default is 0
        extension: Whether to manually set the file extension, default True

    """
    return set_dump(
        group, outfile=outfile, dump_period=None, extension=extension, timestep=timestep
    )


def set_thermo(
    outfile: Path, thermo_period: int = 10000, rigid=True
) -> hoomd.analyze.log:
    """Set the thermodynamic quantities for a simulation.

    Args:
        outfile: File to output thermodynamic data to
        thermo_period: The period with which the thermodynamic
            data should be output. (Default is 10_000)
        rigid: Whether to output the additional information about
            the rigid group. (Default is True)

    """
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
    """Method for creating a Hoomd group instance

    This makes the creation of a group simpler, handling the different types of groups
    which are used in this project, those where there are rigid particles, those without,
    and a group of particles creating an interface.

    Args:
        sys: The system in which the group should be created.
        sim_params: Parameters for the simulation.
        interface: Whether to create a sub-group of mobile particles
            to assist in creating an interface. (Default is False)

    """
    if sim_params.molecule.rigid:
        group = hoomd.group.rigid_center()
    else:
        group = hoomd.group.all()
    if interface is True:
        return _interface_group(sys, group)
    return group


def _interface_group(sys: System, base_group: Group, stationary: bool = False):
    if base_group is None:
        raise ValueError("The base_group argument cannot be None")

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
class _NumBodies:
    particles: int = attr.ib(converter=int)
    molecules: int = attr.ib(converter=int)


def _get_num_bodies(snapshot: Snapshot) -> _NumBodies:
    try:
        num_particles = snapshot.particles.N
        num_mols = max(snapshot.particles.body) + 1
    except (AttributeError, ValueError):
        num_particles = snapshot.particles.N
        num_mols = num_particles
    if num_mols > num_particles:
        num_mols = num_particles

    if num_mols > num_particles:
        raise RuntimeError(
            "There more molecules than particles, calculation has failed"
            f"found {num_mols} molecules and {num_particles} particles."
        )

    if num_mols > len(snapshot.particles.position):
        raise RuntimeError(
            "There are more molecules than position vectors,"
            f"num_mols: {num_mols}, positions: {len(snapshot.particles.position)}"
        )

    if num_particles > len(snapshot.particles.position):
        raise RuntimeError(
            "There are more particles than position vectors,"
            f"num_particles: {num_particles}, positions: {len(snapshot.particles.position)}"
        )

    return _NumBodies(num_particles, num_mols)


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
