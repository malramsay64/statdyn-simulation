#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Series of helper functions for the initialisation of parameters."""

import logging
from functools import partial
from typing import List

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
    elif integration_method == "NVT":
        integrator = md.integrate.nvt(
            group=sim_params.group, kT=sim_params.temperature, tau=sim_params.tau
        )

    if crystal:
        integrator.couple = "none"

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


dump_frame = partial(set_dump, dump_period=None)


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


def set_harmonic_force(
    snapshot: hoomd.data.SnapshotParticleData, sim_params: SimulationParams
) -> None:
    assert sim_params.parameters.get("harmonic_force") is not None
    if sim_params.harmonic_force == 0:
        return
    HarmonicForceCompute(
        sim_params.group,
        snapshot.particles.position,
        snapshot.particles.orientation,
        sim_params.harmonic_force,
        sim_params.harmonic_force,
    )
