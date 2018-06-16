#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""A series of methods for the equilibration of configurations."""

import logging

import hoomd
import hoomd.md
import numpy as np

from .helper import dump_frame, set_dump, set_harmonic_force, set_integrator, set_thermo
from .initialise import initialise_snapshot, make_orthorhombic
from .params import SimulationParams, paramsContext

logger = logging.getLogger(__name__)


def equil_crystal(
    snapshot: hoomd.data.SnapshotParticleData, sim_params: SimulationParams
) -> hoomd.data.SnapshotParticleData:
    """Equilbrate crystal."""
    logger.debug("Simulation steps: %d", sim_params.num_steps)
    temp_context = hoomd.context.initialize(sim_params.hoomd_args)
    sys = initialise_snapshot(snapshot, temp_context, sim_params)

    with temp_context:
        set_integrator(
            sim_params=sim_params, prime_interval=307, simulation_type="crystal"
        )

        set_dump(
            sim_params.filename(prefix="dump"),
            dump_period=sim_params.output_interval,
            group=sim_params.group,
        )

        set_thermo(
            sim_params.filename(prefix="equil"),
            thermo_period=int(np.ceil(sim_params.output_interval / 10)),
            rigid=False,
        )

        logger.debug(
            "Running crystal equilibration for %d steps.", sim_params.num_steps
        )
        hoomd.run(sim_params.num_steps)
        logger.debug("Crystal equilibration completed")

        dump_frame(sim_params.outfile, group=sim_params.group, extension=False)

        equil_snapshot = sys.take_snapshot(all=True)
    return equil_snapshot


def equil_interface(
    snapshot: hoomd.data.SnapshotParticleData, sim_params: SimulationParams
) -> hoomd.data.SnapshotParticleData:
    """Equilbrate an interface at the desired temperature.

    This is first done by equilibrating the crystal phase, which once completed
    the liquid phase is equilibrated.
    """

    # Ensure snapshot is orthorhombic
    snapshot = make_orthorhombic(snapshot)

    logger.debug("Hoomd Arguments: %s", sim_params.hoomd_args)
    temp_context = hoomd.context.initialize(sim_params.hoomd_args)
    sys = initialise_snapshot(snapshot, temp_context, sim_params)
    logger.info("Simulation Box: %f %f %f", sys.box.Lx, sys.box.Ly, sys.box.Lz)
    with temp_context:
        logger.debug("Entering temporary context")

        interface = _interface_group(sys)
        # Set mobile group for integrator
        with paramsContext(sim_params, group=interface):
            set_integrator(sim_params=sim_params, simulation_type="interface")

        set_dump(
            sim_params.filename(prefix="dump"),
            dump_period=sim_params.output_interval,
            group=sim_params.group,
        )
        set_thermo(
            sim_params.filename(prefix="equil"),
            thermo_period=int(np.ceil(sim_params.output_interval / 10)),
            rigid=False,
        )

        hoomd.run(sim_params.num_steps)

        dump_frame(sim_params.outfile, group=sim_params.group, extension=False)
        equil_snapshot = sys.take_snapshot(all=True)
    return equil_snapshot


def equil_harmonic(
    snapshot: hoomd.data.SnapshotParticleData, sim_params: SimulationParams
) -> hoomd.data.SnapshotParticleData:
    assert sim_params.parameters.get("harmonic_force") is not None
    temp_context = hoomd.context.initialize(sim_params.hoomd_args)
    sys = initialise_snapshot(snapshot, temp_context, sim_params, minimize=True)
    with temp_context:
        set_integrator(sim_params, simulation_type="crystal", integration_method="NVT")
        set_thermo(
            sim_params.filename(prefix="thermo"),
            thermo_period=sim_params.output_interval,
        )
        set_dump(
            sim_params.filename(prefix="dump"),
            dump_period=sim_params.output_interval,
            group=sim_params.group,
        )
        set_harmonic_force(sys.take_snapshot(all=True), sim_params)
        hoomd.run(sim_params.num_steps)
        dump_frame(sim_params.outfile, group=sim_params.group)
        equil_snapshot = sys.take_snapshot(all=True)
    return equil_snapshot


def equil_liquid(
    snapshot: hoomd.data.SnapshotParticleData, sim_params: SimulationParams
) -> hoomd.data.SnapshotParticleData:
    """Equilibrate a liquid configuration."""
    temp_context = hoomd.context.initialize(sim_params.hoomd_args)
    sys = initialise_snapshot(snapshot, temp_context, sim_params)
    with temp_context:
        set_integrator(sim_params=sim_params, simulation_type="liquid")
        set_thermo(sim_params.filename("log"), thermo_period=sim_params.output_interval)
        hoomd.run(sim_params.num_steps)
        logger.debug("Outfile: %s", sim_params.outfile)
        dump_frame(sim_params.outfile, group=sim_params.group, extension=False)
        equil_snapshot = sys.take_snapshot(all=True)
    return equil_snapshot


def _interface_group(sys: hoomd.data.system_data, stationary: bool = False):
    stationary_group = hoomd.group.cuboid(
        name="stationary",
        xmin=-sys.box.Lx / 3,
        xmax=sys.box.Lx / 3,
        ymin=-sys.box.Ly / 3,
        ymax=sys.box.Ly / 3,
    )
    if stationary:
        return hoomd.group.intersection(
            "rigid_stationary", stationary_group, hoomd.group.rigid_center()
        )

    return hoomd.group.intersection(
        "rigid_mobile",
        hoomd.group.difference("mobile", hoomd.group.all(), stationary_group),
        hoomd.group.rigid_center(),
    )
