#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Module for the running of molecular dynamics simulations

This module contains functions for running simulations, with a range of
functions for different simulation types.

"""

import logging

import hoomd
import hoomd.md
import numpy as np
from hoomd.data import SnapshotParticleData as Snapshot

from .initialise import (
    init_from_crystal,
    initialise_snapshot,
    make_orthorhombic,
    thermalise,
)
from .params import SimulationParams
from .StepSize import GenerateStepSeries
from .util import dump_frame, get_group, set_dump, set_integrator, set_thermo

logger = logging.getLogger(__name__)


def equilibrate(
    snapshot: Snapshot,
    sim_params: SimulationParams,
    equil_type: str = "liquid",
    thermalisation: bool = False,
) -> Snapshot:
    """Run an equilibration simulation.

    This will configure and run an equilibration simulation of the type specified in
    equil_type. Each of these equilibration types has some slight variations although
    they all follow the same premise of running a simulation with the goal of reaching
    an equilibrated state.

    Args:
        snapshot: The initial snapshot to start the simulation.
        sim_params: The simulation parameters
        equil_type: The type of equilibration to undertake. This is one
            of (liquid|crystal|interface).
        thermalisation: When `True`, an initial velocity and angular momenta to each
            molecule in the simulation, with the values being pulled from the appropriate
            Maxwell-Bolzmann distribution for the temperature.

    Simulations of type liquid or interface will be forced into an orthorhombic shape by
    setting the new orthorhombic box shape and moving particles through the new periodic
    boundary conditions. In the image below, the initial configuration is the tilted
    box, with the vertical bars being the simulation box. Particles outside the new box
    are wrapped into the missing regions on the opposite side.

    ::
           ____________________
          | /               | /
          |/                |/
          /                 /
         /|                /|
        /_|_______________/_|

    The only difference between simulations of type `"liquid"` and `"interface"`, is
    that the interface simulations will only be integrating the motion of a subset of
    molecules, with the central 2/3 of particles remaining stationary.

    For the simulation type `"crystal"`, the momentum is zeroed more often, every 307
    steps instead of 33533 for the liquid and interface simulations. Additionally, to
    allow proper and complete relaxation of the crystal structure, each side of the
    simulation cell is able to move independently and the simulation cell is also
    permitted to tilt.

    """
    # Check for required parameters, failing early if missing
    if equil_type not in ["liquid", "crystal", "interface"]:
        raise ValueError(
            f"equil_type needs to be one of (liquid|crystal|interface), found {equil_type}"
        )

    if sim_params.hoomd_args is None:
        raise ValueError(
            "The hoomd_args cannot be None, no arguments is an empty string."
            "Found {hoomd_args}"
        )

    if sim_params.num_steps is None or sim_params.num_steps < 0:
        raise ValueError(
            "The number of steps has to be a positive number, found {sim_params.num_steps}"
        )

    if sim_params.output_interval is None or sim_params.output_interval < 0:
        raise ValueError(
            "The number of steps between output configurations needs to be a positive integer,"
            "found {sim_params.output_interval}"
        )

    if sim_params.outfile is None:
        raise ValueError("The outfile needs to be defined, found None")

    # Ensure orthorhombic liquid and interface
    if equil_type in ["liquid", "interface"]:
        snapshot = make_orthorhombic(snapshot)

    # Randomise momenta such that the temperature is sim_params.temperature
    if thermalisation:
        snapshot = thermalise(snapshot, sim_params)

    logger.debug("Snapshot box size: %s", snapshot.box)

    temp_context = hoomd.context.initialize(sim_params.hoomd_args)
    sys = initialise_snapshot(snapshot, temp_context, sim_params)

    with temp_context:
        prime_interval = 33533
        simulation_type = equil_type
        group = get_group(sys, sim_params)
        assert group is not None
        integration_method = "NPT"

        if equil_type == "crystal":
            prime_interval = 307

        # Set mobile group for integrator
        if equil_type == "interface":
            integrated_group = get_group(sys, sim_params, interface=True)
            assert integrated_group is not None
        else:
            integrated_group = group

        set_integrator(
            sim_params,
            integrated_group,
            prime_interval,
            simulation_type,
            integration_method,
        )

        set_dump(
            group,
            sim_params.filename(prefix="dump"),
            dump_period=sim_params.output_interval,
        )

        set_thermo(
            sim_params.filename(prefix="equil"),
            thermo_period=int(np.ceil(sim_params.output_interval / 10)),
            rigid=False,
        )

        logger.debug(
            "Running %s equilibration for %d steps.", equil_type, sim_params.num_steps
        )
        logger.debug(
            "Simulation box size: %f, %f, %f", sys.box.Lx, sys.box.Ly, sys.box.Lz
        )
        logger.debug("Simulation box dimensions: %d", sys.box.dimensions)
        logger.debug("Simulation num particles: %d", sys.particles.pdata.getN())

        hoomd.run(sim_params.num_steps)
        logger.debug("Equilibration completed")

        dump_frame(group, sim_params.outfile, extension=False)

        equil_snapshot = sys.take_snapshot()
    return equil_snapshot


def create_interface(sim_params: SimulationParams) -> Snapshot:
    """Helper for the creation of a liquid--crystal interface.

    The creation of a liquid--crystal interface has a number of different steps. This is a helper
    function which goes through those steps with the intent of having melted the liquid component
    and having high temperature liquid--crystal interface.
    """
    if sim_params.init_temp is None or sim_params.init_temp <= 0:
        raise ValueError(
            "The init_temp parameter needs to be set to a positive value"
            "to create an interface, found {sim_params.init_temp}"
        )

    if sim_params.num_steps is None or sim_params.num_steps <= 0:
        raise ValueError(
            "The num_steps parameter needs to be set to a positive integer"
            "to create an interface, found {sim_params.num_steps}"
        )

    # Initialisation typically requires fewer steps than melting
    # 100 initialisation steps is a 'magic number' which works for the p2 crystal. I don't know why
    # this works however it is a TODO item.
    init_steps = min(sim_params.num_steps, 100)

    # Initialise at low init_temp
    with sim_params.temp_context(
        temperature=sim_params.init_temp, num_steps=init_steps, init_temp=None
    ):
        snapshot = init_from_crystal(sim_params)
        # Find NPT minimum of crystal
        snapshot = equilibrate(snapshot, sim_params, equil_type="crystal")
    # Equilibrate interface to temperature with intent to melt the interface
    with sim_params.temp_context(init_temp=None):
        snapshot = equilibrate(
            snapshot, sim_params, equil_type="interface", thermalisation=True
        )
    return snapshot


def production(
    snapshot: Snapshot,
    context: hoomd.context.SimulationContext,
    sim_params: SimulationParams,
    dynamics: bool = True,
    simulation_type: str = "liquid",
) -> None:
    """Initialise and run a hoomd npt simulation for data collection.

    This is a utility function to run a simulation designed for the collection
    of data. This is particularly true when the `dynamics` argument is `True`,
    with a separate sequence of data points on an exponential sequence being
    collected.

    Args:
        snapshot: The configuration from which to start the production simulation
        context: Simulation context in which to run the simulation.
        sim_params: The parameters of the simulation which are to be set.
        dynamics: Whether to output an exponential series of configurations
            to make calculating dynamics properties easier.
        simulation_type: The type of simulation to run. Currently only `"liquid"`
            is supported.

    """
    if sim_params.num_steps is None or sim_params.num_steps < 0:
        raise ValueError(
            "The number of steps has to be a positive number, found {sim_params.num_steps}"
        )

    if sim_params.output_interval is None or sim_params.output_interval < 0:
        raise ValueError(
            "The number of steps between output configurations needs to be a positive integer,"
            "found {sim_params.output_interval}"
        )

    if not isinstance(context, hoomd.context.SimulationContext):
        raise ValueError(
            "The context needs to be a `hoomd.context.SimulationContext' instance",
            f"found {type(context)}",
        )

    if simulation_type not in ["liquid"]:
        raise ValueError(
            "Supported simulation types are (liquid)," "found {simulation_type}"
        )

    sys = initialise_snapshot(snapshot, context, sim_params)
    with context:
        logger.debug("Run metadata: %s", sys.get_metadata())

        group = get_group(sys, sim_params)

        set_integrator(sim_params, group, simulation_type="liquid")

        set_thermo(
            sim_params.filename(prefix="thermo"),
            thermo_period=sim_params.output_interval,
            rigid=sim_params.molecule.rigid,
        )
        set_dump(
            group,
            sim_params.filename(prefix="dump"),
            dump_period=sim_params.output_interval,
        )

        if dynamics:
            iterator = GenerateStepSeries(
                sim_params.num_steps,
                num_linear=sim_params.num_linear,
                max_gen=sim_params.max_gen,
                gen_steps=sim_params.gen_steps,
            )
            # Zeroth step
            curr_step = iterator.next()
            assert curr_step == 0
            dumpfile = dump_frame(group, sim_params.filename(prefix="trajectory"))
            for curr_step in iterator:
                hoomd.run_upto(curr_step, quiet=True)
                dumpfile.write_restart()
        else:
            hoomd.run(sim_params.num_steps)
        dump_frame(group, sim_params.filename())
