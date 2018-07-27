#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""
"""

import logging

import hoomd
import hoomd.md
import numpy as np
from hoomd.data import SnapshotParticleData as Snapshot, system_data as System
from hoomd.group import group as Group

from .initialise import init_from_crystal, initialise_snapshot, make_orthorhombic
from .params import SimulationParams
from .StepSize import GenerateStepSeries
from .util import dump_frame, set_dump, set_harmonic_force, set_integrator, set_thermo

logger = logging.getLogger(__name__)


def equilibrate(
    snapshot: Snapshot, sim_params: SimulationParams, equil_type: str = "liquid"
) -> Snapshot:
    """Run an equilibration simulation.

    This will configure and run an equilibration simulation of the type specified in equil_type.
    Each of these equilibration types has some slight variations althought they all follow the
    same premise of running a simulation with the goal of reaching an equilibrated state.

    Args:
        snapshot: The initial snapshot to start the simulation.
        sim_params: The simulation paramters
        equil_type: The type of equilibration to undertake. This is one of `["liquid", "crystal",
        "interface", "harmonic"]`.

    """
    # Check for required paramters, faililng early if missing
    assert equil_type in ["liquid", "crystal", "interface", "harmonic"]
    assert sim_params.hoomd_args is not None
    assert sim_params.output_interval is not None
    assert sim_params.num_steps is not None
    assert sim_params.num_steps > 0

    # Ensure orthorhombic liquid and interface
    if equil_type in ["liquid", "interface"]:
        snapshot = make_orthorhombic(snapshot)

    logger.debug("Snapshot box size: %s", snapshot.box)

    temp_context = hoomd.context.initialize(sim_params.hoomd_args)
    sys = initialise_snapshot(snapshot, temp_context, sim_params)

    with temp_context:
        prime_interval = 33533
        simulation_type = equil_type
        group = get_group(sys, sim_params)
        assert group is not None
        integration_method = "NPT"

        if equil_type == "harmonic":
            assert sim_params.harmonic_force is not None
            simulation_type = "liquid"
            integration_method = "NVT"
        if equil_type == "crystal":
            prime_interval = 307
        if equil_type == "interface":
            group = get_group(sys, sim_params, interface=True)
            assert group is not None

        # Set mobile group for integrator
        set_integrator(
            sim_params, group, prime_interval, simulation_type, integration_method
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

        # TODO run a check for equilibration and emit a warning if the simulation is not
        # equilibrated properly. This will be through monitoing the thermodynamics.

        dump_frame(group, sim_params.outfile, extension=False)

        equil_snapshot = sys.take_snapshot(all=True)
    return equil_snapshot


def create_interface(sim_params: SimulationParams) -> Snapshot:
    """Helper for the creation of a liquid--crystal interface.

    The creation of a liquid--crystal interface has a number of different steps. This is a helper
    function which goes through those steps with the intent of having melted the liquid component
    and having high temperature liquid--crystal interface.
    """
    assert sim_params.init_temp is not None
    assert sim_params.init_temp > 0
    assert sim_params.num_steps is not None
    assert sim_params.num_steps > 0

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
    snapshot = equilibrate(snapshot, sim_params, equil_type="interface")
    return snapshot


def get_group(
    sys: System, sim_params: SimulationParams, interface: bool = False
) -> Group:
    if sim_params.molecule.num_particles > 1:
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


def production(
    snapshot: Snapshot,
    context: hoomd.context.SimulationContext,
    sim_params: SimulationParams,
    dynamics: bool = True,
    simulation_type: str = "liquid",
) -> None:
    """Initialise and run a hoomd npt simulation.

    Args:
        snapshot (class:`hoomd.data.snapshot`): Hoomd snapshot object
        context:
        sim_params:


    """
    assert sim_params.num_steps is not None
    assert sim_params.output_interval is not None
    assert isinstance(context, hoomd.context.SimulationContext)
    assert simulation_type in ["liquid", "harmonic"]

    with context:
        sys = initialise_snapshot(snapshot, context, sim_params)
        logger.debug("Run metadata: %s", sys.get_metadata())

        group = get_group(sys, sim_params)

        if simulation_type == "harmonic":
            set_integrator(
                sim_params, group, simulation_type="crystal", integration_method="NVT"
            )
            set_harmonic_force(snapshot, sim_params, group)
        else:
            set_integrator(sim_params, group, simulation_type="liquid")

        set_thermo(
            sim_params.filename(prefix="thermo"),
            thermo_period=sim_params.output_interval,
        )
        set_dump(
            group,
            sim_params.filename(prefix="dump"),
            dump_period=sim_params.output_interval,
        )

        if dynamics:
            iterator = GenerateStepSeries(
                sim_params.num_steps,
                num_linear=100,
                max_gen=sim_params.max_gen,
                gen_steps=20000,
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
