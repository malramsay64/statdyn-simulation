#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Module for setting up and running a hoomd simulation."""

import logging

import hoomd

from .helper import (
    SimulationParams,
    dump_frame,
    set_dump,
    set_harmonic_force,
    set_integrator,
    set_thermo,
)
from .initialise import initialise_snapshot
from .StepSize import GenerateStepSeries

logger = logging.getLogger(__name__)


def production(
    snapshot: hoomd.data.SnapshotParticleData,
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

        if simulation_type == "harmonic":
            set_integrator(
                sim_params, simulation_type="crystal", integration_method="NVT"
            )
            set_harmonic_force(snapshot, sim_params)
        else:
            set_integrator(sim_params, simulation_type="liquid")

        set_thermo(
            sim_params.filename(prefix="thermo"),
            thermo_period=sim_params.output_interval,
        )
        set_dump(
            sim_params.filename(prefix="dump"),
            dump_period=sim_params.output_interval,
            group=sim_params.group,
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
            dumpfile = dump_frame(
                sim_params.filename(prefix="trajectory"), group=sim_params.group
            )
            for curr_step in iterator:
                hoomd.run_upto(curr_step, quiet=True)
                dumpfile.write_restart()
        else:
            hoomd.run(sim_params.num_steps)
        dump_frame(sim_params.filename(), group=sim_params.group)
