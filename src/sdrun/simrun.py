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
import numpy as np

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


def run_npt(
    snapshot: hoomd.data.SnapshotParticleData,
    context: hoomd.context.SimulationContext,
    sim_params: SimulationParams,
    dynamics: bool = True,
) -> None:
    """Initialise and run a hoomd npt simulation.

    Args:
        snapshot (class:`hoomd.data.snapshot`): Hoomd snapshot object
        context:
        sim_params:


    """
    with context:
        sys = initialise_snapshot(snapshot, context, sim_params)
        logger.debug("Run metadata: %s", sys.get_metadata())
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


def run_harmonic(
    snapshot: hoomd.data.SnapshotParticleData,
    context: hoomd.context.SimulationContext,
    sim_params: SimulationParams,
) -> None:
    """Initialise and run a simulation with a harmonic pinning potential."""
    with context:
        initialise_snapshot(snapshot, context, sim_params)
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
        set_harmonic_force(snapshot, sim_params)
        hoomd.run(sim_params.num_steps)
        dump_frame(sim_params.filename(), group=sim_params.group)


def read_snapshot(
    context: hoomd.context.SimulationContext, fname: str, rand: bool = False
) -> hoomd.data.SnapshotParticleData:
    """Read a hoomd snapshot from a hoomd gsd file.

    Args:
    fname (string): Filename of GSD file to read in
    rand (bool): Whether to randomise the momenta of all the particles

    Returns:
    class:`hoomd.data.SnapshotParticleData`: Hoomd snapshot

    """
    with context:
        snapshot = hoomd.data.gsd_snapshot(fname)
        if rand:
            nbodies = snapshot.particles.body.max() + 1
            np.random.shuffle(snapshot.particles.velocity[:nbodies])
            np.random.shuffle(snapshot.particles.angmom[:nbodies])
        return snapshot
