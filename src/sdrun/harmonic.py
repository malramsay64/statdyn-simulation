#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Implementation of the harmonic potential.

This is an implementation of using the harmonic potential for the calculation of the free energy of
a crystal.

1. Create equilibrated crystal at NPT
2. Minimise energy as NVT simulation
3. Get positions and orientations from the minimized simulation
4. Run simulations with harmonic potential

"""

import logging

import hoomd
import hoomd.md
from hoomd.harmonic_force import HarmonicForceCompute

from .equilibrate import equil_crystal
from .helper import dump_frame, set_dump, set_integrator, set_thermo
from .initialise import init_from_none, initialise_snapshot
from .params import SimulationParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def nvt_minimize(snapshot, sim_params):
    temp_context = hoomd.context.initialize(sim_params.hoomd_args)
    molecule = sim_params.molecule
    with temp_context:
        sys = hoomd.init.read_snapshot(snapshot)
        molecule.define_potential()
        molecule.define_dimensions()
        rigid = molecule.define_rigid()
        if rigid:
            rigid.check_initialization()
            group = hoomd.group.rigid_center()
        else:
            group = hoomd.group.all()
        logger.debug("Minimizing energy")
        fire = hoomd.md.integrate.mode_minimize_fire(0.001)
        nvt = hoomd.md.integrate.nvt(group=group, kT=sim_params.temperature, tau=1)
        num_steps = 0
        while not fire.has_converged():
            hoomd.run(100)
            num_steps += 100
            if num_steps > 10_000:
                break

        nvt.disable()
        logger.debug("Energy Minimized in %s steps", num_steps)
        return sys.take_snapshot()


def minimize_crystal(sim_params: SimulationParams):
    snap = init_from_none(sim_params)
    equil_snap = equil_crystal(snap, sim_params)
    min_snap = nvt_minimize(equil_snap, sim_params)
    return min_snap


def run_harmonic(
    snapshot: hoomd.data.SnapshotParticleData,
    context: hoomd.context.SimulationContext,
    sim_params: SimulationParams,
) -> None:
    """Initialise and run a hoomd npt simulation with harmonic pinning

    Args:
        snapshot (class:`hoomd.data.snapshot`): Hoomd snapshot object
        context:
        sim_params:

    """
    nmols = min(max(snapshot.particles.body) + 1, snapshot.particles.N)
    intended_positions = snapshot.particles.position[:nmols]
    intended_orientations = snapshot.particles.orientation[:nmols]
    with context:
        sys = initialise_snapshot(
            snapshot=snapshot, context=context, molecule=sim_params.molecule
        )

        hoomd.md.integrate.mode_standard(sim_params.step_size)
        if sim_params.molecule.dimensions == 2:
            hoomd.md.update.enforce2d()
        hoomd.md.integrate.nvt(
            group=sim_params.group, kT=sim_params.temperature, tau=sim_params.tau
        )

        set_thermo(
            sim_params.filename(prefix="thermo"),
            thermo_period=sim_params.output_interval,
        )
        set_dump(
            sim_params.filename(prefix="dump"),
            dump_period=sim_params.output_interval,
            group=sim_params.group,
        )
        HarmonicForceCompute(
            sim_params.group,
            intended_positions,
            intended_orientations,
            getattr(sim_params, "force_constant", 1),
            getattr(sim_params, "force_constant", 1),
        )
        hoomd.run(sim_params.num_steps)
        dump_frame(sim_params.filename(), group=sim_params.group)
