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

from .params import SimulationParams

logger = logging.getLogger(__name__)


def set_integrator(sim_params: SimulationParams,
                   prime_interval: int=33533,
                   crystal: bool=False,
                   create: bool=True,
                   ) -> hoomd.md.integrate.npt:
    """Hoomd integrate method."""
    if create:
        md.update.enforce2d()
        md.integrate.mode_standard(sim_params.step_size)
        if prime_interval:
            md.update.zero_momentum(period=prime_interval, phase=-1)

    if crystal:
        integrator = md.integrate.npt(
            group=sim_params.group,
            kT=sim_params.temperature,
            tau=sim_params.tau,
            P=sim_params.pressure,
            tauP=sim_params.tauP,
            rescale_all=True,
            couple='none',
        )
    else:
        integrator = md.integrate.npt(
            group=sim_params.group,
            kT=sim_params.temperature,
            tau=sim_params.tau,
            P=sim_params.pressure,
            tauP=sim_params.tauP,
        )
    return integrator


def set_dump(outfile: str,
             dump_period: int=10000,
             timestep: int=0,
             group: hoomd.group.group=None,
             ) -> hoomd.dump.gsd:
    """Initialise dumping configuration to a file."""
    if group is None:
        group = hoomd.group.rigid_center()
    # TODO on update to hoomd 2.2.0 change static to dynamic
    return hoomd.dump.gsd(
        outfile+'.gsd',
        time_step=timestep,
        period=dump_period,
        group=group,
        overwrite=True,
        static=['topology', 'attribute', 'momentum']
    )


dump_frame = partial(set_dump, dump_period=None)


def set_thermo(outfile: str,
               thermo_period: int=10000,
               rigid=True,
               ) -> None:
    """Set the thermodynamic quantities for a simulation."""
    default = ['N', 'volume', 'momentum', 'temperature', 'pressure',
               'potential_energy', 'kinetic_energy',
               'translational_kinetic_energy', 'rotational_kinetic_energy',
               'npt_thermostat_energy',
               'lx', 'ly', 'lz',
               'xy', 'xz', 'yz',
               ]
    rigid_thermo = []  # type: List[str]
    if rigid:
        rigid_thermo = ['temperature_rigid_center',
                        'pressure_rigid_center',
                        'potential_energy_rigid_center',
                        'kinetic_energy_rigid_center',
                        'translational_kinetic_energy_rigid_center',
                        'translational_ndof_rigid_center',
                        'rotational_ndof_rigid_center',
                        ]
    # TODO Set logger to hdf5 file on update to hoomd 2.2.0
    hoomd.analyze.log(
        outfile+'.log',
        quantities=default + rigid_thermo,
        period=thermo_period,
    )
