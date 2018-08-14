#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test harmonic module."""


import hoomd

from sdrun.initialise import init_from_crystal, minimize_snapshot
from sdrun.simulation import equilibrate, production


def test_minimize_box(crystal_params):
    """Ensure the box doesn't change size"""
    snap_init = init_from_crystal(crystal_params)
    snap_final = minimize_snapshot(snap_init, crystal_params, ensemble="NVE")
    assert snap_init.box.Lx == snap_final.box.Lx
    assert snap_init.box.Ly == snap_final.box.Ly
    assert snap_init.box.Lz == snap_final.box.Lz
    assert snap_init.box.xy == snap_final.box.xy
    assert snap_init.box.xz == snap_final.box.xz
    assert snap_init.box.yz == snap_final.box.yz


def test_run_harmonic(crystal_params):
    with crystal_params.temp_context(harmonic_force=0.1):
        snap_init = init_from_crystal(crystal_params)
        snap_equil = equilibrate(snap_init, crystal_params, equil_type="harmonic")
        context = hoomd.context.initialize(crystal_params.hoomd_args)
        production(
            snap_equil,
            context,
            crystal_params,
            dynamics=False,
            simulation_type="harmonic",
        )
