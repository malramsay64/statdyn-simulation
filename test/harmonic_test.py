#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test harmonic module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import hoomd
import pytest

from sdrun.crystals import CRYSTAL_FUNCS
from sdrun.equilibrate import equilibrate
from sdrun.initialise import init_from_crystal, minimize_snapshot
from sdrun.params import SimulationParams
from sdrun.simrun import production


@pytest.fixture(params=CRYSTAL_FUNCS.values(), ids=CRYSTAL_FUNCS.keys())
def sim_params(request):
    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "output"
        output_dir.mkdir(exist_ok=True)
        yield SimulationParams(
            temperature=0.4,
            num_steps=100,
            output=output_dir,
            outfile=output_dir / "test.gsd",
            crystal=request.param(),
            cell_dimensions=5,
            harmonic_force=1,
        )


def test_minimize_box(sim_params):
    """Ensure the box doesn't change size"""
    snap_init = init_from_crystal(sim_params)
    snap_final = minimize_snapshot(snap_init, sim_params, ensemble="NVE")
    assert snap_init.box.Lx == snap_final.box.Lx
    assert snap_init.box.Ly == snap_final.box.Ly
    assert snap_init.box.Lz == snap_final.box.Lz
    assert snap_init.box.xy == snap_final.box.xy
    assert snap_init.box.xz == snap_final.box.xz
    assert snap_init.box.yz == snap_final.box.yz


def test_run_harmonic(sim_params):
    snap_init = init_from_crystal(sim_params)
    snap_equil = equilibrate(snap_init, sim_params, equil_type="harmonic")
    context = hoomd.context.initialize(sim_params.hoomd_args)
    production(
        snap_equil, context, sim_params, dynamics=False, simulation_type="harmonic"
    )
