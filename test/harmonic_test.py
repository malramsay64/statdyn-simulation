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
from sdrun.equilibrate import equil_harmonic, minimise_configuration
from sdrun.initialise import init_from_crystal
from sdrun.params import SimulationParams
from sdrun.simrun import run_harmonic


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
            cell_dimensions=[5],
        )


def test_minimize_crystal(sim_params):
    minimize_crystal(sim_params)


def test_nvt_minimize_box(sim_params):
    """Ensure the box doesn't change size"""
    snap_init = minimize_crystal(sim_params)
    snap_final = nvt_minimize(snap_init, sim_params)
    assert snap_init.box.Lx == snap_final.box.Lx
    assert snap_init.box.Ly == snap_final.box.Ly
    assert snap_init.box.Lz == snap_final.box.Lz
    assert snap_init.box.xy == snap_final.box.xy
    assert snap_init.box.xz == snap_final.box.xz
    assert snap_init.box.yz == snap_final.box.yz


def test_run_harmonic(sim_params):
    snap_init = minimize_crystal(sim_params)
    snap_min = nvt_minimize(snap_init, sim_params)
    context = hoomd.context.initialize(sim_params.hoomd_args)
    run_harmonic(snap_min, context, sim_params)
