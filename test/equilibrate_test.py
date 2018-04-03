#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Test the equilibrate module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from sdrun.crystals import CRYSTAL_FUNCS
from sdrun.equilibrate import equil_crystal, equil_interface, equil_liquid
from sdrun.initialise import init_from_crystal
from sdrun.params import SimulationParams


@pytest.fixture(params=CRYSTAL_FUNCS.values(), ids=CRYSTAL_FUNCS.keys())
def sim_params(request):
    with TemporaryDirectory() as tmp_dir:
        yield SimulationParams(
            temperature=0.4,
            num_steps=100,
            crystal=request.param(),
            output=Path(tmp_dir),
            cell_dimensions=(10, 12),
            outfile=Path(tmp_dir) / 'out.gsd',
        )


def test_equil_crystal(sim_params):
    equil_crystal(init_from_crystal(sim_params), sim_params)
    assert True


def test_equil_interface(sim_params):
    equil_interface(init_from_crystal(sim_params), sim_params)
    assert True


def test_equil_liquid(sim_params):
    equil_liquid(init_from_crystal(sim_params), sim_params)
    assert True
