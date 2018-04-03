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

from sdrun import equilibrate, initialise, params
from sdrun.crystals import TrimerPg


@pytest.fixture
def sim_params():
    with TemporaryDirectory() as tmp_dir:
        return params.SimulationParams(
            temperature=0.4,
            num_steps=100,
            crystal=TrimerPg(),
            outfile_path=tmp_dir,
            cell_dimensions=(10, 10),
            outfile=Path(tmp_dir) / 'out.gsd',
        )


def init_frame(sim_params):
    return initialise.init_from_crystal(sim_params)


def test_equil_crystal(sim_params):
    equilibrate.equil_crystal(init_frame(), sim_params)
    assert True


def test_equil_interface(sim_params):
    equilibrate.equil_interface(init_frame(), sim_params)
    assert True


def test_equil_liquid(sim_params):
    equilibrate.equil_liquid(init_frame(), sim_params)
    assert True
