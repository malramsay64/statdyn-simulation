#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""A collection of fixtures which can be used for all tests."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from sdrun import SimulationParams
from sdrun.crystals import CRYSTAL_FUNCS
from sdrun.molecules import MOLECULE_DICT
from sdrun import init_from_crystal
import hoomd


@pytest.fixture(params=MOLECULE_DICT.values(), ids=MOLECULE_DICT.keys())
def molecule(request):
    with hoomd.context.initialize(""):
        yield request.param()


@pytest.fixture(params=MOLECULE_DICT.values(), ids=MOLECULE_DICT.keys())
def mol_params(request):
    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "output"
        output_dir.mkdir(exist_ok=True)
        yield SimulationParams(
            temperature=0.4,
            pressure=1.0,
            num_steps=100,
            molecule=request.param(),
            output=output_dir,
            cell_dimensions=(10, 12, 10),
            outfile=output_dir / "test.gsd",
            hoomd_args="--mode=cpu --notice-level=0",
        )


@pytest.fixture(params=CRYSTAL_FUNCS.values(), ids=CRYSTAL_FUNCS.keys())
def crystal_params(request):
    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "output"
        output_dir.mkdir(exist_ok=True)
        yield SimulationParams(
            temperature=0.4,
            pressure=1.0,
            num_steps=100,
            crystal=request.param(),
            output=output_dir,
            cell_dimensions=(10, 12, 10),
            outfile=output_dir / "test.gsd",
            hoomd_args="--mode=cpu --notice-level=0",
        )


@pytest.fixture(params=CRYSTAL_FUNCS.values(), ids=CRYSTAL_FUNCS.keys())
def snapshot(request):
    """Test the initialisation of all crystals."""
    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "output"
        output_dir.mkdir(exist_ok=True)
        sim_params = SimulationParams(
            temperature=0.4,
            pressure=1.0,
            num_steps=100,
            crystal=request.param(),
            output=output_dir,
            cell_dimensions=(10, 12, 10),
            outfile=output_dir / "test.gsd",
            hoomd_args="--mode=cpu --notice-level=0",
        )
        yield init_from_crystal(sim_params)
