#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Test the SimulationParams class."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from sdrun.crystals import CRYSTAL_FUNCS, CubicSphere, SquareCircle, TrimerP2
from sdrun.molecules import MOLECULE_DICT, Dimer, Disc, Molecule, Sphere, Trimer
from sdrun.params import SimulationParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
MOLECULE_LIST = [Molecule, Sphere, Trimer, Dimer, Disc, None]


@pytest.fixture(scope="function")
def sim_params():
    return SimulationParams(num_steps=1000, temperature=1.0)


@pytest.fixture(params=MOLECULE_DICT.values(), ids=MOLECULE_DICT.keys())
def mol_params(request):
    return SimulationParams(num_steps=1000, temperature=1.0, molecule=request.param)


@pytest.fixture(params=MOLECULE_DICT.values(), ids=MOLECULE_DICT.keys())
def molecules(request):
    return request.param


@pytest.fixture(params=CRYSTAL_FUNCS.values(), ids=CRYSTAL_FUNCS.keys())
def crystal_params(request):
    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "output"
        output_dir.mkdir(exist_ok=True)
        yield SimulationParams(
            temperature=1.0,
            pressure=13.5,
            num_steps=1000,
            crystal=request.param(),
            output=output_dir,
        )


@pytest.fixture(params=MOLECULE_DICT.values(), ids=MOLECULE_DICT.keys())
def mol_params(request):
    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "output"
        output_dir.mkdir(exist_ok=True)
        yield SimulationParams(
            temperature=1.0,
            pressure=13.5,
            num_steps=1000,
            output=output_dir,
            molecule=request.param(),
        )


def test_molecule(sim_params, molecules):
    init_mol = sim_params.molecule
    with sim_params.temp_context(molecule=molecules):
        assert sim_params.molecule == molecules
    assert sim_params.molecule == init_mol


def test_default_molecule(sim_params):
    assert sim_params.molecule == Trimer()


def test_molecule_setter(sim_params, molecules):
    sim_params.molecule = molecules
    assert sim_params.molecule == molecules


@pytest.mark.parametrize("cell_len", range(1, 10))
def test_cell_dimensions_setter(sim_params, cell_len):
    cell_dims = (cell_len, cell_len, cell_len)
    sim_params.cell_dimensions = cell_dims
    assert sim_params.cell_dimensions == cell_dims


def test_group_setter(sim_params):
    group = "testgroup"
    sim_params.group = group
    assert sim_params.group == group


def test_mol_crys(sim_params):
    crys = SquareCircle()
    init_mol = sim_params.molecule
    with sim_params.temp_context(molecule=None, crystal=crys):
        assert sim_params.molecule == crys.molecule
    assert sim_params.molecule == init_mol


@pytest.mark.parametrize("output", ["test/data", Path("test/data")])
def test_output(sim_params, output):
    old_output = sim_params.output
    with sim_params.temp_context(output=output):
        assert sim_params.output == Path(output)
    assert sim_params.outdir == old_output


@pytest.mark.parametrize("outdir", ["test/output", Path("test/output")])
def test_outdir(outdir, sim_params):
    old_outdir = sim_params.outdir
    with sim_params.temp_context(outdir=outdir):
        assert sim_params.outdir == Path(outdir)
    assert sim_params.outdir == old_outdir


def func(sim_params, value):
    return getattr(sim_params, value)


def test_function_passing(sim_params):
    assert sim_params.num_steps == 1000
    with sim_params.temp_context(num_steps=2000):
        assert func(sim_params, "num_steps") == 2000
        assert sim_params.num_steps == 2000

    assert func(sim_params, "num_steps") == 1000
    assert sim_params.num_steps == 1000


def test_cell_dimensions(sim_params):
    with sim_params.temp_context(crystal=TrimerP2(), cell_dimensions=10) as temp_params:
        assert len(temp_params.cell_dimensions) == 3
        assert temp_params.cell_dimensions == (10, 10, 1)
    with sim_params.temp_context(
        crystal=CubicSphere(), cell_dimensions=10
    ) as temp_params:
        assert len(sim_params.cell_dimensions) == 3
        assert sim_params.cell_dimensions == (10, 10, 10)
    with sim_params.temp_context(crystal=CubicSphere(), cell_dimensions=[10, 10]):
        assert len(sim_params.cell_dimensions) == 3
        assert sim_params.cell_dimensions == (10, 10, 1)


def test_crystal_cell_dimensions(crystal_params):
    with crystal_params.temp_context(cell_dimensions=10):
        assert len(crystal_params.cell_dimensions) == 3
    with crystal_params.temp_context(cell_dimensions=[10, 10]):
        assert len(crystal_params.cell_dimensions) == 3
    with crystal_params.temp_context(cell_dimensions=[10, 10, 10]):
        assert len(crystal_params.cell_dimensions) == 3
