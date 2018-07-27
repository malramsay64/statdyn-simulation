#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Test the SimulationParams class."""

import logging
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import hoomd
import pytest
from hypothesis import given
from hypothesis.strategies import floats

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
    if sim_params.molecule.dimensions == 3:
        assert sim_params.cell_dimensions == cell_dims
    else:
        assert sim_params.cell_dimensions == (*cell_dims[:2], 1)


def test_group_setter(sim_params):
    group = "testgroup"
    sim_params.group = group
    assert sim_params.group == group


def get_group_default(sim_params):
    assert sim_params.group is None


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
    assert sim_params.output == old_output


@pytest.mark.parametrize("outfile", ["test/outfile", Path("test/outfile")])
def test_outfile(outfile, sim_params):
    old_outfile = sim_params.outfile
    with sim_params.temp_context(outfile=outfile):
        assert sim_params.outfile == Path(outfile)
    assert sim_params.outfile == old_outfile


@pytest.mark.parametrize("infile", ["test/infile", Path("test/infile")])
def test_infile(infile, sim_params):
    old_outfile = sim_params.infile
    with sim_params.temp_context(infile=infile):
        assert sim_params.infile == Path(infile)
    assert sim_params.infile == old_outfile


@given(floats(allow_nan=False, allow_infinity=False))
def test_set_moment_inertia_scale(sim_params, scaling_factor):
    old_scaling_factor = sim_params.moment_inertia_scale
    with sim_params.temp_context(moment_inertia_scale=scaling_factor):
        assert sim_params.moment_inertia_scale == scaling_factor
    assert sim_params.moment_inertia_scale == old_scaling_factor


def func(sim_params, value):
    return getattr(sim_params, value)


def test_function_passing(sim_params):
    assert sim_params.num_steps == 1000
    with sim_params.temp_context(num_steps=2000):
        assert func(sim_params, "num_steps") == 2000
        assert sim_params.num_steps == 2000

    assert func(sim_params, "num_steps") == 1000
    assert sim_params.num_steps == 1000


@given(floats(min_value=0.1, max_value=20), floats(min_value=0.1, max_value=20))
def test_filename_simple(sim_params, temperature, pressure):
    intended_fname = f"Trimer-P{pressure:.2f}-T{temperature:.2f}.gsd"
    with sim_params.temp_context(temperature=temperature, pressure=pressure):
        assert sim_params.filename().name == intended_fname


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


def filename_params():
    all_params = [
        "molecule",
        "pressure",
        "temperature",
        "moment_inertia_scale",
        "harmonic_force",
        "space_group",
        "iteration_id",
    ]
    space_groups = ["p2", "p2gg", "pg"]
    molecules = [Trimer()]
    values1 = [None, 0, 0.1, 1]
    values2 = [0, 0.1, 1]
    for space_group, molecule, value1, value2 in product(
        space_groups, molecules, values1, values2
    ):
        params = {}
        for param in all_params:
            if param == "molecule":
                params["molecule"] = molecule
            elif param == "space_group":
                params["space_group"] = space_group
            elif param in ["temperature", "pressure"]:
                params[param] = value2
            elif param in ["iteration_id"]:
                params[param] = int(value1) if value1 is not None else None
            else:
                params[param] = value1
        yield params


def get_filename_prefix(key):
    prefixes = {
        "temperature": "T",
        "pressure": "P",
        "moment_inertia_scale": "I",
        "harmonic_force": "K",
        "iteration_id": "ID",
    }
    return prefixes.get(key, "")


@pytest.mark.parametrize("params", filename_params())
def test_filename(sim_params, params):
    with sim_params.temp_context(**params):
        fname = sim_params.filename().stem
    for key, value in params.items():
        if value is not None:
            prefix = get_filename_prefix(key)
            assert isinstance(prefix, str)
            logger.debug("Prefix: %s, Value: %s", prefix, value)
            if isinstance(value, (int, float)):
                if key == "iteration_id":
                    assert f"{prefix}{value:d}" in fname
                else:
                    assert f"{prefix}{value:.2f}" in fname
            else:
                assert f"{prefix}{value}" in fname


def test_get_group(mol_params):
    with hoomd.context.initialize(mol_params.hoomd_args):
        hoomd.init.create_lattice(hoomd.lattice.sq(1), 1)
        group = mol_params.get_group()
        assert group is not None
