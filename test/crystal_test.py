#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Testing the crystal class of sdrun."""

from pathlib import Path
from tempfile import TemporaryDirectory

import hoomd
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sdrun.crystals import (
    Crystal,
    CubicSphere,
    SquareCircle,
    TrimerP2,
    TrimerP2gg,
    TrimerPg,
    _calc_shift,
)
from sdrun.params import SimulationParams

TEST_CLASSES = [Crystal, TrimerP2, TrimerP2gg, TrimerPg, SquareCircle, CubicSphere]


@pytest.fixture(params=TEST_CLASSES)
def crys_class(request):
    return request.param()


@pytest.fixture(params=TEST_CLASSES)
def sim_params(request):
    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "output"
        output_dir.mkdir(exist_ok=True)
        yield SimulationParams(
            temperature=0.4,
            num_steps=100,
            output=output_dir,
            crystal=request.param(),
            cell_dimensions=(32, 40),
        )


def test_shift_position(molecule):
    orientations = np.zeros(1)

    shift = _calc_shift(orientations, molecule)
    assert np.all(shift == molecule.get_relative_positions()[0])


def test_init(crys_class):
    """Check the initialisation of the class."""
    assert isinstance(crys_class, Crystal)
    assert crys_class.dimensions in [2, 3]


@pytest.mark.parametrize("crys", TEST_CLASSES)
@pytest.mark.parametrize("param", ["cell_matrix", "positions", "_orientations"])
def test_seperability(crys, param):
    crystal = crys()
    crystal_copy = crys()
    values = getattr(crystal, param)
    values += 1
    assert crystal_copy != crystal


@pytest.mark.parametrize("crys", TEST_CLASSES)
def test_equality(crys):
    crystal = crys()
    assert crystal == crystal
    crystal_copy = crys()
    assert crystal_copy == crystal
    crystal_copy._orientations += 1
    assert crystal_copy != crystal

    class crys_sub(crys):
        def __init__(self):
            super().__init__()

    subclass = crys_sub()

    assert crystal != subclass


def test_get_orientations(crys_class):
    """Test the orientation is returned as quaternions."""
    orient = crys_class.get_orientations()
    assert orient.dtype == np.float32
    # Output is in quaternions
    assert orient.shape[1] == 4
    # Quaternions are normalised
    # This uses the np.allclose function since I am comparing to a
    # value instead of an array
    assert np.allclose(np.linalg.norm(orient, axis=1), 1)


def test_get_unitcell(crys_class):
    """Test the return type is correct."""
    unit_cell = crys_class.get_unitcell()
    assert isinstance(unit_cell, hoomd.lattice.unitcell)
    assert np.all(unit_cell.mass == crys_class.molecule.mass)
    assert unit_cell.dimensions == crys_class.dimensions
    assert np.all(unit_cell.a1 == crys_class.cell_matrix[0])
    assert np.all(unit_cell.a2 == crys_class.cell_matrix[1])
    assert np.all(unit_cell.a3 == crys_class.cell_matrix[2])
    if crys_class.molecule.num_particles > 1:
        assert unit_cell.type_name == ["R"] * crys_class.get_num_molecules()


def test_compute_volume(crys_class):
    """Test the return type of the volume computation."""
    assert isinstance(crys_class.compute_volume(), float)
    if type(crys_class) is Crystal:
        assert crys_class.compute_volume() == 1
    elif type(crys_class) is CubicSphere:
        assert crys_class.compute_volume() == 8
    elif type(crys_class) is CubicSphere:
        assert crys_class.compute_volume() == 4


def test_relative_positions(crys_class):
    """Check the relative positions function return correctly shaped matrix."""
    rel_pos = crys_class.get_relative_positions()
    assert rel_pos.shape == np.array(crys_class.positions).shape
    assert np.all(0 <= rel_pos)
    assert np.all(rel_pos <= 1)


def test_get_matrix(crys_class):
    matrix = crys_class.cell_matrix
    assert matrix.shape == (3, 3)
    for i in range(3):
        assert matrix[i, i] > 0
    if crys_class.dimensions == 2:
        assert matrix[2, 2] == 1


def test_matrix_values(crys_class):
    matrix = crys_class.cell_matrix
    if crys_class.dimensions == 2:
        assert np.all(matrix[:2, 2] == 0)
        assert np.all(matrix[2, :2] == 0)
        assert np.all(matrix[2, 2] == 1)

    if type(crys_class) is SquareCircle:
        assert matrix[0, 0] == 2
        assert matrix[1, 1] == 2


def test_positions(crys_class):
    positions = crys_class.positions
    cell_lengths = np.sum(np.identity(3) @ crys_class.cell_matrix, axis=0)

    assert positions.shape == (crys_class.num_molecules, 3)
    assert np.all(positions >= 0)
    assert np.all(positions < cell_lengths)


@pytest.mark.xfail()
def test_trimerp2_positions():
    positions = TrimerP2().positions

    # Check against manually computed positions
    manual_large_positions = np.array([[1.3476, 0.816, 0.0], [3.1024, 1.734, 0.0]])
    assert_allclose(manual_large_positions, positions)
