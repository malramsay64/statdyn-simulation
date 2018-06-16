#! /usr/bin/env python
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

from sdrun import crystals
from sdrun.params import SimulationParams

TEST_CLASSES = [
    crystals.Crystal,
    crystals.CrysTrimer,
    crystals.TrimerP2,
    crystals.TrimerP2gg,
    crystals.TrimerPg,
    crystals.SquareCircle,
    crystals.CubicSphere,
]


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


def test_init(crys_class):
    """Check the initialisation of the class."""
    assert isinstance(crys_class, crystals.Crystal)


def test_get_orientations(crys_class):
    """Test the orientation is returned as quaternions."""
    orient = crys_class.get_orientations()
    assert orient.dtype == np.float32
    # Output is in quaternions
    assert orient.shape[1] == 4
    # Quaterninons as normalised
    assert np.allclose(np.linalg.norm(orient, axis=1), 1)


def test_get_unitcell(crys_class):
    """Test the return type is correct."""
    assert isinstance(crys_class.get_unitcell(), hoomd.lattice.unitcell)


def test_compute_volume(crys_class):
    """Test the return type of the volume computation."""
    assert isinstance(crys_class.compute_volume(), float)


def test_abs_positions(crys_class):
    """Check the absolute positions function return corectly shaped matrix."""
    assert crys_class.get_abs_positions().shape == np.array(crys_class.positions).shape


def test_get_matrix(crys_class):
    matrix = crys_class.get_matrix()
    assert matrix.shape == (3, 3)
    assert np.all(matrix >= 0)
    for i in range(3):
        assert matrix[i, i] > 0


def test_matrix_values(crys_class):
    matrix = crys_class.get_matrix()
    if crys_class.dimensions == 2:
        assert np.all(matrix[:2, 2] == 0)
        assert np.all(matrix[2, :2] == 0)
        assert np.all(matrix[2, 2] == 1)

    if isinstance(crys_class, crystals.SquareCircle):
        assert matrix[0, 0] == 2
        assert matrix[1, 1] == 2
