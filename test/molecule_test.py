#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the molecule class."""

import hoomd
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats
from sdrun.molecules import MOLECULE_DICT, Trimer


@pytest.fixture(params=MOLECULE_DICT.values(), ids=MOLECULE_DICT.keys())
def mol(request):
    with hoomd.context.initialize(""):
        yield request.param()


def test_compute_moment_inertia(mol):
    mom_I = np.array(mol.moment_inertia)
    assert np.all(mom_I[:2] == 0)


def test_scale_moment_inertia(mol):
    scale_factor = 10.
    init_mom_I = np.array(mol.moment_inertia)
    mol.scale_moment_inertia(scale_factor)
    final_mom_I = np.array(mol.moment_inertia)
    assert np.all(scale_factor * init_mom_I == final_mom_I)


def test_get_radii(mol):
    radii = mol.get_radii()
    assert radii[0] == 1.


def test_read_only_position(mol):
    assert mol.positions.flags.writeable == False


def test_get_types(mol):
    mol.get_types()


def test_moment_inertia_trimer():
    """Ensure calculation of moment of inertia is working properly."""
    mol = Trimer()
    assert mol.moment_inertia == (0, 0, 1.6666666666666665)
    mol = Trimer(distance=0.8)
    assert mol.moment_inertia[2] < 1.6666666666666665
    mol = Trimer(distance=1.2)
    assert mol.moment_inertia[2] > 1.6666666666666665


@given(floats(min_value=0, allow_infinity=False, allow_nan=False))
def test_moment_inertia_scaling(scaling_factor):
    """Test that the scaling factor is working properly."""
    reference = Trimer()
    with np.errstate(over="ignore"):
        scaled = Trimer(moment_inertia_scale=scaling_factor)
        assert len(reference.moment_inertia) == len(scaled.moment_inertia)
        assert np.allclose(
            np.array(reference.moment_inertia) * scaling_factor,
            np.array(scaled.moment_inertia),
        )


def test_compute_size(mol):
    size = mol.compute_size()
    assert size >= 2.
