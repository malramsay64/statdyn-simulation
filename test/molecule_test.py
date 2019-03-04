#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the molecule class."""

from collections import OrderedDict

import hoomd
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import floats
from numpy.testing import assert_allclose

from sdrun.molecules import Dimer, Disc, Molecule, Sphere, Trimer

ABS_TOLERANCE = 1e-12


@pytest.fixture
def molecule_class():
    return Molecule()


def test_molecule_class(molecule_class):
    assert molecule_class.dimensions == 3
    assert molecule_class.moment_inertia_scale == 1.0
    assert np.all(molecule_class.positions == np.zeros((1, 3)))
    assert molecule_class.particles == ["A"]
    assert molecule_class.potential_args == {"r_cut": 2.5}
    assert molecule_class._radii["A"] == 1.0
    assert molecule_class.positions.flags.writeable == False


def test_molecule_class_mutability(molecule_class):
    """This ensures that new dicts/lists are created on class initialisation."""
    assert molecule_class.particles == ["A"]
    molecule_class.particles.append("B")
    assert molecule_class.particles == ["A", "B"]
    mol2 = Molecule()
    assert mol2.particles == ["A"]

    assert molecule_class._radii == OrderedDict(A=1.0)
    molecule_class._radii["B"] = 2.0
    assert molecule_class._radii == OrderedDict(A=1.0, B=2.0)
    mol2 = Molecule()
    assert mol2._radii == OrderedDict(A=1.0)

    assert molecule_class.potential_args == {"r_cut": 2.5}
    molecule_class.potential_args["B"] = 2.0
    assert molecule_class.potential_args == {"r_cut": 2.5, "B": 2.0}
    mol2 = Molecule()
    assert mol2.potential_args == {"r_cut": 2.5}


def test_compute_moment_inertia(molecule):
    mom_I = np.array(molecule.moment_inertia)
    assert np.all(mom_I[:2] == 0)


def test_scale_moment_inertia(molecule):
    scale_factor = 10.
    init_mom_I = molecule.moment_inertia
    print(init_mom_I)
    molecule.moment_inertia_scale = scale_factor
    final_mom_I = molecule.moment_inertia
    print(final_mom_I)
    assert np.all(scale_factor * init_mom_I == final_mom_I)


def test_get_radii(molecule):
    radii = molecule.get_radii()
    assert radii[0] == 1.


def test_read_only_position(molecule):
    assert molecule.positions.flags.writeable == False


def test_get_types(molecule):
    molecule.get_types()


def test_moment_inertia_shape(molecule):
    assert molecule.moment_inertia.shape == (3,)


def test_mass_trimer():
    assert Trimer().mass == 3


def test_mass_dimer():
    assert Dimer().mass == 2


def test_moment_inertia_trimer():
    """Ensure calculation of moment of inertia is working properly."""
    molecule = Trimer()
    assert_allclose(molecule.moment_inertia, np.array([0, 0, 1.6666666666666665]))
    molecule = Trimer(distance=0.8)
    assert molecule.moment_inertia[0] == 0
    assert molecule.moment_inertia[1] == 0
    assert molecule.moment_inertia[2] < 1.6666666666666665
    molecule = Trimer(distance=1.2)
    assert molecule.moment_inertia[0] == 0
    assert molecule.moment_inertia[1] == 0
    assert molecule.moment_inertia[2] > 1.6666666666666665


def test_moment_inertia_dimer():
    """Ensure calculation of moment of inertia is working properly."""
    molecule = Dimer()
    assert molecule.moment_inertia[0] == 0
    assert molecule.moment_inertia[1] == 0
    assert np.isclose(molecule.moment_inertia[2], 0.5)
    molecule = Dimer(distance=0.8)
    assert molecule.moment_inertia[0] == 0
    assert molecule.moment_inertia[1] == 0
    assert np.isclose(molecule.moment_inertia[2], 0.32)
    molecule = Dimer(distance=1.2)
    assert molecule.moment_inertia[0] == 0
    assert molecule.moment_inertia[1] == 0
    assert np.isclose(molecule.moment_inertia[2], 0.72)


@given(floats(min_value=0, allow_infinity=False, allow_nan=False))
def test_moment_inertia_scaling(scaling_factor):
    """Test that the scaling factor is working properly."""
    reference = Trimer()
    with np.errstate(over="ignore"):
        scaled = Trimer(moment_inertia_scale=scaling_factor)
        assert len(reference.moment_inertia) == len(scaled.moment_inertia)
        assert_allclose(
            np.array(reference.moment_inertia) * scaling_factor,
            np.array(scaled.moment_inertia),
        )


def test_compute_size(molecule):
    size = molecule.compute_size()
    assert size >= 2.


def test_dimensions(molecule):
    assert molecule.dimensions in [2, 3]


def test_rigid(molecule):
    if not molecule.rigid:
        return
    with hoomd.context.initialize("--mode=cpu"):
        sys = hoomd.init.create_lattice(hoomd.lattice.sq(2, type_name="R"), 2)
        for particle_type in molecule.get_types():
            sys.particles.types.add(particle_type)
        molecule.define_rigid()


def test_get_relative_positions(molecule):
    """Test that the center of mass of the relative positions is the origin (0, 0, 0)."""
    center_of_mass = molecule.get_relative_positions()
    assert_allclose(np.mean(center_of_mass, axis=0), np.zeros(3), atol=ABS_TOLERANCE)


@pytest.mark.parametrize("mol", [Molecule, Dimer, Trimer, Disc, Sphere])
def test_equality(mol):
    molecule = mol()
    assert molecule == molecule
    mol_copy = mol()
    assert mol_copy == molecule

    mol_copy.particles.append("test")
    assert mol_copy != molecule

    class mol_sub(mol):
        def __init__(self):
            super().__init__()

    subclass = mol_sub()

    assert molecule != subclass
