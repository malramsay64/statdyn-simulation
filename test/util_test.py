#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the util module."""

import hoomd
import numpy as np
import pytest
from hoomd.data import boxdim, make_snapshot
from numpy.testing import assert_allclose

# TODO set_integrator, set_debug, set_dump, dump_frame, set_thermo
from sdrun.util import (
    NumBodies,
    _get_num_bodies,
    get_num_mols,
    get_num_particles,
    z2quaternion,
)


@pytest.fixture
def create_snapshot():
    num_particles = 10
    with hoomd.context.initialize(""):
        snapshot = make_snapshot(
            N=num_particles, box=boxdim(1, 1, 1), particle_types=["A"] * num_particles
        )
        yield {
            "num_bodies": NumBodies(num_particles, num_particles),
            "snapshot": snapshot,
        }


@pytest.mark.parametrize(
    "type_conv", [int, float, np.float32, np.int32, np.uint32, np.int64]
)
def test_num_bodies_types(type_conv):
    num_particles, num_mols = 300, 100
    n_bodies = NumBodies(type_conv(num_particles), type_conv(num_mols))
    assert type(n_bodies.particles) is int
    assert n_bodies.particles == num_particles
    assert type(n_bodies.molecules) is int
    assert n_bodies.molecules == num_mols


def test_num_bodies(create_snapshot):
    test_bodies = _get_num_bodies(create_snapshot["snapshot"])
    assert test_bodies == create_snapshot["num_bodies"]


def test_get_num_particles(create_snapshot):
    num_particles = get_num_particles(create_snapshot["snapshot"])
    assert type(num_particles) is int
    assert num_particles == create_snapshot["num_bodies"].particles


def test_get_num_mols(create_snapshot):
    num_bodies = get_num_mols(create_snapshot["snapshot"])
    assert type(num_bodies) is int
    assert num_bodies == create_snapshot["num_bodies"].molecules


@pytest.fixture(params=["Interface", "No-Interface"])
def snapshot(request):
    snap_file = "test/data/Trimer-13.50-3.00.gsd"
    if request.param == "Interface":
        snapshot = hoomd.data.gsd_snapshot(snap_file)
        num_mols = get_num_mols(snapshot)
        stationary_particles = int(num_mols / 3)
        snapshot.particles.velocity[:stationary_particles] = np.zeros(
            (stationary_particles, 3), dtype=np.float32
        )
        snapshot.particles.angmom[:stationary_particles] = np.zeros(
            (stationary_particles, 4), dtype=np.float32
        )
        return snapshot
    return hoomd.data.gsd_snapshot("test/data/Trimer-13.50-3.00.gsd")


def test_z2quaternion():
    angles = np.deg2rad(np.arange(360))
    quats = z2quaternion(angles)

    assert quats.dtype == np.float32
    assert_allclose(np.linalg.norm(quats, axis=1), 1.)
    assert np.all(quats[:, 1:3] == 0.)
