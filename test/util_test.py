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

# TODO set_integrator, set_debug, set_dump, dump_frame, set_thermo, set_harmonic_force
from sdrun.util import (
    NumBodies,
    _get_num_bodies,
    get_num_mols,
    get_num_particles,
    randomise_momenta,
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


def test_num_bodies(create_snapshot):
    test_bodies = _get_num_bodies(create_snapshot["snapshot"])
    assert test_bodies == create_snapshot["num_bodies"]


def test_get_num_particles(create_snapshot):
    num_particles = get_num_particles(create_snapshot["snapshot"])
    assert num_particles == create_snapshot["num_bodies"].particles


def test_get_num_mols(create_snapshot):
    num_bodies = get_num_mols(create_snapshot["snapshot"])
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


@pytest.mark.parametrize("interface", [False, True])
def test_randomise_momenta(snapshot, interface):
    snap = randomise_momenta(snapshot, interface)
    num_mols = get_num_mols(snapshot)
    angmom_similarity = np.sum(
        snap.particles.angmom[:num_mols] != snapshot.particles.angmom[:num_mols]
    )
    velocity_similarity = np.sum(
        snap.particles.velocity[:num_mols] != snapshot.particles.velocity[:num_mols]
    )
    assert angmom_similarity < 5
    assert velocity_similarity < 5
    if interface:
        assert np.any(snap.particles.velocity[:num_mols] > 0)
        assert np.any(snap.particles.angmom[:num_mols] > 0)
        velocity_norm = np.linalg.norm(snap.particles.velocity[:num_mols], axis=1)
        assert np.sum(velocity_norm == 0) < 5
        angmom_norm = np.linalg.norm(snap.particles.angmom[:num_mols], axis=1)
        assert np.sum(angmom_norm == 0) < 5


@pytest.mark.parametrize("seed", [0, 1, 10])
def test_randomise_seed_same(snapshot, seed):
    snap1 = randomise_momenta(snapshot, random_seed=seed)
    snap2 = randomise_momenta(snapshot, random_seed=seed)
    assert np.all(snap1.particles.velocity == snap2.particles.velocity)
    assert np.all(snap1.particles.angmom == snap2.particles.angmom)


def test_randomise_seed_different(snapshot):
    num_mols = get_num_mols(snapshot)
    snap1 = randomise_momenta(snapshot, random_seed=0)
    snap2 = randomise_momenta(snapshot, random_seed=1)
    angmom_similarity = np.sum(
        snap1.particles.angmom[:num_mols] != snap2.particles.angmom[:num_mols]
    )
    velocity_similarity = np.sum(
        snap1.particles.velocity[:num_mols] != snap2.particles.velocity[:num_mols]
    )
    assert angmom_similarity < 5
    assert velocity_similarity < 5


def test_z2quaternion():
    angles = np.deg2rad(np.arange(360))
    quats = z2quaternion(angles)

    assert quats.dtype == np.float32
    assert np.allclose(np.linalg.norm(quats, axis=1), 1.)
    assert np.all(quats[:, 1:3] == 0.)
