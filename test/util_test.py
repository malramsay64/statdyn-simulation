#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the util module."""

import hoomd
import pytest
from hoomd.data import boxdim, make_snapshot

# TODO set_integrator, set_debug, set_dump, dump_frame, set_thermo, set_harmonic_force
from sdrun.util import NumBodies, _get_num_bodies, get_num_mols, get_num_particles


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
