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

from sdrun.initialise import initialise_snapshot

# TODO set_debug, set_dump, dump_frame, set_thermo
from sdrun.util import (
    NumBodies,
    _get_num_bodies,
    get_group,
    get_num_mols,
    get_num_particles,
    set_integrator,
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


@pytest.fixture()
def initialised_simulation(snapshot_from_none):
    snapshot = snapshot_from_none["snapshot"]
    sim_params = snapshot_from_none["sim_params"]
    context = hoomd.context.initialize(sim_params.hoomd_args)
    sys = initialise_snapshot(snapshot, context, sim_params)
    with context:
        yield sys, sim_params


@pytest.mark.parametrize("simulation_type", ["liquid", "crystal", "interface"])
@pytest.mark.parametrize("integration_method", ["NPT", "NVT"])
def test_set_integrator(initialised_simulation, simulation_type, integration_method):
    sys, sim_params = initialised_simulation
    group = get_group(sys, sim_params)
    integrator = set_integrator(
        sim_params,
        group=group,
        simulation_type=simulation_type,
        integration_method=integration_method,
    )
    assert integrator.enabled
    assert integrator.kT.val == sim_params.temperature
    assert integrator.group == group
    assert integrator.tau == sim_params.tau

    if integration_method == "NPT":
        assert integrator.tauP == sim_params.tauP
        # The three values set to zero are a stress value
        assert integrator.S == [sim_params.pressure] * 3 + [0] * 3

        # Crystal simulations all pressures are independent
        if simulation_type == "crystal":
            assert integrator.couple == "none"
        else:
            assert integrator.couple == "xyz"

    else:
        with pytest.raises(AttributeError):
            assert integrator.tauP
        with pytest.raises(AttributeError):
            assert integrator.S


def test_set_integrator_vary_temp(initialised_simulation):
    sys, sim_params = initialised_simulation
    with sim_params.temp_context(init_temp=0.2):
        group = get_group(sys, sim_params)
        integrator = set_integrator(
            sim_params, group=group, simulation_type="liquid", integration_method="NPT"
        )
        assert integrator.enabled
        assert integrator.kT.points
        assert integrator.group == group
        assert integrator.tau == sim_params.tau
        assert integrator.tauP == sim_params.tauP
        assert integrator.S == [sim_params.pressure] * 3 + [0] * 3
        assert integrator.couple == "xyz"


def test_get_group(initialised_simulation):
    sys, sim_params = initialised_simulation
    group_rigid = get_group(sys, sim_params)
    if sim_params.molecule.rigid:
        assert group_rigid.name == "rigid_center"
    else:
        assert group_rigid.name == "all"
    group_interface = get_group(sys, sim_params, interface=True)
    assert group_interface.name == "rigid_mobile"
    # Ensure the interface group has fewer particles
    assert (
        group_interface.cpp_group.getNumMembersGlobal()
        < group_rigid.cpp_group.getNumMembersGlobal()
    )
