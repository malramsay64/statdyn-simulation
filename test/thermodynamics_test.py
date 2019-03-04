#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, NamedTuple, Optional

import hoomd
import numpy as np
import pytest
from hoomd.context import SimulationContext as Context
from hoomd.data import SnapshotParticleData as Snapshot

from sdrun.crystals import CRYSTAL_FUNCS
from sdrun.initialise import init_from_crystal, initialise_snapshot
from sdrun.params import SimulationParams
from sdrun.simulation import get_group
from sdrun.util import (
    compute_rotational_ke,
    compute_translational_ke,
    get_num_mols,
    set_integrator,
)

rel_tolerance = 0.2


class SimStatus(NamedTuple):
    sim_params: SimulationParams
    snapshot: Snapshot
    context: Context
    system: Any
    logger: Optional[Any] = None
    thermo: Optional[Any] = None
    integrator: Optional[Any] = None

    @property
    def num_mols(self):
        return get_num_mols(self.snapshot)


@pytest.fixture(params=CRYSTAL_FUNCS.values(), ids=CRYSTAL_FUNCS.keys())
def setup_thermo(request):
    """Test the initialisation of all crystals."""
    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "output"
        output_dir.mkdir(exist_ok=True)
        sim_params = SimulationParams(
            temperature=0.4,
            pressure=1.0,
            num_steps=100,
            crystal=request.param(),
            output=output_dir,
            cell_dimensions=(10, 12, 10),
            outfile=output_dir / "test.gsd",
            hoomd_args="--mode=cpu --notice-level=0",
        )
        context = hoomd.context.initialize(args=sim_params.hoomd_args)
        sys = initialise_snapshot(
            snapshot=init_from_crystal(sim_params),
            context=context,
            sim_params=sim_params,
            thermalisation=True,
        )
        with context:
            group = get_group(sys, sim_params)
            integrator = set_integrator(sim_params, group)
            thermo_log = hoomd.analyze.log(
                None,
                quantities=[
                    "translational_kinetic_energy",
                    "rotational_kinetic_energy",
                    "temperature",
                ],
                period=1,
            )

            hoomd.run(1)
            snap = sys.take_snapshot()
        yield SimStatus(
            sim_params=sim_params,
            snapshot=snap,
            context=context,
            logger=thermo_log,
            integrator=integrator,
            system=sys,
        )


def test_mass(setup_thermo):
    assert np.allclose(
        setup_thermo.snapshot.particles.mass[: setup_thermo.num_mols],
        setup_thermo.sim_params.molecule.mass,
    )


def test_temperature(setup_thermo):
    with setup_thermo.context:
        logged_T = setup_thermo.logger.query("temperature")
        intended_T = setup_thermo.sim_params.temperature
        assert np.isclose(logged_T, intended_T, rtol=rel_tolerance)


def test_translational_ke(setup_thermo):
    with setup_thermo.context:
        trans_dimensions = setup_thermo.sim_params.molecule.dimensions

        computed_trans_ke = compute_translational_ke(setup_thermo.snapshot)
        thermodynamic_ke = trans_dimensions * 0.5 * setup_thermo.sim_params.temperature
        thermodynamic_ke *= setup_thermo.num_mols
        logged_ke = setup_thermo.logger.query("translational_kinetic_energy")

        assert np.isclose(computed_trans_ke, logged_ke, rtol=rel_tolerance)
        assert np.isclose(computed_trans_ke, thermodynamic_ke, rtol=rel_tolerance)


def test_rotational_ke(setup_thermo):
    with setup_thermo.context:
        rot_dimensions = np.sum(setup_thermo.sim_params.molecule.moment_inertia > 0.)

        computed_rot_ke = compute_rotational_ke(setup_thermo.snapshot)
        thermodynamic_ke = rot_dimensions * 0.5 * setup_thermo.sim_params.temperature
        thermodynamic_ke *= setup_thermo.num_mols
        logged_ke = setup_thermo.logger.query("rotational_kinetic_energy")

        assert np.isclose(computed_rot_ke, logged_ke, rtol=rel_tolerance)
        assert np.isclose(computed_rot_ke, thermodynamic_ke, rtol=rel_tolerance)
