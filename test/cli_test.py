#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from sdrun import SimulationParams
from sdrun.main import prod, sdrun
from sdrun.molecules import MOLECULE_DICT


@pytest.fixture
def runner():
    runner = CliRunner()
    with runner.isolated_filesystem():
        yield runner


@pytest.fixture
def sim_params():
    return SimulationParams()


def test_default_dynamics(monkeypatch, runner, sim_params):
    def production(dynamics, infile):
        assert dynamics is True

    monkeypatch.setattr(prod, "callback", production)

    testfile = "test.gsd"
    Path(testfile).touch()
    result = runner.invoke(prod, [testfile])
    assert result.exit_code == 0, result.output
    result = runner.invoke(prod, ["--dynamics", testfile])
    assert result.exit_code == 0, result.output
    result = runner.invoke(prod, ["--no-dynamics", testfile])
    assert result.exit_code != 0


def test_setting_dynamics(monkeypatch, runner):
    def production(dynamics, infile):
        assert dynamics is False

    monkeypatch.setattr(prod, "callback", production)

    testfile = "test.gsd"
    Path(testfile).touch()
    result = runner.invoke(prod, ["--no-dynamics", testfile])
    assert result.exit_code == 0, result.output
    result = runner.invoke(prod, ["--dynamics", testfile])
    assert result.exit_code != 0
    result = runner.invoke(prod)
    assert result.exit_code != 0


def test_default_lattice_lengths(runner):
    testfile = "test.gsd"
    result = runner.invoke(sdrun, ["create", testfile])
    assert result.exit_code == 0, result.output
    assert Path(testfile).exists()


def test_no_warnings(runner):
    testfile = str(Path(__file__).parent / "data/recreate_thermo_warnings.gsd")

    result = runner.invoke(
        sdrun, ["--num-steps=1", "--temperature=0.80", "prod", testfile]
    )
    assert result.exit_code == 0, result.output
    assert r"*Warning*:" not in result.output, result.output


@pytest.mark.parametrize("molecule", MOLECULE_DICT.keys())
def test_molecule(runner, molecule):
    result = runner.invoke(
        sdrun,
        [
            "--num-steps=1",
            "--temperature=0.10",
            f"--molecule={molecule}",
            "create",
            "test.gsd",
        ],
    )
    assert result.exit_code == 0, result.output
    assert r"*Warning*:" not in result.output, result.output
