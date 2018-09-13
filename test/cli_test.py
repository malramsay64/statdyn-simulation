#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import pytest
from click.testing import CliRunner
from sdrun.main import prod

def test_default_dynamics(monkeypatch):
    def production(snapshot, sim_context, sim_params, dynamics, simulation_type):
        assert dynamics == True

    monkeypatch.setattr("sdrun.simulation.production", production)

    runner = CliRunner()
    result = runner.invoke(prod)

def test_setting_dynamics(monkeypatch):
    def production(snapshot, sim_context, sim_params, dynamics, simulation_type):
        assert dynamics == False

    monkeypatch.setattr("sdrun.simulation.production", production)

    runner = CliRunner()
    result = runner.invoke(prod, ['--no-dynamics'])
