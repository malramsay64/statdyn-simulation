#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the parsing of arguments gives the correct results."""

import logging
from typing import Callable, List, NamedTuple

import click
import pytest
from click.testing import CliRunner

from sdrun.main import (
    CRYSTAL_FUNCS,
    MOLECULE_OPTIONS,
    __version__,
    create,
    equil,
    prod,
    sdrun,
)
from sdrun.params import SimulationParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


FUNCS = [
    ("prod", ["infile"]),
    ("equil", ["infile", "outfile"]),
    ("create", ["outfile"]),
]


@pytest.fixture
def runner():
    yield CliRunner()


def print_params_values(sim_params: SimulationParams) -> None:
    """A pretty printing routine for the values of SimulationParams class."""
    for key, value in sim_params.__dict__.items():
        print(f"{key}={value}")


@click.command("dummy_subcommand")
@click.pass_obj
def dummy_subcommand(obj):
    """Command which allows for the testing of the sdrun arguments.

    This prints the values of the SimulationParams object to stdout to allow for testing of the
    input values. Additionally this bypasses the running of another subcommand of sdrun which is a
    requirement for the appropriate exit status.

    """
    print_params_values(obj)


@pytest.fixture(params=["create", "equil", "prod"])
def subcommands(request):
    class Subcommand(NamedTuple):
        command: Callable
        params: List

    if request.param == "create":
        yield Subcommand(create, ["infile"])
    elif request.param == "equil":
        yield Subcommand(equil, ["infile", "outfile"])
    elif request.param == "prod":
        yield Subcommand(prod, ["outfile"])


sdrun.add_command(dummy_subcommand, "dummy_subcommand")


def create_params():
    """Function to create a list of parameters and values to test."""
    for option in [
        "--num-steps",
        "--molecule",
        "--iteration-id",
        "--space-group",
        "--moment-inertia-scale",
        "--tau",
        "--taup",
    ]:
        value = None

        if "molecule" in option:
            for value in MOLECULE_OPTIONS.keys():
                yield {"option": option, "value": value}
        elif "space-group" in option:
            for value in CRYSTAL_FUNCS.keys():
                yield {"option": option, "value": value}
        elif "tau" in option:
            for value in [0.1, 0.5, 1.0, 5.0]:
                yield {"option": option, "value": value}
        else:
            for value in [0, 100, 1000, 10000]:
                yield {"option": option, "value": value}


def test_version(runner):
    result = runner.invoke(sdrun, ["--version"])
    assert "sdrun" in result.output
    assert __version__ in result.output


@pytest.mark.parametrize("params", create_params())
def test_sdanalysis_options(runner, params):
    result = runner.invoke(
        sdrun, [params["option"], params["value"], "dummy_subcommand"]
    )
    assert result.exit_code == 0, result.output
    logger.debug("Command Output: \n%s", result.output)
    option = params["option"].strip("-").replace("-", "_")
    assert f"{option}={params['value']}".lower() in result.output.lower()
