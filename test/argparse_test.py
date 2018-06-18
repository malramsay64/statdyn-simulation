#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the parsing of arguments gives the correct results."""

from typing import NamedTuple

import pytest
from click.testing import CliRunner

from sdrun.main import __version__, create, equil, prod, sdrun

FUNCS = [
    ("prod", ["infile"]),
    ("equil", ["infile", "outfile"]),
    ("create", ["outfile"]),
]


@pytest.fixture
def runner():
    yield CliRunner()


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


def test_version(runner):
    result = runner.invoke(sdrun, ["--version"])
    assert "sdrun" in result.output
    assert __version__ in result.output
