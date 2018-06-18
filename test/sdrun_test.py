#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the sdrun command line tools."""

import logging
import subprocess
import sys
from pathlib import Path
from pprint import pformat
from tempfile import TemporaryDirectory

import pytest
from click.testing import CliRunner

from sdrun.main import sdrun

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture(params=["create", "equil", "prod"])
def arguments(request):
    common_args = [
        "--hoomd-args",
        '"--mode=cpu"',
        "-s",
        "100",
        "-vv",
        "-t",
        "2.50",
        "--space-group",
        "pg",
        "--lattice-lengths",
        "20",
        "24",
    ]
    args = []

    with TemporaryDirectory() as tmp_dst:
        tmp_dst = Path(tmp_dst)
        common_args += ["-o", str(tmp_dst / "output")]

        if request.param == "create":
            args = ["create", str(tmp_dst / "test_create.gsd")]
        elif request.param == "equil":
            args = [
                "equil",
                "--equil-type",
                "liquid",
                "test/data/Trimer-13.50-3.00.gsd",
                str(tmp_dst / "test_equil.gsd"),
            ]
        elif request.param == "prod":
            args = ["prod", "--no-dynamics", "test/data/Trimer-13.50-3.00.gsd"]

        yield common_args + args


@pytest.fixture
def runner():
    return CliRunner()


def test_manually(arguments, runner):
    """Testing the functionality of the argument parsing.

    This test replicates the operations of the main() function in a more manual fashion, ensuring
    the parseing of the arguments works appropriately.

    """
    result = runner.invoke(sdrun, arguments)
    logger.debug("Runner output: \n%s", result.output)
    assert result.exit_code == 0


def test_commands(arguments):
    """Ensure sdrun command line interface works.

    This tests the command line interface is both installed and working correctly, testing each of
    the main arguments.

    """
    command = ["sdrun"] + arguments
    logger.debug("Running command: \n%s", pformat(command))
    ret = subprocess.run(command)
    assert ret.returncode == 0


@pytest.mark.skipif(sys.platform == "darwin", reason="No MPI support on macOS")
def test_commands_mpi(arguments):
    """Ensure sdrun command line interface works with mpi.

    This ensures that running commands with MPI doesn't break things unexpectedly. The test is only
    run on linux systems since macOS doesn't have simple support for MPI. The tests run here are
    exactly the same as the test_commands tests, apart from running with mpi.

    """
    command = ["mpirun", "-np", "4", "sdrun"] + arguments
    ret = subprocess.run(command)
    assert ret.returncode == 0
