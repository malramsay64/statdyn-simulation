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
from tempfile import TemporaryDirectory

import pytest
from sdrun.main import parse_args

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture
def output_directory():
    with TemporaryDirectory() as tmp_dst:
        yield tmp_dst


TEST_ARGS = [
    ["prod", "test/data/Trimer-13.50-3.00.gsd", "-t", "3.00", "--no-dynamics"],
    [
        "create",
        "-t",
        "2.50",
        "--space-group",
        "pg",
        "--lattice-lengths",
        "20",
        "24",
        "test_create.gsd",
    ],
    ["equil", "-t", "2.50", "test/data/Trimer-13.50-3.00.gsd", "test_equil.gsd"],
]

COMMON_ARGS = ["--hoomd-args", '"--mode=cpu"', "-s", "100", "-v"]


@pytest.mark.parametrize("arguments", TEST_ARGS)
def test_manually(arguments, output_directory):
    """Testing the functionality of the argument parsing.

    This test replicates the operations of the main() function in a more manual fashion, ensuring
    the parseing of the arguments works appropriately.

    """
    logging.debug("output_directory: %s", output_directory)
    # Use temporary directory for output files
    if arguments[0] in ["create", "equil"]:
        arguments[-1] = str(Path(output_directory) / arguments[-1])
    func, sim_params = parse_args(arguments + COMMON_ARGS + ["-o", output_directory])
    func(sim_params)


@pytest.mark.parametrize("arguments", TEST_ARGS)
def test_commands(arguments, output_directory):
    """Ensure sdrun command line interface works.

    This tests the command line interface is both installed and working correctly, testing each of
    the main arguments.

    """
    logging.debug("output_directory: %s", output_directory)
    # Use temporary directory for output files
    if arguments[0] in ["create", "equil"]:
        arguments[-1] = str(Path(output_directory) / arguments[-1])
    command = ["sdrun"] + arguments + COMMON_ARGS + ["-o", output_directory]
    ret = subprocess.run(command)
    assert ret.returncode == 0


@pytest.mark.skipif(sys.platform == "darwin", reason="No MPI support on macOS")
@pytest.mark.parametrize("arguments", TEST_ARGS)
def test_commands_mpi(arguments, output_directory):
    """Ensure sdrun command line interface works with mpi.

    This ensures that running commands with MPI doesn't break things unexpectedly. The test is only
    run on linux systems since macOS doesn't have simple support for MPI. The tests run here are
    exactly the same as the test_commands tests, apart from running with mpi.

    """
    logging.debug("output_directory: %s", output_directory)
    # Use temporary directory for output files
    if arguments[0] in ["create", "equil"]:
        arguments[-1] = str(Path(output_directory) / arguments[-1])
    command = (
        ["mpirun", "-np", "4", "sdrun"]
        + arguments
        + COMMON_ARGS
        + ["-o", output_directory]
    )
    ret = subprocess.run(command)
    assert ret.returncode == 0
