#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the sdrun command line tools."""

import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_parameters():
    command_list = ["create", "equil", "prod"]
    crystal_list = ["p2", "p2gg", "pg"]
    for command, crystal in product(command_list, crystal_list):
        if command == "create":
            for interface in [True, False]:
                yield {"command": command, "crystal": crystal, "interface": interface}
        else:
            yield {"command": command, "crystal": crystal}


@pytest.fixture(params=create_parameters())
def arguments(request):
    lattice_x, lattice_y = 20, 24
    crystal = request.param["crystal"]
    # p2gg lattice has twice the particles in the y direction of other crystals
    if crystal == "p2gg":
        lattice_y = int(lattice_y / 2)

    common_args = [
        '--hoomd-args="--mode=cpu"',
        "--num-steps=2000",
        "--temperature=2.50",
        "--pressure=13.50",
        f"--space-group={crystal}",
        "--lattice-lengths",
        str(lattice_x),
        str(lattice_y),
        "-vvv",
    ]
    args = []

    with TemporaryDirectory() as tmp_dst:
        tmp_dst = Path(tmp_dst)
        common_args += ["-o", str(tmp_dst / "output")]

        command = request.param["command"]
        if command == "create":
            if request.param.get("interface"):
                args = [
                    "--init-temp=0.4",
                    "create",
                    "--interface",
                    str(tmp_dst / "test_create.gsd"),
                ]
            else:
                args = ["create", str(tmp_dst / "test_create.gsd")]
        elif command == "equil":
            args = [
                "equil",
                "--equil-type=liquid",
                "test/data/Trimer-13.50-3.00.gsd",
                str(tmp_dst / "test_equil.gsd"),
            ]
        elif command == "prod":
            args = ["prod", "--no-dynamics", "test/data/Trimer-13.50-3.00.gsd"]

        yield common_args + args


def test_commands(arguments):
    """Ensure sdrun command line interface works.

    This tests the command line interface is both installed and working correctly, testing each of
    the main arguments.

    """
    logger.debug("Running command: sdrun %s", " ".join(arguments))
    ret = subprocess.run(["sdrun"] + arguments)
    assert ret.returncode == 0


@pytest.mark.timeout(60)
@pytest.mark.skipif(sys.platform == "darwin", reason="No MPI support on macOS")
def test_commands_mpi(arguments):
    """Ensure sdrun command line interface works with MPI.

    This ensures that running commands with MPI doesn't break things unexpectedly. The test is only
    run on Linux systems since macOS doesn't have simple support for MPI. The tests run here are
    exactly the same as the test_commands tests, apart from running with MPI.

    """
    command = ["mpirun", "-np", "4", "sdrun"] + arguments
    end_time = datetime.now() + timedelta(seconds=50)
    ret = subprocess.run(
        command,
        env=dict(os.environ, HOOMD_WALLTIME_STOP=str(end_time.timestamp())),
        stderr=subprocess.PIPE,
    )
    walltime_alert = b"hoomd._hoomd.WalltimeLimitReached: HOOMD_WALLTIME_STOP reached"
    assert ret.returncode == 0 or (
        ret.returncode == 15 and walltime_alert in ret.stderr
    )
