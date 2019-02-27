#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Run simulation with boilerplate taken care of by the statdyn library."""

import logging
from pathlib import Path
from pprint import pformat

import click
import hoomd.context

from .crystals import CRYSTAL_FUNCS
from .initialise import init_from_crystal, init_from_file, init_from_none
from .molecules import Dimer, Disc, Sphere, Trimer
from .params import SimulationParams
from .simulation import create_interface, equilibrate, production
from .version import __version__

logger = logging.getLogger(__name__)
MOLECULE_OPTIONS = {"trimer": Trimer, "disc": Disc, "sphere": Sphere, "dimer": Dimer}


def _verbosity(ctx, param, value) -> None:  # pylint: disable=unused-argument
    root_logger = logging.getLogger("statdyn")
    levels = {0: "WARNING", 1: "INFO", 2: "DEBUG"}
    log_level = levels.get(value, "DEBUG")
    logging.basicConfig(level=log_level)
    root_logger.setLevel(log_level)
    logger.debug(f"Setting log level to %s", log_level)


@click.group()
@click.version_option(__version__)
@click.option(
    "-v",
    "--verbose",
    count=True,
    default=0,
    callback=_verbosity,
    expose_value=False,
    is_eager=True,
    help="Increase debug level",
)
@click.option("-t", "--temperature", type=float, help="Temperature to run simulation")
@click.option("-p", "--pressure", type=float, help="Pressure to run simulation")
@click.option(
    "-s", "--num-steps", type=int, help="The number of steps to run the simulation"
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    help="Location to save all output files; required to be a directory.",
)
@click.option(
    "--output-interval",
    type=int,
    help="Steps between the output of dump and thermodynamic properties.",
)
@click.option("--init-temp", type=float, help="Temperature to start equilibration.")
@click.option(
    "--keyframe-interval",
    "gen_steps",
    type=int,
    default=20_000,
    help="Timesteps between keyframes in step sequence.",
)
@click.option(
    "--molecule",
    type=click.Choice(MOLECULE_OPTIONS),
    help="Molecule to use for simulation",
)
@click.option(
    "--moment-inertia-scale",
    type=float,
    help="Scaling factor for the moment of inertia of the molecules",
)
@click.option(
    "--iteration-id", type=int, help="Identifier for isoconfigurational simulation run."
)
@click.option(
    "--space-group",
    type=str,
    help=(
        "Crystal to use for initialisation. This is also the space group "
        "to label output file."
    ),
)
@click.option(
    "--lattice-lengths",
    "cell_dimensions",
    nargs=2,
    type=int,
    default=(None, None),
    help="Replications of unit cell in each crystal dimension",
)
@click.option(
    "--hoomd-args",
    type=str,
    help=(
        "Arguments to pass to hoomd on context.initialise."
        "This needs to be quoted to send options, i.e. '--mode=cpu' "
    ),
)
@click.pass_context
def sdrun(ctx, **kwargs) -> None:
    """Run main function."""
    logging.debug("Running main function")
    space_group = kwargs.get("space_group")
    if space_group is not None:
        if space_group not in CRYSTAL_FUNCS:
            raise ValueError(f"The value of 'space_group': {space_group} is not valid.")
        crystal = CRYSTAL_FUNCS.get(space_group)
        assert crystal is not None
        kwargs["crystal"] = crystal()
    if kwargs.get("num_steps") is None:
        logger.warning(
            "Number of steps (--num-steps) not specified, setting to default value of 100."
        )
        kwargs["num_steps"] = 100
    logging.debug("Creating SimulationParams with values:\n%s", pformat(kwargs))
    ctx.obj = SimulationParams(
        **{
            key: val
            for key, val in kwargs.items()
            if val is not None and val != (None, None)
        }
    )
    logging.debug("SimulationParams Created: \n%s", ctx.obj)
    if ctx.obj.output is not None:
        ctx.obj.output.mkdir(exist_ok=True)


@sdrun.command()
@click.pass_obj
@click.option(
    "--dynamics/--no-dynamics",
    is_flag=True,
    default=True,
    help="Use exponential steps to capture dynamics properties",
)
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def prod(sim_params: SimulationParams, dynamics: bool, infile: Path) -> None:
    """Run simulations on equilibrated phase."""
    logger.debug("running prod")
    sim_params.infile = infile
    logger.debug("Reading %s", sim_params.infile)

    snapshot = init_from_file(
        sim_params.infile, sim_params.molecule, hoomd_args=sim_params.hoomd_args
    )
    logger.debug("Snapshot initialised")

    sim_context = hoomd.context.initialize(sim_params.hoomd_args)

    simulation_type = "liquid"

    production(snapshot, sim_context, sim_params, dynamics, simulation_type)


@sdrun.command()
@click.pass_obj
@click.option(
    "--equil-type",
    type=click.Choice(["liquid", "crystal", "interface"]),
    default="liquid",
    help="""The type of equilibration the simulation will undergo.
        - liquid -> A standard NPT simulation ensuring an orthorhombic simulation cell
        - crystal -> An NPT simulation with decoupled pressure tensors and allowing the
            cell to tilt
        - interface -> A NPT simulation which only integrates the particles in the outer
            1/3 of the simulation cell creating a liquid--crystal interface.
        """,
)
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("outfile", type=click.Path(file_okay=True, dir_okay=False))
def equil(
    sim_params: SimulationParams, equil_type: str, infile: Path, outfile: Path
) -> None:
    """Command group for the equilibration of configurations."""
    logger.debug("Running equilibration")
    sim_params.infile = infile
    sim_params.outfile = outfile
    snapshot = init_from_file(
        sim_params.infile, sim_params.molecule, hoomd_args=sim_params.hoomd_args
    )

    equilibrate(snapshot, sim_params, equil_type, thermalisation=True)

    logger.debug("Equilibration completed")


@sdrun.command()
@click.pass_obj
@click.option("--interface", is_flag=True, help="Whether to create an interface")
@click.argument("outfile", type=click.Path(file_okay=True, dir_okay=False))
def create(sim_params, interface: bool, outfile: Path) -> None:
    """Create things."""
    logger.debug("Running create.")
    sim_params.outfile = outfile
    if interface:
        create_interface(sim_params)
        return

    if sim_params.crystal is not None:
        snapshot = init_from_crystal(sim_params)
    else:
        snapshot = init_from_none(sim_params)
    equilibrate(snapshot, sim_params, equil_type="crystal")
