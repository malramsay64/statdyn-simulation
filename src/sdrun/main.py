#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Run simulation with boilerplate taken care of by the statdyn library."""

import argparse
import logging
from pathlib import Path
from typing import Callable, List, Tuple

import hoomd.context

from .crystals import CRYSTAL_FUNCS
from .equilibrate import equil_crystal, equil_harmonic, equil_interface, equil_liquid
from .initialise import init_from_crystal, init_from_file, init_from_none
from .molecules import Dimer, Disc, Sphere, Trimer
from .params import SimulationParams
from .simrun import run_harmonic, run_npt
from .version import __version__

logger = logging.getLogger(__name__)
MOLECULE_OPTIONS = {"trimer": Trimer, "disc": Disc, "sphere": Sphere, "dimer": Dimer}
EQUIL_OPTIONS = {
    "interface": equil_interface,
    "liquid": equil_liquid,
    "crystal": equil_crystal,
    "harmonic": equil_harmonic,
}


def sdrun() -> None:
    """Run main function."""
    logging.debug("Running main function")
    func, sim_params = parse_args()
    func(sim_params)


def prod(sim_params: SimulationParams) -> None:
    """Run simulations on equilibrated phase."""
    logger.debug("running prod")
    logger.debug("Reading %s", sim_params.infile)
    snapshot = init_from_file(
        sim_params.infile, sim_params.molecule, hoomd_args=sim_params.hoomd_args
    )
    logger.debug("Snapshot initialised")
    sim_context = hoomd.context.initialize(sim_params.hoomd_args)
    if sim_params.harmonic_force is not None:
        run_harmonic(snapshot, sim_context, sim_params)
    else:
        run_npt(snapshot, sim_context, sim_params)


def equil(sim_params: SimulationParams) -> None:
    """Command group for the equilibration of configurations."""
    logger.debug("Running %s equil", sim_params.equil_type)
    logger.debug("Equil hoomd args: %s", sim_params.hoomd_args)
    snapshot = init_from_file(
        sim_params.infile, sim_params.molecule, hoomd_args=sim_params.hoomd_args
    )
    EQUIL_OPTIONS.get(sim_params.equil_type)(snapshot, sim_params=sim_params)


def create(sim_params: SimulationParams) -> None:
    """Create things."""
    logger.debug("Running create.")
    if sim_params.parameters.get("crystal"):
        snapshot = init_from_crystal(sim_params=sim_params)
    else:
        snapshot = init_from_none(sim_params=sim_params)
    equil_crystal(snapshot=snapshot, sim_params=sim_params)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--steps",
        dest="num_steps",
        type=int,
        help="The number of steps for which to run the simulation.",
    )
    parser.add_argument(
        "--output-interval",
        type=int,
        help="Steps between output of dump and thermodynamic quantities.",
    )
    parser.add_argument(
        "--hoomd-args",
        type=str,
        help="Arguments to pass to hoomd on context.initialize",
    )
    parser.add_argument("--pressure", type=float, help="Pressure for simulation")
    parser.add_argument(
        "-t", "--temperature", type=float, help="Temperature for simulation"
    )
    parser.add_argument(
        "-o", "--output", dest="output", type=str, help="Directory to output files to"
    )
    parser.add_argument(
        "-k", "--harmonic-force", type=float, help="Harmonic force constant"
    )
    parse_molecule = parser.add_argument_group("molecule")
    parse_molecule.add_argument("--molecule", choices=MOLECULE_OPTIONS.keys())
    parse_molecule.add_argument(
        "--distance", type=float, help="Distance at which small particles are situated"
    )
    parse_molecule.add_argument(
        "--moment-inertia-scale",
        type=float,
        help="Scaling factor for the moment of inertia.",
    )
    parse_crystal = parser.add_argument_group("crystal")
    parse_crystal.add_argument(
        "--space-group",
        choices=CRYSTAL_FUNCS.keys(),
        help="Space group of initial crystal.",
    )
    parse_crystal.add_argument(
        "--lattice-lengths",
        dest="cell_dimensions",
        nargs=2,
        type=int,
        help="Number of repetitiions in the a and b lattice vectors",
    )
    parse_steps = parser.add_argument_group("steps")
    parse_steps.add_argument("--gen-steps", type=int)
    parse_steps.add_argument("--max-gen", type=int)
    default_parser = argparse.ArgumentParser(add_help=False)
    default_parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Enable debug logging flags."
    )
    default_parser.add_argument(
        "--version", action="version", version="sdrun {0}".format(__version__)
    )
    # TODO write up something useful in the help
    simtype = argparse.ArgumentParser(add_help=False, parents=[default_parser])
    subparsers = simtype.add_subparsers()
    parse_equilibration = subparsers.add_parser(
        "equil", add_help=False, parents=[parser, default_parser]
    )
    parse_equilibration.add_argument(
        "--init-temp",
        type=float,
        help="Temperature to start equilibration from if differnt from the target.",
    )
    parse_equilibration.add_argument(
        "--equil-type", default="liquid", choices=EQUIL_OPTIONS.keys()
    )
    parse_equilibration.add_argument("infile", type=str)
    parse_equilibration.add_argument("outfile", type=str)
    parse_equilibration.set_defaults(func=equil)
    parse_production = subparsers.add_parser(
        "prod", add_help=False, parents=[parser, default_parser]
    )
    parse_production.add_argument(
        "--no-dynamics", dest="dynamics", action="store_false"
    )
    parse_production.add_argument("--dynamics", action="store_true")
    parse_production.add_argument("infile", type=str)
    parse_production.set_defaults(func=prod)
    parse_create = subparsers.add_parser(
        "create", add_help=False, parents=[parser, default_parser]
    )
    parse_create.add_argument("--interface", default=False, action="store_true")
    parse_create.add_argument("outfile", type=str)
    parse_create.set_defaults(func=create)
    return simtype


def _verbosity(level: int = 0) -> None:
    root_logger = logging.getLogger("statdyn")
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    log_level = levels.get(level, logging.DEBUG)
    logging.basicConfig(level=log_level)
    root_logger.setLevel(log_level)


def parse_args(
    input_args: List[str] = None
) -> Tuple[Callable[[SimulationParams], None], SimulationParams]:
    """Logic to parse the input arguments."""
    parser = create_parser()
    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)
    # Handle verbosity
    _verbosity(args.verbose)
    del args.verbose
    # Handle subparser function
    try:
        func = args.func
        del args.func
    except AttributeError:
        parser.print_help()
        exit()
    # Parse Molecules
    my_mol = MOLECULE_OPTIONS.get(getattr(args, "molecule", None))
    if my_mol is None:
        my_mol = Trimer
    mol_kwargs = {}
    for attr in ["distance", "moment_inertia_scale"]:
        if getattr(args, attr, None) is not None:
            mol_kwargs[attr] = getattr(args, attr)
    args.molecule = my_mol(**mol_kwargs)
    # Parse space groups
    if func == create:
        try:
            args.crystal = CRYSTAL_FUNCS[args.space_group]()
        except KeyError:
            args.crystal = None
    set_args = {key: val for key, val in vars(args).items() if val is not None}
    sim_params = SimulationParams(**set_args)
    # Ensure directories exist
    try:
        print(f"Making outfile directory {Path(sim_params.outfile).parent}")
        logger.debug("Making outfile directory %s", Path(sim_params.outfile).parent)
        Path(sim_params.outfile).parent.mkdir(exist_ok=True)
    except AttributeError:
        pass
    logger.debug("Making output directory %s", sim_params.output)
    Path(sim_params.output).mkdir(exist_ok=True)
    return func, sim_params
