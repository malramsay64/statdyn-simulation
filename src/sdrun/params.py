#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Parameters for passing between functions."""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple, Union

import attr
import hoomd

from .crystals import Crystal
from .molecules import Molecule, Trimer

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class SimulationParams(object):
    """Store the parameters of the simulation."""
    # Thermodynamic Params
    _temperature: float = 0.4
    tau: float = attr.ib(default=1.0, repr=False)
    pressure: float = 13.5
    tauP: float = attr.ib(default=1.0, repr=False)
    init_temp: Optional[float] = None
    _group: Optional[hoomd.group.group] = None
    equil_type: str = attr.ib("liquid")
    minimize: bool = False

    # Molecule params
    _molecule: Optional[Molecule] = None
    moment_inertia_scale: Optional[float] = None
    harmonic_force: Optional[float] = None

    # Crystal Params
    crystal: Optional[Crystal] = None
    _cell_dimensions: Tuple[int, ...] = (30, 42, 30)
    space_group: str = None

    # Step Params
    num_steps: Optional[int] = None
    step_size: float = 0.005
    max_gen: int = attr.ib(default=500, repr=False)
    gen_steps: int = attr.ib(default=20_000, repr=False)
    output_interval: int = attr.ib(default=10_000, repr=False)
    dynamics: bool = attr.ib(False, repr=False)

    # File Params
    _infile: Optional[Path] = attr.ib(
        default=None, converter=attr.converters.optional(Path), repr=False
    )
    _outfile: Optional[Path] = attr.ib(
        default=None, converter=attr.converters.optional(Path), repr=False
    )
    _output: Path = attr.ib(
        default=Path.cwd(), converter=attr.converters.optional(Path), repr=False
    )

    hoomd_args: str = attr.ib(default="", repr=False)

    @property
    def temperature(self) -> Union[float, hoomd.variant.linear_interp]:
        """Temperature of the system."""
        if self.init_temp is None:
            return self._temperature

        return hoomd.variant.linear_interp(
            [
                (0, self.init_temp),
                (int(self.num_steps * 0.75), self._temperature),
                (self.num_steps, self._temperature),
            ],
            zero="now",
        )

    @temperature.setter
    def temperature(self, value: float) -> None:
        assert value is not None
        self._temperature = float(value)

    @property
    def molecule(self) -> Molecule:
        """Return the appropriate molecule.

        Where there is no custom molecule defined then we return the molecule of
        the crystal.

        """
        if self._molecule is not None:
            mol = self._molecule
        elif self.crystal is not None:
            mol = self.crystal.molecule
        else:
            logger.debug("Using default molecule")
            mol = Trimer()
        # Ensure scale_moment_inertia set on molecule
        if self.moment_inertia_scale is not None:
            mol.moment_inertia = mol.compute_moment_inertia(self.moment_inertia_scale)
        return mol

    @molecule.setter
    def molecule(self, value: Molecule) -> None:
        self._molecule = value

    @property
    def cell_dimensions(self) -> Tuple[int, int, int]:
        logger.debug("self._cell_dimensions %s", self._cell_dimensions)
        if isinstance(self._cell_dimensions, int):
            cell_dims = [self._cell_dimensions] * self.molecule.dimensions
        else:
            cell_dims = list(self._cell_dimensions)

        if len(cell_dims) == 1:
            cell_dims = cell_dims * self.molecule.dimensions

        if len(cell_dims) == 3:
            if self.molecule.dimensions == 2:
                cell_dims[2] = 1
            return tuple(cell_dims)

        elif len(cell_dims) == 2:
            cell_dims = cell_dims + [1]
            return tuple(cell_dims)

    @cell_dimensions.setter
    def cell_dimensions(self, value: Tuple[int, ...]) -> None:
        self._cell_dimensions = value

    @property
    def group(self) -> hoomd.group.group:
        """Return the appropriate group."""
        if self._group:
            return self._group

        if self.molecule.num_particles == 1:
            return hoomd.group.all()

        return hoomd.group.rigid_center()

    @group.setter
    def group(self, value: hoomd.group.group) -> None:
        self._group = value

    @property
    def infile(self) -> Path:
        return self._infile

    @infile.setter
    def infile(self, value: Path) -> None:
        # Ensure value is a Path
        self._infile = Path(value)

    @property
    def outfile(self) -> Path:
        return self._outfile

    @outfile.setter
    def outfile(self, value: Optional[Path]) -> None:
        # Ensure value is a Path
        if value is not None:
            self._outfile = Path(value)

    @property
    def output(self) -> Path:
        return self._output

    @output.setter
    def output(self, value: Optional[Path]) -> None:
        # Ensure value is a Path
        if value is not None:
            self._output = Path(value)

    @equil_type.validator
    def validate_equil_type(self, attribute, value):
        if value not in ["liquid", "crystal", "interface", "harmonic"]:
            raise ValueError(
                "'equil_type' must be one of '(liquid|crystal|interface|harmonic)'"
            )

    def filename(self, prefix: str = None) -> Path:
        """Use the simulation parameters to construct a filename."""
        base_string = "{molecule}-P{pressure:.2f}-T{temperature:.2f}"
        if prefix is not None:
            base_string = "{prefix}-" + base_string
        if self.moment_inertia_scale is not None:
            base_string += "-I{mom_inertia:.2f}"
        if self.harmonic_force is not None:
            base_string += "-K{harmonic_force:.2f}"

        space_group = None
        if self.space_group is not None:
            base_string += "-{space_group}"

        logger.debug("filename base string: %s", base_string)
        logger.debug("Temperature: %.2f", self._temperature)

        # Default extension, required as with_suffix replaces existsing extension
        # which is mistaken for the final decimal points.
        base_string += ".gsd"

        fname = base_string.format(
            prefix=prefix,
            molecule=self.molecule,
            pressure=self.pressure,
            temperature=self._temperature,
            mom_inertia=self.moment_inertia_scale,
            space_group=self.space_group,
            harmonic_force=self.harmonic_force,
        )
        return self.output / fname

    @contextmanager
    def temp_context(self, **kwargs):
        old_params = {
            key: val
            for key, val in self.__dict__.items()
            if not isinstance(val, property)
        }
        for key, value in kwargs.items():
            setattr(self, key, value)
        yield self
        self.__dict__.update(old_params)
