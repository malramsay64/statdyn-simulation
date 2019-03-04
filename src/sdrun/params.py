#! /usr/bin/env python3
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
from typing import TYPE_CHECKING, Optional, Tuple, Union

import attr
import hoomd

from .molecules import Molecule, Trimer

if TYPE_CHECKING:
    from .crystals import Crystal  # pylint: disable=unused-import

logger = logging.getLogger(__name__)


def _to_path(value: Optional[Path]) -> Optional[Path]:
    if value is not None:
        return Path(value)
    return value


@attr.s(auto_attribs=True)
class SimulationParams:
    """Store the parameters of the simulation."""

    # Thermodynamic Params
    _temperature: float = 0.4
    tau: float = attr.ib(default=1.0, repr=False)
    pressure: float = 13.5
    tauP: float = attr.ib(default=1.0, repr=False)
    init_temp: Optional[float] = None

    # Molecule params
    _molecule: Optional[Molecule] = None
    moment_inertia_scale: Optional[float] = None

    # Crystal Params
    crystal: Optional["Crystal"] = None
    _cell_dimensions: Tuple[int, ...] = (30, 42, 30)
    space_group: Optional[str] = None

    # Step Params
    num_steps: int = 0
    step_size: float = 0.005
    num_linear: int = attr.ib(default=100, repr=False)
    max_gen: int = attr.ib(default=500, repr=False)
    gen_steps: int = attr.ib(default=20_000, repr=False)
    output_interval: int = attr.ib(default=10_000, repr=False)

    # File Params
    _infile: Optional[Path] = attr.ib(default=None, converter=_to_path, repr=False)
    _outfile: Optional[Path] = attr.ib(default=None, converter=_to_path, repr=False)
    _output: Path = attr.ib(factory=Path.cwd, converter=Path, repr=False)

    hoomd_args: str = attr.ib(default="--notice-level=1", repr=False)
    iteration_id: Optional[int] = None

    @property
    def temperature(self) -> Union[float, hoomd.variant.linear_interp]:
        """Temperature of the system."""
        assert self._temperature is not None
        assert self._temperature >= 0

        if self.init_temp is None:
            return self._temperature

        assert self.init_temp > 0
        assert self.num_steps is not None
        assert self.num_steps > 0

        ramp_steps = int(min(0.75e6, self.num_steps * 0.75))
        logger.debug("Ramp steps: %d", ramp_steps)
        return hoomd.variant.linear_interp(
            [
                (0, self.init_temp),
                (ramp_steps, self._temperature),
                (self.num_steps, self._temperature),
            ],
            zero="now",
        )

    @temperature.setter
    def temperature(self, value: float) -> None:
        assert value is not None
        assert value >= 0, f"Temperature cannot be negative. You gave: {value}"
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
            mol.moment_inertia_scale = self.moment_inertia_scale
        return mol

    @molecule.setter
    def molecule(self, value: Molecule) -> None:
        self._molecule = value

    @property
    def cell_dimensions(self) -> Tuple[int, int, int]:
        """The number of unit cells in the direction of each lattice vector."""
        logger.debug("self._cell_dimensions %s", self._cell_dimensions)
        assert self._cell_dimensions is not None

        # Convert the immutable tuple to a mutable list
        if isinstance(self._cell_dimensions, int):
            cell_dims = [self._cell_dimensions] * self.molecule.dimensions
        else:
            cell_dims = list(self._cell_dimensions)

        # Handle list of length 1 (Integer values handled on conversion to list)
        if len(cell_dims) == 1:
            cell_dims = cell_dims * self.molecule.dimensions

        # Ensure the list is of length 3 by adding dimension with value of 1
        if len(cell_dims) == 2:
            cell_dims = cell_dims + [1]
            return (cell_dims[0], cell_dims[1], cell_dims[2])

        # 2D molecules need to have z dimension of 1
        if self.molecule.dimensions == 2:
            cell_dims[2] = 1

        return (cell_dims[0], cell_dims[1], cell_dims[2])

    @cell_dimensions.setter
    def cell_dimensions(self, value: Tuple[int, ...]) -> None:
        if value is None:
            raise ValueError("Cannot set cell_dimensions to None")
        if isinstance(value, (list, tuple)) and not value:
            raise ValueError("Length of cell_dimensions cannot be 0")
        self._cell_dimensions = value

    @property
    def infile(self) -> Optional[Path]:
        """The input file of the simulation."""
        return self._infile

    @infile.setter
    def infile(self, value: Path) -> None:
        # Ensure value is a Path
        self._infile = Path(value)

    @property
    def outfile(self) -> Optional[Path]:
        """The path to write the output configuration to."""
        return self._outfile

    @outfile.setter
    def outfile(self, value: Optional[Path]) -> None:
        # Ensure value is a Path
        if value is not None:
            self._outfile = Path(value)

    @property
    def output(self) -> Path:
        """The Path to write all output files.

        The output files include thermodynamics data, trajectory data and anything else
        generated by the simulation.

        """
        return self._output

    @output.setter
    def output(self, value: Optional[Path]) -> None:
        # Ensure value is a Path
        if value is not None:
            self._output = Path(value)

    def filename(self, prefix: str = None) -> Path:
        """Use the simulation parameters to construct a filename."""
        base_string = "{molecule}-P{pressure:.2f}-T{temperature:.2f}"
        if prefix is not None:
            base_string = "{prefix}-" + base_string
        if self.moment_inertia_scale is not None:
            base_string += "-I{mom_inertia:.2f}"

        if self.space_group is not None:
            base_string += "-{space_group}"

        if self.iteration_id is not None:
            base_string += "-ID{iteration_id}"

        logger.debug("filename base string: %s", base_string)
        logger.debug("Temperature: %.2f", self._temperature)

        # Default extension, required as with_suffix replaces existing extension
        # which is mistaken for the final decimal points.
        base_string += ".gsd"

        fname = base_string.format(
            prefix=prefix,
            molecule=self.molecule,
            pressure=self.pressure,
            temperature=self._temperature,
            mom_inertia=self.moment_inertia_scale,
            space_group=self.space_group,
            iteration_id=self.iteration_id,
        )
        return self.output / fname

    @contextmanager
    def temp_context(self, **kwargs):
        """Temporarily change values of the SimulationParams object.

        This enables the use of python's context constuct, i.e. the `with` block, to
        temporarily change the value of one of the parameters of a SimulationParams
        object.

        """
        old_params = {
            key: val
            for key, val in self.__dict__.items()
            if not isinstance(val, property)
        }
        for key, value in kwargs.items():
            setattr(self, key, value)
        yield self
        self.__dict__.update(old_params)
