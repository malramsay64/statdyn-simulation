#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Crystals module for generating unit cells for use in hoomd."""

from typing import Tuple

import attr
import hoomd
import numpy as np
import rowan

from .molecules import Disc, Molecule, Sphere, Trimer
from .util import z2quaternion


@attr.s(auto_attribs=True, eq=False)
class Crystal:
    """Defines the base class of a crystal lattice."""

    cell_matrix: np.ndarray = attr.ib(default=attr.Factory(lambda: np.identity(3)))
    molecule: Molecule = attr.ib(default=attr.Factory(Molecule))
    positions: np.ndarray = attr.ib(default=attr.Factory(lambda: np.zeros((1, 3))))
    _orientations: np.ndarray = attr.ib(default=attr.Factory(lambda: np.zeros(1)))

    @property
    def dimensions(self) -> int:
        return self.molecule.dimensions

    @property
    def num_molecules(self) -> int:
        """Return the number of molecules."""
        return len(self._orientations)

    def get_cell_len(self) -> Tuple[Tuple[float, ...], ...]:
        """Return the unit cell parameters.

        Returns:
            tuple: A tuple containing all the unit cell parameters

        """
        return tuple(tuple(i) for i in self.cell_matrix)

    def get_relative_positions(self) -> np.ndarray:
        """Return the relative positions of the molecules.

        This converts the absolute positions that the positions are stored in with the relative
        positions and returns.

        Returns:
            class:`numpy.ndarray`: Positions of each molecule

        """
        return self.positions @ np.linalg.inv(self.cell_matrix)

    def get_unitcell(self) -> hoomd.lattice.unitcell:
        """Return the hoomd unit cell parameter."""
        a1, a2, a3 = self.get_cell_len()  # pylint: disable=invalid-name
        mass = self.molecule.mass
        num_mols = self.num_molecules
        type_name = "R" if self.molecule.rigid else "A"
        return hoomd.lattice.unitcell(
            N=num_mols,
            a1=a1,
            a2=a2,
            a3=a3,
            position=self.positions,
            dimensions=self.dimensions,
            orientation=self.get_orientations(),
            type_name=[type_name] * num_mols,
            mass=[mass] * num_mols,
            moment_inertia=([self.molecule.moment_inertia] * num_mols),
        )

    def compute_volume(self) -> float:
        """Calculate the volume of the unit cell.

        If the number of dimensions is 2, then the returned value will be the
        area rather than the volume.

        Returns:
            float: Volume or area in unitless quantity

        """
        a1, a2, a3 = tuple(self.cell_matrix)  # pylint: disable=invalid-name
        if self.dimensions == 3:
            return np.linalg.norm(np.dot(a1, np.cross(a2, a3)))

        if self.dimensions == 2:
            return np.linalg.norm(np.cross(a1, a2))

        raise ValueError("Dimensions needs to be either 2 or 3")

    def get_orientations(self) -> np.ndarray:
        """Return the orientation quaternions of each molecule.

        Args:
            angle (float): The angle that a molecule is oriented

        Returns:
            class:`numpy.ndarray`: Quaternion representation of the angles

        """
        # Convert from degrees to radians
        angles = np.deg2rad(self._orientations).astype(np.float32)
        return z2quaternion(angles)

    def get_num_molecules(self) -> int:
        """Return the number of molecules."""
        return len(self._orientations)

    def __eq__(self, other) -> bool:
        if self.__class__ is other.__class__:
            return (
                np.allclose(self.cell_matrix, other.cell_matrix)
                and self.molecule == other.molecule
                and np.allclose(self.positions, other.positions)
                and np.allclose(self._orientations, other._orientations)
            )
        return False


def _calc_shift(orientations: np.ndarray, molecule: Molecule) -> np.ndarray:
    """A function to calculate the shift of large particle positions to COM positions.

    The positions defined in the Trimer classes are for the center of the 'large' particle since
    this was the output from the packing algorithm. This offsets these values to be on the center of
    mass, taking into account the orientation of the molecule.

    Args:
        orientations: Orientation of particles in degrees
        molecule: The molecule for which the shift is occurring.

    """
    pos_shift = molecule.get_relative_positions()[0]
    orient_quat = rowan.from_euler(np.deg2rad(orientations), 0, 0)
    return rowan.rotate(orient_quat, pos_shift)


class TrimerP2(Crystal):
    """Defining the unit cell of the p2 group of the Trimer molecule."""

    def __init__(self) -> None:
        molecule = Trimer()
        pos = np.array([[0.70, 0.32, 0.5]])
        orientation = 309

        cell_matrix = np.array([[3.82, 0, 0], [-0.63, 2.55, 0], [0, 0, 1]])
        # These are the relative positions within the unit cell
        positions = np.concatenate([pos, 1 - pos]) @ cell_matrix
        orientations = np.array([orientation, orientation + 180])
        # Divide (matrix inverse) by unit cell lengths to get relative positions
        positions -= _calc_shift(orientations, molecule)
        super().__init__(
            cell_matrix=cell_matrix,
            positions=positions,
            orientations=orientations,
            molecule=molecule,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TrimerP2gg(Crystal):
    """Unit cell of p2gg trimer.

    The positions are given in fractional coordinates.

    """

    def __init__(self):
        molecule = Trimer()
        cell_matrix = np.array([[2.63, 0, 0], [0, 7.38, 0], [0, 0, 1]])
        # These are the relative positions within the unit cell
        positions = (
            np.array(
                [
                    [0.061, 0.853, 0],
                    [0.561, 0.647, 0],
                    [0.439, 0.353, 0],
                    [0.939, 0.147, 0],
                ]
            )
        ) @ cell_matrix
        orientations = np.array([24, 156, -24, 204])
        # Divide (matrix inverse) by unit cell lengths to get relative positions
        positions -= _calc_shift(orientations, molecule)
        super().__init__(
            cell_matrix=cell_matrix,
            positions=positions,
            orientations=orientations,
            molecule=molecule,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TrimerPg(Crystal):
    """Unit Cell of pg Trimer."""

    @staticmethod
    def glide(position: np.ndarray) -> np.ndarray:
        return position * np.array([-1, 1, 1]) + np.array([1, 0.5, 0])

    def __init__(self):
        molecule = Trimer()
        position = np.array([[0.65, 0.45, 0.5]])
        angle = 21.4

        cell_matrix = np.array([[2.71, 0, 0], [0, 3.63, 0], [0, 0, 1]])
        # These are the relative positions within the unit cell
        positions = np.concatenate([position, self.glide(position)]) @ cell_matrix
        orientations = np.array([180 - angle, 180 + angle])
        # Divide (matrix inverse) by unit cell lengths to get relative positions
        positions -= _calc_shift(orientations, molecule)
        super().__init__(
            cell_matrix=cell_matrix,
            positions=positions,
            orientations=orientations,
            molecule=molecule,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CubicSphere(Crystal):
    """Create a simple cubic lattice."""

    def __init__(self):
        cell_matrix = np.identity(3)
        super().__init__(cell_matrix=cell_matrix, molecule=Sphere())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SquareCircle(Crystal):
    """Create a square lattice."""

    def __init__(self):
        cell_matrix = np.identity(3)
        cell_matrix[2, 2] = 1
        super().__init__(cell_matrix=cell_matrix, molecule=Disc())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class HexagonalCircle(Crystal):
    def __init__(self, length=1.0):
        cell_matrix = np.array([[length, 0, 0], [0, np.sqrt(3) * length, 0], [0, 0, 1]])
        position = np.array([[0, 0, 0], [length / 2, np.sqrt(3) * length / 2, 0]])
        orientation = np.array([0, 0])
        super().__init__(
            cell_matrix=cell_matrix,
            positions=position,
            orientations=orientation,
            molecule=Disc(),
        )


CRYSTAL_FUNCS = {
    "p2": TrimerP2,
    "p2gg": TrimerP2gg,
    "pg": TrimerPg,
    "SquareCircle": SquareCircle,
    "CubicSphere": CubicSphere,
    "HexagonalCircle": HexagonalCircle,
}
