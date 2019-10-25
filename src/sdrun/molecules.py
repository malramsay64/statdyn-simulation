#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Module to define a molecule to use for simulation."""

import logging
from itertools import combinations_with_replacement
from typing import Any, Dict, List, Optional

import attr
import hoomd
import hoomd.md
import numpy as np
from hoomd.md.pair import pair as Pair

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True, eq=True)
class Molecule:
    """Molecule class holding information on the molecule for use in hoomd.

    This class contains all the parameters required to initialise the molecule
    in a hoomd simulation. This includes all interaction potentials, the rigid
    body interactions and the moments of inertia.

    The Molecule class is a template class that defines a number of functions
    subclasses can use to set these variables however it generates no sensible
    molecule itself.

    """

    dimensions: int = 3
    moment_inertia_scale: float = 1
    positions: np.ndarray = attr.ib(
        default=attr.Factory(lambda: np.zeros((1, 3))), repr=False, eq=False
    )
    potential: Pair = attr.ib(hoomd.md.pair.slj, repr=False)
    particles: List[str] = attr.ib(default=attr.Factory(lambda: ["A"]))
    potential_args: Dict[str, Any] = attr.ib(default=attr.Factory(dict), repr=False)
    _radii: Dict[str, float] = attr.ib(default=attr.Factory(dict))
    rigid: bool = False

    def __attrs_post_init__(self) -> None:
        self.potential_args.setdefault("r_cut", 2.5)
        self._radii.setdefault("A", 1.0)
        self.positions.flags.writeable = False

    @property
    def center_of_mass(self) -> np.ndarray:
        return np.mean(self.positions, axis=0)

    @property
    def mass(self) -> float:
        return float(len(self.particles))

    @property
    def num_particles(self) -> int:
        """Count of particles in the molecule."""
        if self.rigid:
            # Rigid bodies have an additional center-of-mass particle
            return len(self.particles) + 1
        return len(self.particles)

    @property
    def moment_inertia(self) -> np.ndarray:
        """The moment of inertia of the particle.

        The moment-of-inertia for the Lx, Ly, and Lz dimensions. This assumes all particles have a
        mass of 1, with the mass all concentrated at a single point. A 2D molecule will only have
        a moment of inertia in the Lz dimension.

        """
        off_diagonal = 1 - np.identity(3)
        # The moment of inertia for a dimension is comprised of the squared distance of the other
        # dimensions. The off diagonal terms are the remaining dimensions.
        moment_inertia = np.square(self.get_relative_positions()) @ off_diagonal
        # Sum over all the particles
        moment_inertia = moment_inertia.sum(axis=0)
        moment_inertia *= self.moment_inertia_scale
        # A 2D molecule only has an Lz
        if self.dimensions == 2:
            moment_inertia[:2] = 0
        return moment_inertia

    def get_types(self) -> List[str]:
        """Get the types of particles present in a molecule."""
        if self.rigid:
            return ["R"] + sorted(list(self._radii.keys()))
        return sorted(list(self._radii.keys()))

    def define_dimensions(self) -> None:
        """Set the number of dimensions for the simulation.

        This takes into account the number of dimensions of the molecule,
        a 2D molecule can only be a 2D molecule, since there will be no
        rotations in that 3rd dimension anyway.
        """
        if self.dimensions == 2:
            hoomd.md.update.enforce2d()

    def define_potential(self) -> hoomd.md.pair.pair:
        r"""Define the potential in the simulation context.

        A helper function that defines the potential to be used by the hoomd
        simulation context. The default values for the potential are a
        Lennard-Jones potential with a cutoff of :math:`2.5\sigma` and
        interaction parameters of :math:`\epsilon = 1.0` and
        :math:`\sigma = 2.0`.

        Returns:
            class:`hoomd.md.pair`: The interaction potential object class.

        """
        potential = self.potential(**self.potential_args, nlist=hoomd.md.nlist.cell())
        # Each combination of two particles requires a pair coefficient to be defined
        sites = list(self._radii.keys())
        if self.rigid:
            sites.append("R")
        for i, j in combinations_with_replacement(sites, 2):
            if "R" in [i, j]:
                potential.pair_coeff.set(i, j, epsilon=0, sigma=0)
            else:
                potential.pair_coeff.set(
                    i, j, epsilon=1, sigma=self._radii[i] + self._radii[j]
                )
        return potential

    def define_rigid(
        self, params: Dict[Any, Any] = None
    ) -> Optional[hoomd.md.constrain.rigid]:
        """Define the rigid constraints of the molecule.

        This is a helper function to define the rigid body constraints of the
        particular molecules within the hoomd context.

        Args:
            create (bool): Flag that toggles the option of creating the
                additional particles when creating the rigid bodies. Defaults
                to False.
            params (dict): Dictionary defining the rigid body structure. The
                default values for the `type_name` of A and the `types` of the
                `self.particles` variable should work for the vast majority of
                systems, so the only value required should be the topology.

        Returns:
            class:`hoomd.md.constrain.rigid`: Rigid constraint object

        """
        if not self.rigid:
            logger.info("Not a rigid body")
            return None

        if params is None:
            params = dict()

        params["type_name"] = "R"
        params["types"] = self.particles
        params.setdefault("positions", list(self.get_relative_positions()))
        rigid = hoomd.md.constrain.rigid()
        rigid.set_param(**params)
        logger.debug("Rigid: %s", rigid)
        return rigid

    def __str__(self) -> str:
        return type(self).__name__

    def get_radii(self) -> np.ndarray:
        """Radii of the particles."""
        return np.array([self._radii[p] for p in self.particles])

    def get_relative_positions(self) -> np.ndarray:
        """The positions of the particles relative to the COM.

        This computes the positions relative to the center-of-mass, i.e. using the center-of-mass as
        the origin.

        """
        return self.positions - self.center_of_mass

    def compute_size(self):
        """Compute the maximum possible size of the molecule.

        This is a rough estimate of the size of the molecule for the creation
        of a lattice that contains no overlaps.

        """
        length = np.max(np.max(self.positions, axis=1) - np.min(self.positions, axis=1))
        return length + 2 * self.get_radii().max()

    def __eq__(self, other) -> bool:
        if self.__class__ is other.__class__:
            return (
                self.dimensions == other.dimensions
                and self.moment_inertia_scale == other.moment_inertia_scale
                and np.allclose(self.positions, other.positions)
                and self.particles == other.particles
                and self.potential_args == other.potential_args
                and self._radii == other._radii
                and self.rigid == other.rigid
            )
        return False


class Disc(Molecule):
    """Defines a 2D particle."""

    def __init__(self) -> None:
        """Initialise 2D disc particle."""
        super().__init__(dimensions=2)


class Sphere(Molecule):
    """Define a 3D sphere."""

    def __init__(self) -> None:
        """Initialise Spherical particle."""
        super().__init__()


class Trimer(Molecule):
    """Defines a Trimer molecule for initialisation within a hoomd context.

    This defines a molecule of three particles, shaped somewhat like Mickey
    Mouse. The central particle is of type `'A'` while the outer two
    particles are of type `'B'`. The type `'B'` particles, have a variable
    radius and are positioned at a specified distance from the central
    type `'A'` particle. The angle between the two type `'B'` particles,
    subtended by the type `'A'` particle is the other degree of freedom.


    """

    def __init__(
        self,
        radius: float = 0.637_556,
        distance: float = 1.0,
        angle: float = 120,
        moment_inertia_scale: float = 1.0,
    ) -> None:
        """Initialise trimer molecule.

        Args:
            radius: Radius of the small particles. Default is 0.637556
            distance: Distance of the outer particles from the central
                one. Default is 1.0
            angle: Angle between the two outer particles in degrees.
                Default is 120
            moment_inertia_scale: Scale the moment of inertia by this
                factor.

        """
        assert isinstance(radius, (float, int))
        assert isinstance(distance, (float, int))
        assert isinstance(angle, (float, int))
        self.radius = radius
        self.distance = distance
        self.angle = angle
        rad_ang = np.deg2rad(angle)
        particles = ["A", "B", "B"]
        radii = {"A": 1.0, "B": self.radius}
        positions = np.array(
            [
                [0, 0, 0],
                [-distance * np.sin(rad_ang / 2), distance * np.cos(rad_ang / 2), 0],
                [distance * np.sin(rad_ang / 2), distance * np.cos(rad_ang / 2), 0],
            ]
        )
        super().__init__(
            positions=positions,
            dimensions=2,
            radii=radii,
            particles=particles,
            moment_inertia_scale=moment_inertia_scale,
            rigid=True,
        )

    @property
    def rad_angle(self) -> float:
        return np.radians(self.angle)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(radius={self.radius}, distance={self.distance}, "
            f"angle={self.angle}, moment_inertia_scale={self.moment_inertia_scale})"
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return (
                self.radius == other.radius
                and self.distance == other.distance
                and self.angle == other.angle
            )

        return False


class Dimer(Molecule):
    """Defines a Dimer molecule for initialisation within a hoomd context.

    This defines a molecule of three particles, shaped somewhat like Mickey
    Mouse. The central particle is of type `'A'` while the outer two
    particles are of type `'B'`. The type `'B'` particles, have a variable
    radius and are positioned at a specified distance from the central
    type `'A'` particle. The angle between the two type `'B'` particles,
    subtended by the type `'A'` particle is the other degree of freedom.

    """

    def __init__(
        self,
        radius: float = 0.637_556,
        distance: float = 1.0,
        moment_inertia_scale: float = 1.0,
    ) -> None:
        """Initialise Dimer molecule.

        Args:
            radius: Radius of the small particles. Default is 0.637556
            distance: Distance of the outer particles from the central
                one. Default is 1.0
            moment_inertia_scale: Scale the moment of inertia by this
                factor.

        """
        self.radius = radius
        self.distance = distance
        particles = ["A", "B"]
        radii = {"A": 1.0, "B": self.radius}
        positions = np.array([[0, 0, 0], [0, self.distance, 0]])
        super().__init__(
            dimensions=2,
            particles=particles,
            positions=positions,
            radii=radii,
            moment_inertia_scale=moment_inertia_scale,
            rigid=True,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(radius={self.radius}, distance={self.distance},"
            f" moment_inertia_scale={self.moment_inertia_scale})"
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return self.radius == other.radius and self.distance == other.distance

        return False


MOLECULE_DICT = {"trimer": Trimer, "dimer": Dimer, "disc": Disc}

MOLECULE_LIST = [Trimer(), Dimer(), Disc()]
