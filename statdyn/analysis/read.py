#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Read input files and compute dynamic and thermodynamic quantities."""

import logging
from typing import List

import gsd.hoomd
import pandas

from ..StepSize import GenerateStepSeries
from .dynamics import dynamics

logger = logging.getLogger(__name__)


def process_gsd(infile: str,
                gen_steps: int=20000,
                step_limit: int=None,
                ) -> pandas.DataFrame:
    """Read a gsd file and compute the dynamics quantities.

    This computes the dynamic quantities from a gsd file returning the
    result as a pandas DataFrame. This is only suitable for cases where
    all the data will fit in memory, as there is no writing to a file.

    Args:
        infile (str): The filename of the gsd file from which to read the
            configurations.
        gen_steps (int): The value of the parameter `gen_steps` used when
            running the dynamics simulation. (default: 20000)
        step_limit (int): Limit the timescale of the processing. A value of
            ``None`` (default) will process all the files.

    Returns:
        (py:class:`pandas.DataFrame`): DataFrame with the dynamics quantities.

    """
    dataframes: List[pandas.DataFrame] = []
    keyframes: List[dynamics] = []

    curr_step = 0
    with gsd.hoomd.open(infile, 'rb') as src:
        if step_limit:
            num_steps = step_limit
        else:
            num_steps = src[-1].configuration.step
        logger.debug('Infile: %s contains %d steps', infile, num_steps)
        step_iter = GenerateStepSeries(num_steps,
                                       num_linear=100,
                                       gen_steps=gen_steps,
                                       max_gen=1000)
        curr_step = next(step_iter)
        for frame in src:
            logger.debug('Step %d with index %s',
                         curr_step, step_iter.get_index())
            if curr_step == frame.configuration.step:
                indexes = step_iter.get_index()
                for index in indexes:
                    try:
                        mydyn = keyframes[index]
                    except IndexError:
                        logger.debug('Create keyframe at step %s', curr_step)
                        keyframes.append(dynamics(
                            timestep=frame.configuration.step,
                            box=frame.configuration.box,
                            position=frame.particles.position,
                            orientation=frame.particles.orientation,
                        ))
                        mydyn = keyframes[index]

                    dataframes.append(pandas.DataFrame({
                        'time': mydyn.computeTimeDelta(curr_step),
                        'rotation': mydyn.get_rotations(frame.particles.orientation),
                        'translation': mydyn.get_displacements(frame.particles.position),
                        'molid': mydyn.get_molid(),
                        'start_index': index,
                    }))
                curr_step = next(step_iter)

            # This handles when the generators don't match up
            elif curr_step > frame.configuration.step:
                logger.warning('Step missing in iterator: current %d, frame %d',
                               curr_step, frame.configuration.step)
                continue

            elif curr_step < frame.configuration.step:
                logger.warning('Step missing in frame: current %d, frame %d',
                               curr_step, frame.configuration.step)
                curr_step = next(step_iter)

    return pandas.concat(dataframes)