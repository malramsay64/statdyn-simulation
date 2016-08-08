#!/usr/bin/env python
""" A set of classes used for computing the dynamic properties of a Hoomd MD
simulation"""

from __future__ import print_function
import os
import numpy as np
from hoomd_script import (init,
                          update,
                          pair,
                          integrate,
                          analyze,
                          group,
                          run_upto,
                          run,
                          dump)
import StepSize
from TimeDep import TimeDep2dRigid








def compute_dynamics(input_xml,
                     temp,
                     press,
                     steps,
                     rigid=True):
    """ Run a hoomd simulation calculating the dynamic quantites on a power
    law scale such that both short timescale and long timescale events are
    vieable on the same figure while retaining a reasonable runtime.
    for the simulation

    Args:
        input_xml (string): Filename of the file containing the input
            configuration
        temp (float): The target temperature at which to run the simulation
        press (float): The target pressure at which to run the simulation
        rigid (bool): Boolean value indicating whether to integrate using rigid
            bodes.
    """
    if init.is_initialized():
        init.reset()
    basename = os.path.splitext(input_xml)[0]

    # Fix for issue where pressure is higher than desired
    press /= 2.2

    # Initialise simulation parameters
    # context.initialize()
    system = init.read_xml(filename=input_xml, time_step=0)
    update.enforce2d()

    potentials = pair.lj(r_cut=2.5)
    potentials.pair_coeff.set('1', '1', epsilon=1, sigma=2)
    potentials.pair_coeff.set('2', '2', epsilon=1, sigma=0.637556*2)
    potentials.pair_coeff.set('1', '2', epsilon=1, sigma=1.637556)

    # Create particle groups
    gall = group.all()

    # Set integration parameters
    integrate.mode_standard(dt=0.005)
    if rigid:
        integrate.npt_rigid(group=gall, T=temp, tau=1, P=press, tauP=1)
    else:
        integrate.npt(group=gall, T=temp, tau=1, P=press, tauP=1)

    # initial run to settle system after reading file
    run(10000)

    thermo = analyze.log(filename=basename+"-thermo.dat",
                         quantities=['temperature', 'pressure',
                                     'potential_energy',
                                     'rotational_kinetic_energy',
                                     'translational_kinetic_energy'
                                    ],
                         period=1000)

    # Initialise dynamics quantities
    dyn = TimeDep2dRigid(system)
    dyn.print_heading(basename+"-dyn.dat")
    tstep_init = system.get_metadata()['timestep']
    new_step = StepSize.PowerSteps(start=tstep_init)
    struct = [(new_step.next(), new_step, dyn)]
    timestep = tstep_init
    key_rate = 20000
    xml = dump.xml(all=True)
    xml.write(filename=input_xml)

    while timestep < steps+tstep_init:
        index_min = struct.index(min(struct))
        next_step, step_iter, dyn = struct[index_min]
        timestep = next_step
        print(timestep, file=open("timesteps.dat", 'a'))
        run_upto(timestep)
        dyn.print_all(system, outfile=basename+"-dyn.dat")
        # dyn.print_corr_dist(system, outfile=basename+"-corr.dat")

        struct[index_min] = (step_iter.next(), step_iter, dyn)
        # Add new key frame when a run reaches 10000 steps
        if (timestep % key_rate == 0 and
                len(struct) < 5000 and
                len([s for s in struct if s[0] == timestep+1]) == 0):
            new_step = StepSize.PowerSteps(start=timestep)
            struct.append((new_step.next(), new_step, TimeDep2dRigid(system)))
        xml.write(filename=input_xml)
    thermo.query('pressure')


