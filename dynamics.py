#!/usr/bin/env python
""" A set of classes used for computing the dynamic properties of a Hoomd MD
simulation"""

from __future__ import print_function
import os
import math
import hoomd
from hoomd import md, deprecated
import StepSize
from TimeDep import TimeDep2dRigid
from CompDynamics import CompRotDynamics


def compute_dynamics(input_xml,
                     temp,
                     press,
                     steps,
                    ):
    """ Run a hoomd simulation calculating the dynamic quantites on a power
    law scale such that both short timescale and long timescale events are
    vieable on the same figure while retaining a reasonable runtime.
    for the simulation

    Args:
        input_xml (string): Filename of the file containing the input
            configuration
        temp (float): The target temperature at which to run the simulation
        press (float): The target pressure at which to run the simulation
    """
    basename = os.path.splitext(input_xml)[0]

    # Initialise simulation parameters
    hoomd.context.initialize()
    system = deprecated.init.read_xml(filename=input_xml, time_step=0)
    md.update.enforce2d()

    for particle in system.particles:
        if particle.type == '1':
            particle.moment_inertia = (1.65, 10, 10)

    potentials = md.pair.lj(r_cut=2.5, nlist=md.nlist.cell())
    potentials.pair_coeff.set('1', '1', epsilon=1, sigma=2)
    potentials.pair_coeff.set('2', '2', epsilon=1, sigma=0.637556*2)
    potentials.pair_coeff.set('1', '2', epsilon=1, sigma=1.637556)

    rigid = md.constrain.rigid()
    rigid.set_param('1', positions=[(math.sin(math.pi/3),
                                     math.cos(math.pi/3), 0),
                                    (-math.sin(math.pi/3),
                                     math.cos(math.pi/3), 0)],
                    types=['2', '2']
                   )

    rigid.create_bodies(create=False)
    center = hoomd.group.rigid_center()

    # Set integration parameters
    md.integrate.mode_standard(dt=0.001)
    md.integrate.npt(group=center, kT=temp, tau=2, P=press, tauP=2)

    # initial run to settle system after reading file
    hoomd.run(10000)

    thermo = hoomd.analyze.log(filename=basename+"-thermo.dat",
                               quantities=['temperature', 'pressure',
                                           'potential_energy',
                                           'rotational_kinetic_energy',
                                           'translational_kinetic_energy'
                                          ],
                               period=1000)

    # Initialise dynamics quantities
    dyn = TimeDep2dRigid(system)
    CompRotDynamics().print_heading(basename+"-dyn.dat")
    tstep_init = system.get_metadata()['timestep']
    new_step = StepSize.PowerSteps(start=tstep_init)
    struct = [(new_step.next(), new_step, dyn)]
    timestep = tstep_init
    key_rate = 20000
    hoomd.dump.gsd(filename=basename+'.gsd',
                   period=10000000,
                   group=hoomd.group.all(),
                   overwrite=True,
                   truncate=True,
                  )

    while timestep < steps+tstep_init:
        index_min = struct.index(min(struct))
        next_step, step_iter, dyn = struct[index_min]
        timestep = next_step
        # print(timestep, file=open("timesteps.dat", 'a'))
        hoomd.run_upto(timestep)
        dyn.print_all(system, outfile=basename+"-dyn.dat")
        # dyn.print_data(system, outfile=basename+"-tr.dat")
        # dyn.print_corr_dist(system, outfile=basename+"-corr.dat")

        struct[index_min] = (step_iter.next(), step_iter, dyn)
        # Add new key frame when a run reaches 10000 steps
        if (timestep % key_rate == 0 and
                len(struct) < 5000 and
                len([s for s in struct if s[0] == timestep+1]) == 0):
            new_step = StepSize.PowerSteps(start=timestep)
            struct.append((new_step.next(), new_step, TimeDep2dRigid(system)))
    thermo.query('pressure')


