#!/usr/bin/env python
""" Calculate the dynamic properties for the Trimer molecule
for a series of temperatures"""

# import os.path
import hoomd
import dynamics

PRESS = 13.5
# Holds tuples of the temperature and number of steps
# to iterate through
STEPS = 10000000
TEMPERATURES = [(5.00, 1*STEPS),\
                (4.00, 1*STEPS),\
                (3.50, 2*STEPS),\
                (3.00, 2*STEPS),\
                (2.50, 4*STEPS),\
                (2.00, 4*STEPS),\
                (1.80, 8*STEPS),\
                (1.60, 8*STEPS),\
                (1.50, 16*STEPS),\
                (1.40, 16*STEPS),\
                (1.35, 32*STEPS),\
                (1.30, 32*STEPS),\
                (1.25, 64*STEPS),\
                (1.20, 128*STEPS),\
                (1.15, 128*STEPS),\
                (1.10, 256*STEPS),\
                (1.05, 512*STEPS),\
               ]


if __name__ == "__main__":
    hoomd.context.initialize()

    for temp, steps in TEMPERATURES:
        input_file = "Trimer-{press:.2f}-{temp:.2f}.gsd"\
                .format(press=PRESS, temp=temp)
        dynamics.compute_dynamics(input_file, temp, PRESS, steps)