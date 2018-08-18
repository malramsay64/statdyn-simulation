#! /bin/sh
#
# recreate_intput.sh
# Copyright (C) 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
#
# This is a script to regenerate the Trimer-13.50-3.00.gsd file when there are breaking changes.

sdrun \
    --num-steps 50_000 \
    --pressure 13.50 \
    --temperature 0.40 \
    --lattice-lengths 20 20 \
    create \
    temp_create.gsd

sdrun \
    --num-steps 1_000_000 \
    --init-temp 0.40 \
    --pressure 13.50 \
    --temperature 3.00 \
    --lattice-lengths 20 20 \
    equil \
    --equil-type liquid \
    temp_create.gsd \
    Trimer-13.50-3.00.gsd

