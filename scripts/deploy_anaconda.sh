#! /bin/sh
#
# deploy_anaconda.sh
# Copyright (C) 2017 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
#

conda config --set anaconda_upload yes
source activate base
conda build . --token "$CONDA_UPLOAD_TOKEN" --user "$CONDA_USER"
