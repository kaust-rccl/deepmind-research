#!/bin/bash --login

# create the conda environment
export ENV_PREFIX=$PWD/alphafold_casp13/env
conda env create --prefix $ENV_PREFIX --file alphafold_casp13/environment.yml --force
conda activate $ENV_PREFIX
. alphafold_casp13/postBuild
