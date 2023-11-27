#!/bin/bash
#PBS -P vp91
#PBS -q normal
#PBS -l walltime=01:00:00
#PBS -l ncpus=1
#PBS -l mem=16gb
#PBS -N gw_build
#PBS -l wd

source $PBS_O_WORKDIR/modules.sh

julia --project=$PBS_O_WORKDIR $PBS_O_WORKDIR/compile/compile.jl
