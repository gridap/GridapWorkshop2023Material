#!/bin/bash
#PBS -P vt91
#PBS -q normal
#PBS -l walltime=00:30:00
#PBS -l ncpus=1
#PBS -l mem=4gb
#PBS -N gw_build
#PBS -l wd

source $PBS_O_WORKDIR/modules.sh

julia --project=$PBS_O_WORKDIR $PBS_O_WORKDIR/compile/compile.jl
