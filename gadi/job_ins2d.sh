#!/bin/bash
#PBS -P vp91
#PBS -q normal
#PBS -l walltime=00:20:00
#PBS -l ncpus=8
#PBS -l mem=190gb
#PBS -N gw_ins2d
#PBS -l wd

source $PBS_O_WORKDIR/modules.sh

mpiexec -n 8 julia --project=$PBS_O_WORKDIR -e'
  using GadiTutorial;
  main_ins(;nprocs=8,mesh="perforated_plate.msh",T=0.5)
'
