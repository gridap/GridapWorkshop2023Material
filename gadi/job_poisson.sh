#!/bin/bash
#PBS -P vp91
#PBS -q normal
#PBS -l walltime=00:20:00
#PBS -l ncpus=4
#PBS -l mem=16gb
#PBS -N gw_poisson
#PBS -l wd

source $PBS_O_WORKDIR/modules.sh

mpiexec -n 4 julia --project=$PBS_O_WORKDIR -J$PBS_O_WORKDIR/GadiTutorial.so -e'
  using GadiTutorial;
  main_poisson(;nprocs=(2,2),ncells=(100,100))
'
