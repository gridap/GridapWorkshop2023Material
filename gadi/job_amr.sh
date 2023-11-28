#!/bin/bash
#PBS -P vp91
#PBS -q normal
#PBS -l walltime=00:20:00
#PBS -l ncpus=16
#PBS -l mem=64gb
#PBS -N gw_amr
#PBS -l wd

source $PBS_O_WORKDIR/modules.sh

mpiexec -n 16 julia --project=$PBS_O_WORKDIR -e'
  using GadiTutorial;
  main_amr(;nprocs=16,nrefs=4,num_amr_steps=6)
'
