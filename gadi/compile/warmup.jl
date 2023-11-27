
using GadiTutorial
main_poisson(;nprocs=(1,1))
main_ins(;nprocs=1,mesh="perforated_plate_tiny.msh",T=0.02)
main_ins(;nprocs=1,mesh="perforated_slab_tiny.msh",T=0.02)
main_amr(;nprocs=1,nrefs=1,num_amr_steps=1)