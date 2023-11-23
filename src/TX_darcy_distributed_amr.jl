# The purpose of this tutorial is two-fold: 
#  - To illustrate how to solve a system of PDEs where one of the unknowns
#    lives in $H({\rm div};\Omega)$ (as opposed to $H^1(\Omega)$), thus requiring 
#    the construction of a div-conforming FE space (as opposed to a grad-conforming FE space).
#  - To showcase the *dynamic* Adaptive Mesh Refinement (AMR) parallel distributed-memory 
#    capabilities provided by `GridapP4est.jl`. To this end, we will solve a system of PDEs featuring a 
#    multi-scale solution where the use of these techniques is particularly beneficial from a computational
#    point of view.

# ## Problem statement (strong form)
#
# We consider as a model problem the so-called Darcy equations, which can be used 
# as a physical model of fluid flow in porous media. The PDE problem reads: find the fluid velocity $u$, and the fluid pressure $p$ such that:
#
# $$
#    \left\lbrace
#    \begin{aligned}
#       \Kappa^{-1} u + \nabla p = {\bf 0}  \ &\text{in} \ \Omega,\\
#       \nabla \cdot u = f  \ &\text{in} \ \Omega,\\
#       p = g \ &\text{on}\ \Gamma_{\rm N},\\
#       u \cdot n = h \ &\text{on}\  \Gamma_{\rm D},\\
#    \end{aligned}
#    \right.
# $$
#
# with $n$ being the outwards unit normal vector to the boundary $\partial\Omega$, and $\Kappa$ the so-called hydraulic conductivity tensor.  The first equation is known as Darcy’s law and was formulated by Henry Darcy in 1856; the second equation is the mass conservation equation.

# In this particular tutorial, for simplicity, we consider the unit square $\Omega \doteq (0,1)^2$ as the computational domain. Besides, we consider pure Neumann boundary conditions, that is, the Neumann boundary $\Gamma_{\rm N}$ is the full boundary of $\Omega$, and $\Gamma_{\rm D}$ is the empty set (i.e., no Dirichlet boundary condition). Finally, we consider the hydraulic conductivity tensor $\Kappa$ to be just the identity tensor. In any case, we stress that the general version of the equations stated above are also supported by Gridap.

# The source term $f$ in the mass conservation equation and the Neumann data $g$ are chosen such that the exact (manufactured) fluid pressure is:
# $$
# p_{\rm exact}(\boldsymbol{x}) := \mathrm{arctan}(\alpha(\sqrt{(\boldsymbol{x}-\boldsymbol{x}_c)\cdot(\boldsymbol{x}-\boldsymbol{x}_c)}-r)).
# $$
# Besides, using Darcy's law (i.e., first equation of the system above), we manufacture $u_{\rm exact}=- \Kappa^{-1} \nabla p_{\rm exact}$. 

# The solution $p_{\rm exact}(\boldsymbol{x})$ has a sharp circular/spherical wave front of radius $r$ centered at $\boldsymbol{x}_c$. For example, for the combination of parameter values $\alpha=200$, $r=0.7$, and $\boldsymbol{x}_c=(-0.05, -0.05)$, $p_{\rm exact}(\boldsymbol{x})$ looks as in the picture below:
# ![](../figures/darcy_amr/circular_sharp_wave_2d_scaled.png)
# As consequence of the multi-scale features of this solution, uniform mesh refinement techniques can only reduce the error at a very slow pace with increasing mesh resolution, and thus are very computationally inefficient. 

# ## Problem statement (weak form)

# We denote by $H(\mathrm{div};\Omega)$ the space of vector-valued fields in $\Omega$, whose components and divergence are in $L^2(\Omega)$. With these notation, the weak form reads of our problem: find $(u,p)\in H(\mathrm{div};\Omega)\times L^2(\Omega)$ such that $a((u,p),(v,q)) = b(v,q)$ for all $(v,q)\in H(\mathrm{div};\Omega)\times L^2(\Omega)$, where
#
# $$
# \begin{aligned}
# a((u,p),(v,q)) &\doteq \int_{\Omega}  v \cdot u \ {\rm d}\Omega - \int_{\Omega} (\nabla \cdot v)\ p \ {\rm d}\Omega + \int_{\Omega} q\ (\nabla \cdot u) \ {\rm d}\Omega,\\
# b(v,q) &\doteq \int_{\Omega} q\ f \ {\rm  d}\Omega - \int_{\Gamma_{\rm N}} (v\cdot n)\ g  \ {\rm  d}\Gamma.
# \end{aligned}
# $$

# This weak form was obtained from the strong form as usual, i.e., multiplication by suitable test functions, and integration by parts in order to transfer derivatives from trial to test functions to reduce the regularity constraints on the weak solution.

#  ## Numerical scheme
#
# In this tutorial, we use the div-conforming Raviart-Thomas (RT) space of polynomial order $k\geq0$ for the fluid velocity approximation, and a discontinuous space of cell-wise polynomials of order $k$ in each spatial dimension (denoted as $Q_k$) for the fluid pressure approximation (see [1] for specific details). This pair of finite element spaces form a so-called *discrete inf-sup stable pair*. This mathematical property guarantees that the discrete problem is *well-posed*, i.e., it has a unique solution.  

# ## Adaptive Mesh Refinement (AMR) with forest-of-trees 

# In this tutorial we leverage a more clever/efficient domain discretization approach provided by [`GridapP4est.jl`](https://github.com/gridap/GridapP4est.jl). In particular, we will used a particular Gridap's `DiscreteModel` that efficiently supports *dynamic* $h$-adaptivity techniques (a.k.a. AMR), i.e., the ability of the mesh to be refined *in the course of the simulation* in those regions of the domain that present a complex behaviour (e.g., the internal layer in the case of our problem at hand), and to be coarsened in those areas where essentially nothing relevant happens (e.g., those areas away from the internal layer).

# In order to support AMR techniques, `GridapP4est.jl` relies on the so-called forest-of-trees approach for efficient mesh generation and adaptation as provided by the `p4est` library [2]. Forest-of-trees can be seen as a two-level decomposition of $\Omega$, referred to as macro and micro level, resp. In the macro level, we have the so-called coarse mesh, i.e., a *conforming* partition $\mathcal{C}_h$ of $\Omega$. For efficiency reasons, $\mathcal{C}_h$ should be as coarse as possible, but it should also keep the geometrical discretization error within tolerable margins. For complex domains, $\mathcal{C}_h$ is usually generated by an unstructured mesh generator, and then imported into the program using, e.g., using `GridapGmsh`. For simple domains, such as boxes, a single coarse cell is sufficient to resolve the geometry of $\Omega$. On the other hand, in the micro level, each of the cells of $\mathcal{C}_h$ becomes the root of an adaptive tree that can be subdivided arbitrarily (i.e., recursively refined) into finer cells.

# In the case of quadrilateral (2D) or hexahedral (3D) adaptive meshes, the recursive application of the standard isotropic 1:4 (2D) and 1:8 (3D) refinement rule to the coarse mesh cells (i.e., to the adaptive tree roots) leads to adaptive trees that are referred to as quadtrees and octrees, resp., and the data structure resulting from patching them together is called *forest-of-quadtrees* and *-octrees*, resp., although the latter term is typically employed in either case. The figure below shows a forest-of-quadtrees mesh with two quadtrees (i.e., $|\mathcal{C}_h|=2$):

# ![](../figures/darcy_amr/forest_of_trees_partition.png)

# Tree-based meshes provide multi-resolution capability by local adaptation. The cells in the mesh (i.e., the leaves of the adaptive trees) might be located at different refinement level.  However, these meshes are (potentially)  *non-conforming*, i.e., they contain the so-called *hanging* vertices, edges, and faces.  These occur at the interface of neighboring cells with different refinement levels. Mesh non-conformity introduces additional complexity in the implementation of conforming finite element formulations [3].  Despite the aforementioned, we note the following. First, the degree of implementation complexity is significantly reduced by enforcing the so-called *2:1 balance* constraint, i.e., adjacent cells may differ at most by a single level of refinement; the $h$-adaptive triangulation in `GridapP4est.jl` always satisfies this constraint. Second, `Gridap` is written such that it is entirely responsible for handling such complexity. As demonstrated in this tutorial, library users are not aware of mesh non-conformity when coding the weak form of the finite element formulation at hand.

# ## Parallelization

# It order to scale finite element simulations to large core counts, the adaptive mesh must be partitioned (distributed) among the parallel tasks such that each of these only holds a local portion of the global mesh. (The same requirement applies to the rest of data structures in the finite element simulation pipeline, i.e., finite element space, linear system, solver, etc.) Besides, as the solution might exhibit highly localized features, dynamic mesh adaptation can result in an unacceptable amount of load imbalance. Thus, it urges that the adaptive mesh data structure supports *dynamic load-balancing*, i.e., that it can be re-distributed among the parallel processes in the course of the simulation.

# Modern forest-of-trees manipulation engines, such as `p4est`, provide a scalable, linear runtime solution to the mesh (re-)partitioning problem based on the exploitation of Space-Filling-Curves (SFCs). SFCs provide a natural means to assign an ordering of the forest-of-trees leaves, which is exploited for the parallel arrangement of data. For example, in `GridapP4est.jl`, the forest-of-octrees leaves are arranged in a global one-dimensional data array in increasing Morton index ordering. This ordering corresponds geometrically with the traversal of a $z$-shaped SFC (a.k.a. Morton SFC). This approach allows for *fast dynamic repartitioning*. A partition of the mesh is simply generated by dividing the leaves in the linear ordering induced by the SFCs into as many equally-sized segments as parallel tasks involved in the computation.

# As an illustration, the figure below shows a 2:1 balanced forest-of-quadtrees mesh with two quadtrees (i.e., $|\mathcal{C}_h|=2$) distributed among two processors, 1:4 refinement and the Morton SFC. Local cells are depicted with continuous boundary lines, while those in the ghost layer with dashed ones.

# ![](../figures/darcy_amr/forest_of_trees.png)

# `GridapP4est.jl` reconstructs the local portion of the mesh corresponding to each parallel task from the distributed forest-of-octrees that the `p4est` library handles internally. These local portions are illustrated in the figure above when the forest-of-octrees is distributed among two processors. The local portion of each task is composed by a set of cells that it owns, i.e., the *local cells* of the task, and a set of off-processor cells (owned by remote processors) which are in touch with its local cells, i.e., the *ghost cells* of the task. This overlapped mesh partition is used by the library to exchange data among nearest neighbours, and to glue together the global degrees-of-freedom of the FE space which are sitting on the interface among subdomains, as required in order to construct finite element spaces for conforming finite element formulations in a distributed setting.


# ## Homework
#
# Extend the present tutorial to 3D. 
#
#

# ## References
#
# [1] F. Brezzi and M. Fortin. *Mixed and hybrid finite element methods*. Springer-Verlag, 1991.
# [2] C. Burstedde, L. C. Wilcox, O. Ghattas. *p4est: Scalable Algorithms for Parallel Adaptive Mesh Refinement on Forests of Octrees*. SIAM Journal on Scientific Computing 33 (3) (2011).
# [3] S. Badia, A. F. Martin, E. Neiva, and F. Verdugo. A Generic Finite Element Framework on Parallel Tree-Based Adaptive Meshes. SIAM Journal on Scientific Computing 46 (2) (2020).
 