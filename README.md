# GridapWorkshop2023Material

## Before the workshop

### Required software

Before being able to work on the workshop material, you will need to instal the following software:

- Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), the version control system we will use. **For Windows users, we strongly recommend installing [git for Windows](https://gitforwindows.org/). This will also install a bash shell, which will allow you to follow the rest of the installation instructions verbatim.**
- Install [Julia](https://julialang.org/downloads/platform/).
- Install [VSCode](https://code.visualstudio.com/download). To get the fully-interactive REPL experience, you will also need to install the [Julia extension for VSCode](https://code.visualstudio.com/docs/languages/julia).
- Install [Paraview](https://www.paraview.org/download/), or any other software that can read and display `.vtk` files.
- Finally, you will need an **ssh** client to connect to Gadi. Generally, every modern OS should have one installed by default. To check if you have one, open a terminal and type `ssh`. If a message like `usage: ssh ...` appears, you are good to go.

### Getting the workshop material

To get the workshop material (available [here](https://github.com/gridap/GridapWorkshop2023Material)) you can either download it as a zip file or clone (and optionally fork) the repository using git. We strongly recommend the latter.

If your system has an ssh client, you can clone the repository by using

```bash
  git clone git@github.com:gridap/GridapWorkshop2023Material.git
```

Alternative methods to clone the repository can be found [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

### Setting up the environment on your local computer

Move into the newly cloned repository and start Julia by typing

```bash
  julia --project=.
```

Then, press `]` to enter the package manager and run

```julia
  (GridapWorkshop) pkg> instantiate
```

to install and precompile all the packages needed for the workshop. This may take a while.

More information on Julia Environments can be found [here](https://pkgdocs.julialang.org/v1/environments/).

### Setting up the environment on Gadi

First, we will need to log into Gadi. You will have received an email with your username and password (which you should change). If your username is `aaa777`, you can connect to Gadi by typing

```bash
ssh aaa777@gadi.nci.org.au
```

When prompted, enter your password. You should now be logged into Gadi and located in your home directory. We would also recommend setting up passwordless ssh access to Gadi (but it is not required).

In addition to your home directory, you have access to a scratch space, a project-wide shared filesystem that is optimized for parallel access. This is where we will be working during the workshop. Start by creating a personal folder within the project's scratch space and linking it back to your home directory:

```bash
  mkdir /scratch/vp91/$USER
  ln -s /scratch/vp91/$USER $HOME/scratch
```

Move into your newly created scratch directory and clone the workshop repository as described above. Once the workshop repository is cloned, load the Gadi environment by running

```bash
  source modules.sh
```

The script `modules.sh` has several purposes, and needs to be sourced every time you log in. It loads the julia module, the Intel-MPI modules, and sets up some environment variables we will need.

Next, repeat the steps described in the previous section to setup the Julia environment in serial.
In addition, you will need to setup the environment for parallel. To do so, run from the workshop directory

```bash
  julia --project=. -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
  julia --project=. -e 'using Pkg; Pkg.build()'
```

The first line sets up MPI.jl to work with the Intel-MPI binaries installed on Gadi instead of the julia-specific artifacts. The second line does the same for GridapPETSc and GridapP4est, which are the Gridap wrappers for PETSc and P4est, respectively.

### Creating a system image

