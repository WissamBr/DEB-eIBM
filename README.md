# DEB-eIBM
This is a repository with the code for the paper "Environment-dependent population dynamics emerging from dynamic energy budgets and individual-scale movement behaviour". It includes the source functions, example scripts and simulated data for the DEB-εIBM: a spatially-explicit individual-based model for physiologically-structured population dynamics.

### src
The core functions of the model are found under "src/" and consist of the "Individual" datatype and a module for the functions implementing the model processes. These can be loaded into a script to run the model.

### scripts
Under "scripts/", you can find an example script to run the model. This script consists of the parametrization and calls to initialize and run the model as well as save the simulation data to "data/". The "run_debeibm.py" script currently holds the parametrization coinciding with the case study shown in the paper. It also has a notebook "read_simulation_data.ipynb" that reads the simulation data and provides example visualizations thereof.

### data
The resulting simulation data from the script "run_debeibm.py" are available in "data/", from where "read_simulation_data.ipynb" can read them.
