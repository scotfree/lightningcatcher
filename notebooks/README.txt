# Lightningcatcher Notebooks and Tooling

## Jupyter Notebooks

We use plain Jupyter notebooks for much of this.
```
pip install jupyter
# missing bit here to set the password
jupyter notebook
```

If you're running locally, that's about it. For running remote, theres a little work to do for (secure) access.

And if you're using a hosted notebook, like `sagemaker` or `google colab`, you may not be able to run the parts of these notebooks that involve running third party executables.

## Other software
We assume the usual:
* pandas
* numpy
* 

## Computational Fluid Dynamics

* CFD_DS_Toolchan.ipyn: Notebook example of parameterized geometry and CFD analysis
* lightningcatcher_su2.py: install notes and python methods for using SU2 as our CFD engine
* lightningcatcher_gmsh.py: install notes and python methods for `gmsh` as CAD and Meshing
* lightningcatcher_visualization.py: install notes and python methods for `paraview` as our visualization engine
* base.cfg: template CFD sim job configuration
* para.py: template script for generating vis frame for a single completed sim

To get started, do the simple installation for the engines you'll use for CFD, Geometry, meshing, and Vis.
