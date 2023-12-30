# Convenience and Utility functions for GMSH

import gmsh
import math

import subprocess, sys,os, shutil, math
import pandas
import numpy as np
from itertools import chain

import lightningcatcher_gmsh as geometry

# export files for mesh and image of mesh/model

# SU2 Case setup
# build a config for a Solver to run a case:        
def save_config(context, base_config_path='base.cfg'):
    config_path=os.path.join(context['project_name'], f"{context['case_name']}.cfg")
    print(f"writing CNF: '{config_path}'\nCONTEXT: {context}\n")
    project_config=open(base_config_path).read().format(**context)
    
    with open(config_path,'w') as f:
        f.write(project_config)




### SU2 Run Simulation

def run_simulation(config, as_row=True, extract=['lift', 'drag']):

    config['surface_flow_path'] = f"{config['case_name']}-surface_flow.csv"
    subprocess.run(["/Users/scot/Projects/SU2/bin/SU2_CFD", config['simconfpath']])
    return ['surface_flow_path']
    
extractors = {
    'lift': lambda sdf: float(sdf[['Momentum_y']].sum()),
    'drag': lambda sdf: float(sdf[['Momentum_x']].sum()),
    #'time': lambda sdf, cvg: float(cvg[['Time']].max())
}

    
def extract_simulation_values(config, extracts=['lift', 'drag']):
    sdf=pandas.read_csv( os.path.join(config['project_name'],config['surface_flow_path']))
    # cvg=pandas.read_csv( os.path.join(config['project_name'],config['convergence_history_path']))
    for extract_name in extracts:
        config[extract_name] = extractors[extract_name](sdf)
    return extracts

# SU2 Execution


def single_simulation(base_conf, build_model, do_meshing=True, do_run=True, do_extract=[],
                      define_surface_groups=True, show_group_boundary=False, 
                      interactive=False, 
                      as_row=False, new_columns=['meshpath', 'imagepath'],
                      **kwargs):
    c = dict(chain(base_conf.items(), kwargs.items()))
    c['imagepath']=f"{c['project_name']}/{c['case_name']}.jpg"
    c['meshpath'] = f"{c['project_name']}/{c['case_name']}.su2"
    c['simconfpath'] = f"{c['project_name']}/{c['case_name']}.cfg"
    
    m1 = build_model(do_meshing=do_meshing, **c)
    
    gmsh.model.occ.synchronize()
    if do_meshing:
        boundary_scale_z = 2.0
        boundary_scale_y = 4.0
        box = geometry.make_sheet([-1*c['x_size'], 0.0, -boundary_scale_z*c['y_size']],
                    [3*c['x_size'], 0.0, -boundary_scale_z*c['y_size']],
                    [3*c['x_size'], 0.0, boundary_scale_z*c['y_size']],
                    [-1*c['x_size'], 0.0, boundary_scale_z*c['y_size']], 
                    c['box_size'])#     boundary_scale_y*c['y_size'])
        v1, v1h = gmsh.model.occ.cut(box,m1)
        gmsh.model.occ.synchronize()
        # Assign a mesh size to all the points:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), c['meshsize_large'])
        gmsh.model.mesh.generate(3)

    model_center = [c['x_size']/2,0,0]
    if show_group_boundary:
        geometry.make_sphere(model_center, c['group_boundary_size'])
    gmsh.model.occ.synchronize()
    
    if define_surface_groups:
        near_group, far_group = geometry.group_surfaces_radially(center=model_center, threshold=c['group_boundary_size'] )
        geometry.label_surfaces(['Plane', 'Walls'], [near_group, far_group])
        gmsh.model.occ.synchronize()
    
    geometry.save_mesh(c,c['meshpath'], c['imagepath'])
    save_config(context=c)
    if interactive:
        gmsh.fltk.run()
    gmsh.finalize()
    
    if do_run:
        new_columns += run_simulation(c)
        if do_extract:
            new_columns += extract_simulation_values(c, do_extract)
    if as_row:
        return pandas.Series([run_dict[c] for c in new_columns])
    else:
        return c

