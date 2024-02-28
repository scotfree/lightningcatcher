# Convenience and Utility functions for GMSH

# Installation & Interface
# ========================
# The binary can be aqcuired from here. Make sure to grab the correct version, ie, Mac with/out ARM
#     https://gmsh.info/#Download
#
#
# There are a few lovely `gmsh` python interfaces and toolsets out there, but we'll just use the one that comes with the software itself. So you should be fine with a:
# `pip install gmsh`
# ------------------------


import gmsh
import math

import subprocess, sys,os, shutil, math
import pandas
import numpy as np
from itertools import chain

pi = gmsh.pi
copy = gmsh.model.occ.copy
mirror = gmsh.model.occ.mirror
synchronize = gmsh.model.occ.synchronize
fuse = gmsh.model.occ.fuse
finalize = gmsh.finalize


default_config = {
    "meshsize_large": 5.0,           # uniform meshing for now
    "meshsize_small": 0.8,  
    "mesh_resolution": 0.2,
    "mesh_sizemin_scale": 5.0,
    "mesh_sizemax_scale": 25.0,
    "mesh_distmin_scale": 10.0,
    "mesh_distmax_scale": 50.0,
    "group_boundary_size": 6.0,
    "bounding_box_size": 15.0,    
    "full_fold_radians" : pi,     # usually pi, unless we fold loosely
    "wing_tilt_radians" : pi/2.0,  # wing angle with fuselage
    "camZoom":5.0,
    "camOffsetX":0.0,
    "camOffsetY":0.0,
    "paper_size_x" : 11.0,                 # sheet of paper height
    "paper_size_z": 4.25,                  # paper width
    "paper_thickness":0.1, 
    "body_tilt_radians" : pi/8.0,  # main fuselage fold angle
    "camX":30.0,                     # Camera positioning
    "camY":60.0,
    "camZ":0.0,
    "camOffsetZ":0.0
}
    


def image_filename(sim):
    return f"{sim['project_name']}/{sim['case_name']}.jpg"

# e2,e2['meshpath'], f"{e2['project_name']}/{e2['project_name']}-001.jpg"
# export files for mesh and image of mesh/model
def save_mesh(context, meshpath, imagepath):     
    if  not os.path.isdir(context['project_name']):
        print(f"Creating project dir: `{context['project_name']}`")
        print( os.mkdir(context['project_name']))
        
    if imagepath:
        gmsh.fltk.initialize()
        gmsh.write( imagepath)
        gmsh.fltk.finalize()
        context['imagepath'] = imagepath
    if meshpath:
        gmsh.option.setNumber('Mesh.SaveAll', 1)
        gmsh.write(meshpath)
        context['meshpath'] = meshpath
    gmsh.finalize()
    return context


# build a case - geometry, mesh, physical context, solver params. Maybe should be factored a little...

def set_camera(camX=30.0,camY=60.0,camZ=0.0,camZoomX=5.0,camZoomY=5.0,camZoomZ=5.0, rotX=0.0, rotY=0.0, rotZ=0.0):
    # sim.set_camera
    gmsh.option.setNumber("General.Trackball", 0)

    gmsh.option.setNumber("General.RotationX", camX) #187.3729455233209))
    gmsh.option.setNumber("General.RotationY", camY) #276.7547707531851))
    gmsh.option.setNumber("General.RotationZ", camZ) # 296.8673379509896))
    gmsh.option.setNumber("General.TranslationX", rotX) #187.3729455233209))
    gmsh.option.setNumber("General.TranslationY", rotY) #276.7547707531851))
    gmsh.option.setNumber("General.TranslationZ", rotZ) # 296.8673379509896))
    gmsh.option.setNumber("General.ScaleX", camZoomX) 
    gmsh.option.setNumber("General.ScaleY", camZoomY) 
    gmsh.option.setNumber("General.ScaleZ", camZoomZ)  
    gmsh.option.setNumber("General.Axes", 1)  

def initialize():
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

# SU2 Geometry

def points_to_normal(p1,p2,p3):
    # print(f"PTS again: p1: {p1},p2: {p2}, p3: {p3}")
    v1 = np.array(p1)-np.array(p2)
    v2 = np.array(p3)-np.array(p2)
    # print(f"v1: {v1} [{np.linalg.norm(v1)}] v2: {v2} [{np.linalg.norm(v2)} ] ")
    v3=np.cross(v1,v2)
    # print(f"v3: {v3} [{np.linalg.norm(v3)}]")
    return v3 / np.linalg.norm(v3)

def make_sphere(c, radius=1.0):
    gmsh.model.occ.addSphere(c[0], c[1], c[2], radius, tag=-1, angle3=gmsh.pi) #,angle1=-gmsh.pi/4, angle2=gmsh.pi/2, angle3=4.0)

# Build a triangular volume    
# This has a dependency on the order the points come in, which would be nice to fix...
def make_triangle(rp0,rp1,rp2,thickness, mesh_size=1.0, recenter=True):
    #print(f"making TRI: {rp0} | {rp1} | {rp2}...")
    p1 = gmsh.model.occ.addPoint(rp0[0], rp0[1], rp0[2], mesh_size)
    p2 = gmsh.model.occ.addPoint(rp1[0], rp1[1], rp1[2], mesh_size)
    p3 = gmsh.model.occ.addPoint(rp2[0], rp2[1], rp2[2], mesh_size)
    l1 = gmsh.model.occ.addLine(p1,p2)
    l2 = gmsh.model.occ.addLine(p2,p3)
    l3 = gmsh.model.occ.addLine(p3,p1)
    c1 = gmsh.model.occ.addCurveLoop([l1,l2,l3])
    s1 = gmsh.model.occ.addPlaneSurface([c1])
    nv = thickness*points_to_normal(rp0,rp1,rp2)
    v1 = gmsh.model.occ.extrude([(2,s1)], nv[0], nv[1], nv[2])
    newVolTag = [x for x in v1 if x[0]==3]
    if recenter:
         gmsh.model.occ.translate(newVolTag, 0, 0, -d/2.0)
    #
    return newVolTag

def make_sheet(rp0,rp1,rp2,rp3, thickness, mesh_size=1.0, recenter=True):
    # print(f"making TRI: {rp0} | {rp1} | {rp2}...")
    p1 = gmsh.model.occ.addPoint(rp0[0], rp0[1], rp0[2], mesh_size)
    p2 = gmsh.model.occ.addPoint(rp1[0], rp1[1], rp1[2], mesh_size)
    p3 = gmsh.model.occ.addPoint(rp2[0], rp2[1], rp2[2], mesh_size)
    p4 = gmsh.model.occ.addPoint(rp3[0], rp3[1], rp3[2], mesh_size)
    l1 = gmsh.model.occ.addLine(p1,p2)
    l2 = gmsh.model.occ.addLine(p2,p3)
    l3 = gmsh.model.occ.addLine(p3,p4)
    l4 = gmsh.model.occ.addLine(p4,p1)
    c1 = gmsh.model.occ.addCurveLoop([l1,l2,l3, l4])
    s1 = gmsh.model.occ.addPlaneSurface([c1])
    nv = thickness*points_to_normal(rp1,rp2,rp3)
    v1 = gmsh.model.occ.extrude([(2,s1)], nv[0],nv[1],nv[2])
    newVolTag = [x for x in v1 if x[0]==3]
    if recenter:
         gmsh.model.occ.translate(newVolTag, -0.5*nv[0], -0.5*nv[1],-0.5*nv[2])
    #
    return newVolTag

def split(vols, sp1,sp2, p,d, remove_tool=True, remove_object=True):
    
    # build a box to contain the stuff we'll fold/rotate:
    extend = 0.2 * (np.array(sp2) - np.array(sp1) )
    n1 = points_to_normal(sp1, sp2, p)
    sp1a = np.array(sp1) + np.array(n1)
    n2 = points_to_normal(sp1, sp1a, sp2)
    
    np1 = np.array(sp1) - extend + 5.0*n2
    np2 =  np.array(sp2) + extend + 5.0*n2
    # print(f"N to sheet {n1} \nOffset: {sp1a} \nN to cut: {n2}\nMaking: {sp1}\n{np1}\n{np2}\n{sp2}")
    
    tool = make_sheet(sp1-extend, np1, np2, sp2+extend, 10.0, recenter=True)
    # gmsh.view.option.setColor(tool, 'blsh', 255,100,130, a=255)
    #tool = make_triangle(sp1,p,sp2,d*2.0)
    gmsh.model.occ.synchronize()
    # negate
    remain, v1b = gmsh.model.occ.cut(vols,tool, removeObject=False, removeTool=False)
    # intersect
    stuff, v1b = gmsh.model.occ.intersect(vols,tool,removeTool=remove_tool, removeObject=remove_object)
    
    return stuff, remain

def fold(vols, fp1, fp2, p, a, do_rotate=True, remove_tool=True, fuse=True, remove_object=True):
    d= 5.0
    moved, fixed = split(vols, fp1, fp2, p, d, remove_tool=remove_tool, remove_object=remove_object )
    
    if do_rotate:
        gmsh.model.occ.rotate(moved, fp1[0],fp1[1],fp1[2],fp1[0]-fp2[0],fp1[1]-fp2[1],fp1[2]-fp2[2],a )
    if fuse:
        folded, v1b = gmsh.model.occ.fuse(moved, fixed)
        return folded
    return moved+fixed

def group_surfaces_radially(center=[0,0,0], threshold=6.0 ):
    near_surfaces = []
    far_surfaces = []

    for s in gmsh.model.occ.getEntities(dim=2):
        com = gmsh.model.occ.getCenterOfMass(s[0], s[1])
        if np.allclose(com, center,atol=threshold):
            near_surfaces.append(s[1])
        else:
            far_surfaces.append(s[1])
    return near_surfaces, far_surfaces    
    
def label_surfaces(labels, groups):
    # Set up physical groups for boundaries and markers in Simulation   
    for i,l,g in zip(range(len(groups)),labels, groups):
        # print(f"adding Group {i+1} '{l}': [{g}]")
        res=gmsh.model.addPhysicalGroup(2,  g, i+1)
        gmsh.model.setPhysicalName(2, i+1, l)       
        # print(f"res: {res}")


        
def add_mesh(config):
    # Create a big bounding box around our object:
    volume = config['geometry_index']
    # print(f"Volume: {volume}\nConfig: {config}")
    boundary_scale_z = 2.0
    boundary_scale_y = 4.0
    c,m1 = config,volume
    box = make_sheet([-1*c['paper_size_x'], 0.0, -boundary_scale_z*c['paper_size_z']],
                [3*c['paper_size_x'], 0.0, -boundary_scale_z*c['paper_size_z']],
                [3*c['paper_size_x'], 0.0, boundary_scale_z*c['paper_size_z']],
                [-1*c['paper_size_x'], 0.0, boundary_scale_z*c['paper_size_z']], 
                c['bounding_box_size'])#     boundary_scale_y*c['paper_size_y'])
    v1, v1h = gmsh.model.occ.cut(box,m1)
    gmsh.model.occ.synchronize()
    model_center = [c['paper_size_x']/2,0,0]
    res = c['mesh_resolution']
    near_group, far_group = group_surfaces_radially(center=model_center, threshold=c['group_boundary_size'] )

    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "FacesList", near_group)        
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", c['mesh_sizemin_scale']*res)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", c['mesh_sizemax_scale']*res)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", c['mesh_distmin_scale']*res)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", c['mesh_distmax_scale']*res)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.model.occ.synchronize()  
    # if define_surface_groups:        
    label_surfaces(['Plane', 'Walls'], [near_group, far_group])
    gmsh.model.occ.synchronize()       
    c['meshpath'] = f"{c['project_name']}/{c['case_name']}.su2"
    # save_mesh(c,c['meshpath'], c['imagepath'])
    return c