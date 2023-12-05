# GMSH meshing

import gmsh
import math

import gmsh

import subprocess, sys,os, shutil, math
import pandas
import numpy as np



# export files for mesh and image of mesh/model
def save_mesh(context, meshpath, gifpath):  
    if  not os.path.isdir(context['project_dir']):
        print(f"Creating project dir: `{context['project_dir']}`")
        print( os.mkdir(context['project_dir']))
        
    if gifpath:
        gmsh.fltk.initialize()
        gmsh.write(os.path.join(context['project_dir'], gifpath))
        gmsh.fltk.finalize()        
    if meshpath:
        gmsh.option.setNumber('Mesh.SaveAll', 1)
        gmsh.write(os.path.join(context['project_dir'],meshpath))

# SU2 Case setup
# build a config for a Solver to run a case:        
def save_config(context, base_config_path='base.cfg'):
    config_path=os.path.join(context['project_dir'], f"{context['project_name']}.cfg")
    project_config=open(base_config_path).read().format(**context)
    print(f"writing CNF: '{config_path}'")
    with open(config_path,'w') as f:
        f.write(project_config)

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

def init():
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    

    
def build_parametrized_plane(
        fuselage_rad=gmsh.pi/16.0, 
        wing_rad=gmsh.pi/4.0, wedge_rad=gmsh.pi/16.0, 
        p_height=11.0, p_width=8.5, p_thickness=0.2 ):
    # x coordinate of wing fold
    clip_wing=True
    fuselage_height = p_height* math.sin(fuselage_rad)

    # right vertical wedge; build then tilt; RF Right Fuselage [a,b,c]
    rfb = [0.0, 0.0, 0.0] # back corner of fuselage; origin
    rfa = [0.0, fuselage_height, 0.0]
    rfc = [p_height, 0.0, 0.0] # tip of front
    
    # right wing, vertical
    rwb = rfa.copy()
    rwa = rwb.copy()
    rwa[1] += p_width/2.0 - fuselage_height

    rwc = rfc.copy()

    rf=make_triangle(rfa, rfb, rfc, 0.2)
    rw=make_triangle(rwa, rwb, rwc, 0.2)    
    
    # trim wing
    if clip_wing:
        cutbox=gmsh.model.occ.addBox(
           rwa[0], p_width/2.0-rfa[1], rfa[2]-5*p_thickness,
           p_height,   p_width, 10*p_thickness )   
        rw, v1b = gmsh.model.occ.cut(rw,[(3,cutbox)], removeTool=True)

    #rotate wing along top edge of fuselage
    gmsh.model.occ.rotate(rw,rfa[0],rfa[1],rfa[2],rfc[0]-rfa[0],rfc[1]-rfa[1],rfc[2]-rfa[2],wing_rad)
    
    # merge fuselage and wing
    rh, rhb = gmsh.model.occ.fuse(rf,rw)
    
    #rotate halfplane along bottom of plane
    gmsh.model.occ.rotate(rh,rfb[0],rfb[1],rfb[2],rfc[0]-rfb[0],rfc[1]-rfb[1],rfc[2]-rfb[2],wedge_rad)
        
    lf=make_triangle(rfa, rfb, rfc, 0.2)
    lw=make_triangle(rwa, rwb, rwc, 0.2)
    
    if clip_wing:
        cutbox2=gmsh.model.occ.addBox(
            rwa[0], p_width/2.0-rfa[1], rfa[2]-5*p_thickness,
           p_height,   p_width, 10.0*p_thickness )   
        lw, v1b = gmsh.model.occ.cut(lw,[(3,cutbox2)])
    
    #rotate wing along top edge of fuselage
    gmsh.model.occ.rotate(lw,rfa[0],rfa[1],rfa[2],rfc[0]-rfa[0],rfc[1]-rfa[1],rfc[2]-rfa[2],-wing_rad)
    
    # merge fuselage and wing
    #print(f"Fuselage: {rf} Wing: {rw}")
    lh, lhb = gmsh.model.occ.fuse(lf,lw)
    
    #rotate halfplane along bottom of plane
    gmsh.model.occ.rotate(lh,rfb[0],rfb[1],rfb[2],rfc[0]-rfb[0],rfc[1]-rfb[1],rfc[2]-rfb[2],-wedge_rad)
    
    p1, p1b =  gmsh.model.occ.fuse(lh,rh)
    
    # build skybox
    b1=gmsh.model.occ.addBox(-4*p_height,-4*p_width,-4*p_width, 8*p_height, 8*p_width, 8*p_width)

    # create negative volume around plane model
    v1, v1b = gmsh.model.occ.cut([(3,b1)],p1)

    gmsh.model.occ.synchronize()

    
    
# platform independent parametrized paper airplane model    
def create_sim(project_name, fuselage_rad=gmsh.pi/16.0, wing_rad=gmsh.pi/4.0, wedge_rad=gmsh.pi/16.0, 
               meshing=True, 
                xshift=0, camX=30.0,camY=60.0,camZ=0.0,camZoom=5.0,rotX=0.0,rotY=0.0,rotZ=0.0,
                meshpath=None, gifpath=None, popup=True, angle_of_attack=0.0, meshsize_large=5.0,
              clip_wing=True, case_index=None, case_name=None, project_dir=None):

    # Setup Geometry
    plane_geometry_params = {
        'p_height' : 11.0,
        'p_width' : 8.5,
        'p_thickness' :  0.2,
        'fuselage_rad': fuselage_rad}
 
    init()
    gmsh.model.add(project_name)
    set_camera(camX, camY, camZ, camZoom, camZoom, camZoom,rotX, rotY, rotZ)

    # Actual model! Not platform specific
    # phi = fuselage_rad # wedge_radians #gmsh.pi/16.0

    
    plane_model=build_parametrized_plane(**plane_geometry_params)
    
    if meshing:
        lcar_big = meshsize_large
        lcar2 = 1.0
        lcar3 = .055

        # Assign a mesh size to all the points:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lcar_big)
        #gmsh.model.mesh.setSize([(2,s) for s in wall_surfaces], lcar1)
        gmsh.model.mesh.generate(3)

    plane_surfaces = []
    wall_surfaces = []

    p_height=plane_geometry_params['p_height']
    surfaces = gmsh.model.occ.getEntities(dim=2)
    for s in surfaces:
        com = gmsh.model.occ.getCenterOfMass(s[0], s[1])
        if np.allclose(com, [p_height/2.0, 0, 0],atol=6.0):
            plane_surfaces.append(s[1])
        else:
            wall_surfaces.append(s[1])

    # Set up physical groups for boundaries and markers in Simulation        
    wall_marker = 1
    plane_marker = 2
    gmsh.model.addPhysicalGroup(2, wall_surfaces, wall_marker)
    gmsh.model.setPhysicalName(2, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(2, plane_surfaces, plane_marker)
    gmsh.model.setPhysicalName(2, plane_marker, "Plane")

    save_mesh(locals(),meshpath, gifpath)
    save_config(context=locals())
    if popup:
        gmsh.fltk.run()
    gmsh.finalize()

# SU2 Geometry
# Build a triangular volume    
def make_triangle(rp0,rp1,rp2,d, mesh_size=1.0, recenter=True):
    #print(f"making TRI: {rp0} | {rp1} | {rp2}...")
    p1 = gmsh.model.occ.addPoint(rp0[0], rp0[1], rp0[2], mesh_size)
    p2 = gmsh.model.occ.addPoint(rp1[0], rp1[1], rp1[2], mesh_size)
    p3 = gmsh.model.occ.addPoint(rp2[0], rp2[1], rp2[2], mesh_size)
    l1 = gmsh.model.occ.addLine(p1,p2)
    l2 = gmsh.model.occ.addLine(p2,p3)
    l3 = gmsh.model.occ.addLine(p3,p1)
    c1 = gmsh.model.occ.addCurveLoop([l1,l2,l3])
    s1 = gmsh.model.occ.addPlaneSurface([c1])
    v1 = gmsh.model.occ.extrude([(2,s1)], 0.0,0.0,d)
    newVolTag = [x for x in v1 if x[0]==3]
    if recenter:
         gmsh.model.occ.translate(newVolTag, 0, 0, -d/2.0)
    #
    return newVolTag

### SU2 Run Simulation

# SU2 Execution
def run_conf(conf, i, meshing, run, as_row=True, popup=False, extract=['lift', 'drag'], project_name='lcp'):
    case_name=f"{project_name}-{i:0>3}"
    #i = conf['case_index']
    #meshsize = conf['mesh_size']
    meshpath=f"{case_name}.su2"
    gifpath=f"{project_name}-{i:0>3}.jpg"
    
    create_sim(case_name, meshing=meshing, 
        gifpath=gifpath, 
        meshpath=meshpath if meshing else None,
        popup=popup, #meshsize_large=meshsize,
        **conf)
    if run:
        subprocess.run(["/Users/scot/Projects/SU2/bin/SU2_CFD", f"{conf['project_dir']}/{case_name}.cfg"],stdout=subprocess.PIPE) 
    run_dict =  {'meshpath': meshpath, 'gifpath':gifpath, 
                 'surface_flow_path': f"{case_name}-surface_flow.csv"}
    if extract:
        sdf=pandas.read_csv( os.path.join(conf['project_dir'],run_dict['surface_flow_path']))
        extracts = {
            'lift':float(sdf[['Momentum_y']].sum()),
            'drag':float(sdf[['Momentum_x']].sum())
        }
        run_dict.update(extracts)
    
    if as_row:
        return pandas.Series([run_dict[c] for c in ['meshpath', 'gifpath', 'surface_flow_path','lift','drag']])
    else:
        return run_dict
    
    
def run_confs(conf_list, meshing, run, meshsize=5.0, popup=False, animation=False):

    for i in range(len(conf_list)):
        run_conf(conf_list[i], i, meshing)

    if animation:
        ffmpeg_path='/Users/scot/bin/ffmpeg'
        subprocess.run([ffmpeg_path, "-r", "6", "-i", f"/Users/scot/Projects/{project_name}-%03d.jpg", f"{project_name}.gif"]) 

# platform methods
# Functions to setup and run case (SU2 is implicit)
# build cases with no mesh or sim
geometry_only = lambda x: run_conf(x,x['case_index'], False, False, extract=True)
# mesh and simulate
full_sim = lambda x: run_conf(x,x['case_index'], True, True)