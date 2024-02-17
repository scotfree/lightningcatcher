# Installation
import math

import subprocess, sys,os, shutil, math
import pandas
import numpy as np
from itertools import chain

# * paraview
# * ffmpeg

def animate(config, file_paths, framerate="2", remove_anim_frames=False):
    # FIXME: remove local paths!
    ffmpeg_path='/Users/scot/bin/ffmpeg'
    base_dir=config['base_dir']
    output_dir = config['project_name']
    project_name = config['project_name']
    for i,file_path in enumerate(file_paths):
        shutil.copyfile(os.path.join(base_dir,file_path), os.path.join(base_dir, output_dir, f"animation_frame-{i:0>3}.png"))
    subprocess.run([ffmpeg_path, "-r", framerate,  "-y", "-i", 
                    f"{config['base_dir']}/{project_name}/animation_frame-%03d.png", 
                    f"{config['base_dir']}/{project_name}/{project_name}.gif"])
    if remove_anim_frames:
        for i,file_path in enumerate(file_paths):
           os.remove(os.path.join(output_dir, f"animation_frame_{i:0>3}.png"))
    return f"/Users/scot/Projects/{project_name}/{project_name}.gif"


# Notes on the paraview interface hack

# Running your scripts interactively within paraview vs. from the notebook/python/shell



def visualize_case(project_dir, config, base_vis_path="para.py", text="\"Flow Vis\""):
    base_vis = open(base_vis_path).read()
    pvpath = "/Applications/ParaView-5.11.0.app/Contents/bin/pvpython" # para.py fold-004
    with open(f"{config['project_name']}/{config['case_name']}-paraview.log", "w") as outfile:
        subprocess.run([pvpath, base_vis_path, 
                        f"{project_dir}/{config['case_name']}", text,
                       config['base_dir']],stdout=outfile)
    return f"{project_dir}/{config['case_name']}-para.png"
    #return f"{config['base_dir']}/{project_dir}/{config['case_name']}-para.png"
    
    
# * HTML and Notebook convenience methods
def html_imgtable (captions, images):
    return "<table><tr><td>" + "</td><td>".join( captions) + \
        "</td></tr><TR><td>" + \
        "</td><td>".join( [f"<img src=\'{img}\'>" for img in images] ) + \
        "</td></tr></table>"