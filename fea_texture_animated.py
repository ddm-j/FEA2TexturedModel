import time

import matplotlib.colors
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import pymeshlab as pml
from pathlib import Path
import shutil
import pandas as pd
import moviepy.video.io.ImageSequenceClip
import os
from tqdm import tqdm
from contextlib import contextmanager
import sys, os, io, warnings

@contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:


    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()

def main(geometryfile, datapath, steps, cmap, length, fps):

    # Import Data
    print("Pre-processing data.")
    meshname = geometryfile.split('/')[-1].split('.')[0]
    mesh = o3d.io.read_triangle_mesh(geometryfile)
    mesh.compute_vertex_normals()
    steps = [np.loadtxt(datapath+'/pointmap'+str(i)+".csv", delimiter=',') for i in range(int(steps))]
    verts = np.asarray(mesh.vertices).copy()
    tris = np.asarray(mesh.vertices).copy()

    # Normalize Data based on last load step
    print("Normalizing point map data.")
    MIN = min(steps[-1][:, 3])
    MAX = max(steps[-1][:, 3])
    color_steps = [(i[:, 3]-MIN)/(MAX - MIN) for i in steps]

    # Map Each Stress Point to Mesh
    print("Mapping point map data to 3D model.")
    fname = "results/animatedModel/intermediary/mapped_color_steps.npy"
    if Path.is_file(Path(fname)):
        print("Loading saved mapped color data.")
        mapped_color_steps = np.load(fname, allow_pickle=True)
    else:
        print("Saved mapped color data not found, re-computing.")
        mapped_color_steps = []
        for stress_map, color_step in zip(steps, color_steps):

            tree = cKDTree(np.asarray(mesh.vertices))
            colors = np.empty(len(verts))
            colors[:] = np.nan
            for index, point in enumerate(stress_map):
                _, i = tree.query(point[0:3])
                colors[i] = color_step[index]

            # Interpolate missing colors
            missing_color = np.where(np.isnan(colors))[0]
            has_color = np.where(~np.isnan(colors))[0]
            new_colors = colors.copy()
            tree_mesh = cKDTree(verts[has_color])
            for id in missing_color:
                d, i = tree_mesh.query(verts[id], k=4)
                int_color = np.sum(np.multiply(colors[has_color][i[1:]], d[1:]))/np.sum(d[1:])
                new_colors[id] = int_color

            mapped_color_steps.append(new_colors)
        np.save(fname, mapped_color_steps, allow_pickle=True)

    # Create animation color data
    clip_length = length
    fps = fps
    frame_count = clip_length*fps

    fname = "results/animatedModel/intermediary/color_data.npy"
    if Path.is_file(Path(fname)):
        print("Loading animation color data.")
        color_data = np.load(fname, allow_pickle=True)
    else:
        print("Saved animation color data not found, re-computing.")
        color_data = np.empty((len(mapped_color_steps[0]), frame_count))
        color_data[:] = np.nan

        col_ids = (np.linspace(0, frame_count - 1, len(color_steps))).astype(np.int)

        for i, ind in enumerate(col_ids):
            color_data[:, ind] = mapped_color_steps[i]

        # Interpolate Missing Frame Colors
        df = pd.DataFrame(color_data)
        df.interpolate(axis=1, inplace=True)

        # Convert back to nd-array
        color_data = df.to_numpy()

        np.save(fname, color_data, allow_pickle=True)

    # Create animation texture for each step
    image_folder = "results/animatedModel/frames"
    if Path.is_file(Path('{0}/pointmap_texture_{1}.png'.format(image_folder, frame_count-1))):
        print("Animation textures from previous run detected.")
    else:
        print("Animation textures not detected. Recomputing.")
        map = plt.get_cmap(cmap)

    # Wait for a second
    time.sleep(1)
    for i in tqdm(range(frame_count)):
        # Suppress output from loud PyMeshLab functions
        with stdchannel_redirected(sys.stdout, os.devnull):
            with stdchannel_redirected(sys.stderr, os.devnull):
                #print("Generating texture for frame {0}".format(i))

                # Map color number to RGBA
                color = map(color_data[:, i])

                texture_dimension = 4096
                m = pml.Mesh(verts, face_matrix = np.asarray(mesh.triangles), v_normals_matrix=np.asarray(mesh.vertex_normals), v_color_matrix=color) # color is N x 4 with alpha info
                ms = pml.MeshSet()
                ms.add_mesh(m, "fea_{0}".format(i))
                ms.apply_filter("parametrization_trivial_per_triangle", textdim=4096)
                # create texture using UV map and vertex colors
                ms.apply_filter("transfer_vertex_color_to_texture", textname="pointmap_texture_{1}".format(image_folder, i), textw=texture_dimension, texth=texture_dimension)
                # texture file won't be saved until you save the mesh
                ms.save_current_mesh("results/animatedModel/{0}_animated.obj".format(meshname))

                # Move texture file to different folder
                shutil.move('results/animatedModel/pointmap_texture_{0}.png'.format(i), '{0}/pointmap_texture_{1}.png'.format(image_folder, i))

    # Generate the texture animation
    print("Generating MP4 from FEA Textures.")
    image_files = [os.path.join(image_folder,img)
                   for img in os.listdir(image_folder)
                   if img.endswith(".png")]
    image_ids = np.array([int(i.split('_')[-1][:-4]) for i in image_files])
    sorted_ids = np.argsort(image_ids)
    image_files = list(np.array(image_files)[sorted_ids])
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile('results/animatedModel/animatedTexture.mp4')


if __name__ == "__main__":

    import argparse

    # Mandatory Inputs
    parser = argparse.ArgumentParser(description='Create a textured OBJ 3D model from FEA data and an existing 3D model')
    parser.add_argument('geometryfile', help='3D file to map the FEA data onto. Accepts STL, OBJ, and other common mesh files.')
    parser.add_argument('datapath', help='Path to pointmaps for animation. Must be of name pointmap#.csv where # corresponds to the step number', default=None)
    parser.add_argument('steps',
                        help='Number of steps for animation. Must correspond with number of steps in the datappath',
                        default=None)

    # Optional Inputs
    parser.add_argument('--cmap', help='Color map for display. Must be one of Matplotlibs supported cmap names. Default is turbo', default='turbo')
    parser.add_argument('--length', help='Length of the animation in seconds. Default is 6 seconds.', default=6)
    parser.add_argument('--fps', help='Frames per second of the animation. Default is 60 FPS.', default=60)

    args = parser.parse_args()
    main(**vars(args))



