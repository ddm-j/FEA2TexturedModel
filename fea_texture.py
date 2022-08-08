import matplotlib.colors
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import pymeshlab as pml
from utilities import laplacian_smooth

def main(geometryfile, pointmap, cmap, smooth):

    # Import geometry and Data
    print("Pre-processing")
    fname = geometryfile.split('/')[-1].split('.')[0]
    mesh = o3d.io.read_triangle_mesh(geometryfile)
    mesh.compute_vertex_normals()
    stress_map = np.loadtxt(pointmap, delimiter=',')

    # Normalize Stress Data
    print('Normalizing Point Map Data')
    stress = stress_map[:, 3]
    normalized = (stress - np.min(stress))/(np.max(stress) - np.min(stress))

    # Map Each Stress Point to Mesh
    print('Mapping point map data to 3D model')
    tree = cKDTree(np.asarray(mesh.vertices))
    verts = np.asarray(mesh.vertices).copy()
    tris = np.asarray(mesh.vertices).copy()
    colors = np.empty(len(verts))
    colors[:] = np.nan
    for index, point in enumerate(stress_map):
        _, i = tree.query(point[0:3])
        colors[i] = normalized[index]

    # Interpolate missing colors
    print('Interpolating Missing Vertex Colors')
    missing_color = np.where(np.isnan(colors))[0]
    has_color = np.where(~np.isnan(colors))[0]
    new_colors = colors.copy()
    tree_mesh = cKDTree(verts[has_color])
    for id in missing_color:
        d, i = tree_mesh.query(verts[id], k=4)
        int_color = np.sum(np.multiply(colors[has_color][i[1:]], d[1:]))/np.sum(d[1:])
        new_colors[id] = int_color
    colors = new_colors

    # Apply Laplacian Smoothing of Color data
    if smooth:
        print('Smoothing Color Mapped Data')
        colors = laplacian_smooth(verts, new_colors, iter=5)

    # Map colors to the data
    map = plt.get_cmap(cmap)
    colors = map(colors)

    # Create Texture
    if True:
        print("UV Unwrapping Mesh")
        texture_dimension = 4096
        m = pml.Mesh(verts, face_matrix = np.asarray(mesh.triangles), v_normals_matrix=np.asarray(mesh.vertex_normals), v_color_matrix=colors) # color is N x 4 with alpha info
        ms = pml.MeshSet()
        ms.add_mesh(m, "fea_model")
        #ms.apply_filter("surface_reconstruction_screened_poisson", depth=13, scale=1.1)
        # not familiar with the crop API, but I'm sure it's doable
        # now we generate UV map; there are a couple options here but this is a naive way
        #ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
        ms.apply_filter("parametrization_trivial_per_triangle", textdim=4096)
        # create texture using UV map and vertex colors
        #ms.compute_texmap_from_color(textname=f"texture") # textname will be filename of a png, should not be a full path
        print("Converting vertex colors to face colors.")
        ms.apply_filter("transfer_vertex_color_to_texture", textname="{0}_textured".format(fname), textw=texture_dimension, texth=texture_dimension)
        # texture file won't be saved until you save the mesh
        ms.save_current_mesh("results/texturedModel/{0}_textured.obj".format(fname))


if __name__ == "__main__":

    import argparse

    # Mandatory Inputs
    parser = argparse.ArgumentParser(description='Create a textured OBJ 3D model from FEA data and an existing 3D model')
    parser.add_argument('geometryfile', help='3D file to map the FEA data onto. Accepts STL, OBJ, and other common mesh files.')
    parser.add_argument('pointmap', help='Field/Pointmap data to display on the 3D model. Accepts a CSV in format: x, y, z, P. P - point data (stress, strain, etc)', default=None)

    # Optional Inputs
    parser.add_argument('--cmap', help='Color map for display. Must be one of Matplotlibs supported cmap names. Default is turbo', default='turbo')
    parser.add_argument('--smooth', help='Smooth the mapped color data using laplacian smoothing.',
                        action='store_true')
    parser.add_argument('--no-smooth', help='Disable smoothing of mapped color data.', dest='smooth',
                        action='store_false')
    parser.set_defaults(smooth=False)

    args = parser.parse_args()
    main(**vars(args))