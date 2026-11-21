"""
meshes.py
Author: Henry Yuan

Description: A set of functions for building and making calculations on 3D triangular meshes
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
import trimesh
from skimage.filters import gaussian
import igl

def from_volume(volume,sigma=1.0):
    """
    Builds a smoothed triangle mesh of the surface of a binary volume. Applies a Gaussian
    blur to the volume, runs marching cubes, then applies 10 iterations of Taubin smoothing.

    Inputs:
        volume (np.ndarray): 3D binary array (nonzero/True = particle).
        sigma (float): standard deviation (in voxels) of the Gaussian blur applied before
            marching cubes. Default is 1.0.
    Outputs:
        verts (np.ndarray): (N, 3) float vertex coordinates.
        faces (np.ndarray): (M, 3) int triangle vertex indices.
    """
    # check that volume is binary
    assert volume.max() == 1

    smoothed_volume = gaussian(volume.astype(float), sigma=sigma, mode='constant', preserve_range=True)
    if smoothed_volume.max() <= 0.5:
        # then grain is too small for making mesh suitable for roundness calculation
        return None, None
    verts, faces, _, _ = marching_cubes(smoothed_volume, level=0.5, allow_degenerate=False) # _ is values
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # make sure normals pointing outward
    mesh.fix_normals()
    trimesh.smoothing.filter_taubin(mesh, iterations=10)
    
    # check mesh validity
    if not mesh.is_watertight:
        print("Warning: mesh touches volume boundary, orientation check unreliable")
    if not mesh.is_winding_consistent:
        print("Warning: mesh face winding is not consistent")

    verts = mesh.vertices
    faces = mesh.faces

    if verts is None or faces is None:
        raise ValueError(f"Volume results in invalid mesh")
    
    return verts, faces

# calculates curvature at all vertices of mesh defined by verts and faces
# Krad is radius of Kring when calculating curvature with libigl
# parameter curv can be ['mean','gaussian','max','min']
# Default Krad = 5, which is default for igl.principal_curvature
def compute_curvatures(verts, faces, Krad=5, curv_type='max'):
    """
    Computes a curvature value at every vertex of a mesh using libigl's principal
    curvature estimator.

    Inputs:
        verts (np.ndarray): (N, 3) float vertex coordinates.
        faces (np.ndarray): (M, 3) int triangle vertex indices.
        Krad (int): size of the k-ring neighborhood used for the fit. Default is 5.
        curv_type (str): 'max' (largest principal curvature), 'min' (smallest),
            'mean' (average of the two), or 'gauss' (product of the two).
    Output:
        np.ndarray: (N,) float per-vertex curvature (1/length units of verts).
    """
    pd1, pd2, pv1, pv2, bad_vertices = igl.principal_curvature(verts, faces, radius=Krad)

    # check for errors in curvature calculation
    n_bad = len(bad_vertices)
    n_nan = np.isnan(pv1).sum() + np.isnan(pv2).sum()
    if n_bad > 0 or n_nan > 0:
        print(f"Warning: Curvature computation failed at {n_bad} vertices ({n_nan} NaN values).")

    if curv_type == 'mean':
        K = (pv1 + pv2) / 2
    elif curv_type == 'max':
        K = pv1
    elif curv_type == 'gaussian':
        K = pv1 * pv2
    elif curv_type == 'min':
        K = pv2
    else:
        raise ValueError(f"Invalid curv_type '{curv_type}'. Must be 'mean', 'max', 'gaussian', or 'min'.")

    return K
    
# calculates areas of each vertex in the mesh defined by verts and faces
def compute_vertex_areas(verts, faces):
    """
    Computes the surface area associated with each vertex using Voronoi mass matrix with libigl.

    Inputs:
        verts (np.ndarray): (N, 3) float vertex coordinates.
        faces (np.ndarray): (M, 3) int triangle vertex indices.
    Output:
        np.ndarray: (N,) float area per vertex (length units of verts, squared).
    """
    M = igl.massmatrix(verts, faces, igl.MASSMATRIX_TYPE_VORONOI)
    A = M.diagonal()
    return A

# plots 3D mesh defined by verts and faces using skimage
def plot_mesh(verts, faces):
    """
    Plots a triangle mesh with 3D matplotlib.

    Inputs:
        verts (np.ndarray): (N, 3) float vertex coordinates.
        faces (np.ndarray): (M, 3) int triangle vertex indices.
    Outputs:
        fig (matplotlib.figure.Figure): the figure.
        ax (mpl_toolkits.mplot3d.Axes3D): the 3D axes.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mins, maxs = verts.min(0), verts.max(0)
    center = (mins + maxs) / 2
    r = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    ax.set_box_aspect((1, 1, 1))

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    plt.tight_layout()
    return fig, ax
