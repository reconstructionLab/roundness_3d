"""
roundness_3d.py
Author: Henry Yuan

Description: A set of functions that compute different 3D roundness metrics from a 3D mesh or binary volume
Some metrics have two functions: one for binary volume (default) and one for a mesh (labeled as _from_mesh)
"""

import numpy as np
import igl
import trimesh
import meshes
from scipy.ndimage import distance_transform_edt

def std_curvature(bin_vol: np.ndarray, Krad=2, curv_type='max'):
    """
    Roundness metric: area-weighted standard deviation of surface curvature
    for a binary volume. Builds a mesh from the volume, then calls std_curvature_from_mesh.
    Normalized only for curv_type='max'
    Lower values = rounder (a perfect sphere is ~1.0 after normalization).

    Inputs:
        bin_vol (np.ndarray): 3D binary array (nonzero/True = particle).
        Krad (int/float): K-ring radius used for curvature estimation.
        curv_type (str): curvature type passed to meshes.compute_curvatures
            Options: 'max', 'min', 'gaussian', 'mean'. Normalization is only valid for 'max'.
    Output:
        float: normalized weighted std of curvature (un-normalized if curv_type != 'max').
    """

    # construct mesh
    v, f = meshes.from_volume(bin_vol)

    return std_curvature_from_mesh(v, f, Krad=Krad, curv_type=curv_type)

def std_curvature_from_mesh(verts, faces, Krad=2, curv_type='max'):
    """
    Same metric as std_curvature, but starting from a triangle mesh.
    Computes per-vertex curvature and per-vertex area, takes the area-weighted standard
    deviation of curvature, and divides by an empirical constant derived from spheres.

    Inputs:
        verts (np.ndarray): (N, 3) float array of vertex coordinates.
        faces (np.ndarray): (M, 3) int array of triangle vertex indices.
        Krad (int/float): K-ring radius for curvature estimation.
        curv_type (str): curvature type
            Options: 'max', 'min', 'gaussian', 'mean'. Normalization is only valid for 'max'.
    Output:
        float: normalized weighted std of curvature (un-normalized, with a printed
        warning, if curv_type != 'max').
    """
    
    CONST_NORMALIZATION = 0.03291137616918748 # normalization factor, defined as
    # the mean weighted standard deviation of spheres with between 1,000 and 10,000 vertices
    # computed from 500 randomly sized spheres -- see generate_random_synthetics.ipynb

    # compute curvatures and areas at each vertex
    K = meshes.compute_curvatures(verts, faces, Krad=Krad, curv_type=curv_type)
    A = meshes.compute_vertex_areas(verts, faces)

    wstd_curv = weighted_std(K, A)

    # normalization constant is computed specifically for curv_type='max', Krad=2, and spheres
    # 1,000 - 10,000 vertices;
    # if different parameters, return unnormalized value and print warning
    if curv_type != 'max' or Krad != 2:
        print("WARNING: normalization constant is only for curv_type='max' and Krad=2. Returning un-normalized weighted standard deviation of curvature")
        return wstd_curv
    
    return wstd_curv / CONST_NORMALIZATION

def wadell_curvature_estimate(bin_vol: np.ndarray, Krad=3, curv_type='max'):
    """
    Curvature-based estimate of Wadell roundness for a binary volume. Builds a mesh,
    computes the maximum inscribed sphere radius, and calls
    wadell_curvature_estimate_from_mesh. See wadell_curvature_estimate_from_mesh for references.

    Inputs:
        bin_vol (np.ndarray): 3D binary array (nonzero/True = particle).
        Krad (int/float): K-ring radius for curvature estimation.
        curv_type (str): curvature type passed to meshes.compute_curvatures.
            Options: 'max', 'min', 'gaussian', 'mean'.
    
    Output:
        float: roundness estimate (~1 for a sphere, smaller for more angular particles).
    """

    
    # construct mesh
    v, f = meshes.from_volume(bin_vol)

    # compute maximum inscribed sphere radius
    mis_rad = max_inscribed_sphere_radius(bin_vol)

    return wadell_curvature_estimate_from_mesh(v, f, mis_rad, Krad=Krad, curv_type=curv_type)

def wadell_curvature_estimate_from_mesh(verts, faces, max_ins_rad, Krad=3, curv_type='max'):
    """
    Curvature-based estimate of Wadell roundness from a mesh.
    Adapted implementation of curvature-based Wadell estimate methods presented in
    Bullard and Garboczi (2013) and Zhao and Wang (2016)

    Inputs:
        verts (np.ndarray): (N, 3) float vertex coordinates.
        faces (np.ndarray): (M, 3) int triangle indices.
        max_ins_rad (float): maximum inscribed sphere radius, in the same units as verts.
        Krad (int/float): K-ring radius for curvature estimation.
        curv_type (str): curvature type passed to meshes.compute_curvatures.
            Options: 'max', 'min', 'gaussian', 'mean'.
    Output:
        float: roundness estimate in (0, 1].

    References:
        Bullard, J.W. and Garboczi, E.J. (2013) “Defining shape measures for 3D star-shaped
        particles: Sphericity, roundness, and dimensions,” Powder Technology, 249, pp. 241–252.
        Available at: https://doi.org/10.1016/j.powtec.2013.08.015.

        Zhao, B. and Wang, J. (2016) “3D quantitative shape analysis on form, roundness, and
        compactness with μCT,” Powder Technology, 291, pp. 262–275. Available at:
        https://doi.org/10.1016/j.powtec.2015.12.029.
    """
    
    # compute curvatures and areas at each vertex
    K = meshes.compute_curvatures(verts, faces, Krad=Krad, curv_type=curv_type)
    A = meshes.compute_vertex_areas(verts, faces)

    # convert maximum inscribed radius to curvature value
    MIS_curv = 1.0 / max_ins_rad

    # find all vertices with curvature > MIS_curv
    bool_inds = (K > MIS_curv)

    # if no vertices with K > MIS_curv, must be very round; return 1
    if np.all(bool_inds == False):
        return 1.0

    wadell_estimate = np.sum(A[bool_inds] * MIS_curv / K[bool_inds]) / np.sum(A[bool_inds])
    return wadell_estimate

def bullard_garboczi_2013(bin_vol: np.ndarray):
    """
    Roundness measure adapted from Bullard & Garboczi (2013): area-weighted mean of
    |cos(angle)| between the unit radial vector (from the particle center) and the
    surface normal at each mesh vertex. Equals 1 for a sphere; designed specifically for
    star-shaped particles.

    Inputs:
        bin_vol (np.ndarray): 3D binary array (nonzero/True = particle).
    Output:
        float: roundness value in [0, 1].

    References:
        Bullard, J.W. and Garboczi, E.J. (2013) “Defining shape measures for 3D star-shaped
        particles: Sphericity, roundness, and dimensions,” Powder Technology, 249, pp. 241–252.
        Available at: https://doi.org/10.1016/j.powtec.2013.08.015.
    """

    # calculate normals and areas of each vertex
    v, f = meshes.from_volume(bin_vol)

    return bullard_garboczi_2013_from_mesh(v, f)

def bullard_garboczi_2013_from_mesh(verts, faces):
    """
    Roundness measure adapted from Bullard & Garboczi (2013), computed from mesh vertices and faces:
    area-weighted mean of |cos(angle)| between the unit radial vector (from the particle center) and
    the surface normal at each mesh vertex. Equals 1 for a sphere; designed specifically for
    star-shaped particles.

    Inputs:
        verts (np.ndarray): (N, 3) float vertex coordinates.
        faces (np.ndarray): (M, 3) int triangle indices.
    Output:
        float: roundness value in [0, 1].

    References:
        Bullard, J.W. and Garboczi, E.J. (2013) “Defining shape measures for 3D star-shaped
        particles: Sphericity, roundness, and dimensions,” Powder Technology, 249, pp. 241–252.
        Available at: https://doi.org/10.1016/j.powtec.2013.08.015.
    """

    n = igl.per_vertex_normals(verts, faces)

    # compute vertex areas
    A = meshes.compute_vertex_areas(verts,faces)

    # calculate radius unit vectors
    center = trimesh.Trimesh(verts, faces).center_mass
    r = verts - center
    norms = np.linalg.norm(r, axis=1)
    r_unit = r / norms[:, np.newaxis]

    dots = np.abs(np.sum(r_unit * n, axis=1))
    return np.sum(dots * A) / np.sum(A)

def sphericity_crofton(bin_vol: np.ndarray):
    """
    Sphericity of a binary volume, using voxel count for volume and a Crofton estimation
    for surface area.

    Inputs:
        bin_vol (np.ndarray): 3D binary array (nonzero/True = particle).
    Output:
        float: sphericity (1 for a perfect sphere, <1 otherwise).
    """
    assert bin_vol.max() == 1

    V = bin_vol.sum()
    SA = surface_area_crofton(bin_vol)
    return (4*np.pi)**(1./3) * (3*V)**(2./3) / SA

def sphericity_naive(bin_vol: np.ndarray):
    """
    Sphericity of a binary volume using the mesh volume and mesh surface area

    Inputs:
        bin_vol (np.ndarray): 3D binary array (nonzero/True = particle).
    Output:
        float: sphericity (1 for a perfect sphere, <1 otherwise).
    """
    v, f = meshes.from_volume(bin_vol)
    return sphericity_naive_from_mesh(v, f)

def sphericity_naive_from_mesh(verts, faces):
    """
    Sphericity from a triangle mesh using trimesh's volume and area.
    Mesh should be closed (watertight) with outward-facing normals.

    Inputs:
        verts (np.ndarray): (N, 3) float vertex coordinates.
        faces (np.ndarray): (M, 3) int triangle indices.
    Output:
        float: sphericity (1 for a perfect sphere, <1 otherwise).
    """
    mesh = trimesh.Trimesh(vertices=verts,faces=faces)
    V = mesh.volume
    SA = mesh.area
    return (4*np.pi)**(1./3) * (3*V)**(2./3) / SA

# ****************
# HELPER FUNCTIONS
# ****************

def weighted_std(xi, wi):
    """
    Weighted (population) standard deviation of values xi with weights wi.
    Weights are normalized internally.

    Inputs:
        xi (np.ndarray): 1D array of values.
        wi (np.ndarray): 1D array of non-negative weights, same length as xi.
    Output:
        float: weighted standard deviation.
    """
    # normalize weights
    norm_wi = wi / np.sum(wi)

    # calculate weighted mean
    wm = np.sum(xi*norm_wi)

    # calculate weighted standard deviation
    wsd = np.sqrt(np.sum(norm_wi*(xi - wm)**2))

    return wsd

def max_inscribed_sphere_radius(bin_vol: np.ndarray):
    """
    Radius of the largest sphere that fits inside bin_vol, computed as the maximum
    of the Euclidean distance transform.

    Inputs:
        bin_vol (np.ndarray): 3D binary array (nonzero/True = particle).
    Output:
        float: maximum inscribed sphere radius in voxel units.
    """
    dists = distance_transform_edt(bin_vol)
    return dists.max()

# crofton surface area estimator using 13 directions
def surface_area_crofton(bin_vol):
    """
    Estimates the surface area of a binary volume with the Crofton formula, counting
    foreground/background boundary crossings along 13 lattice directions (3 axis-aligned,
    6 face-diagonal, 4 space-diagonal), each weighted by its solid-angle fraction. Based on
    method described in Ohser and Mucklich (2000).

    Inputs:
        bin_vol (np.ndarray): 3D binary array (nonzero/True = particle), isotropic unit voxels.
    Output:
        float: estimated surface area in voxel units squared.

    References:
        Ohser, Joachim and Frank Mücklich (2000). Statistical Analysis of Microstructures in
        Materials Science. Chichester, England: John Wiley & Sons, Inc.
    """
    volume = (bin_vol > 0).astype(np.uint8)
    weights = [0.045778, 0.036981, 0.035196] # solid angle coverage fractions for axis aligned, face diagonal, and space diagonal directions,
    # from Ohser and Mucklich 2000, p. 116, for 26 directions
    line_densities = [1,2/np.sqrt(2), 8/3/np.sqrt(3)] # line densities for axis aligned, face diagonal, and space diagonal directions (lines per area)

    directions_13 = [
        # Axis-aligned (3)
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],

        # Face diagonals (6)
        [1, 1, 0],
        [1, -1, 0],
        [1, 0, 1],
        [1, 0, -1],
        [0, 1, 1],
        [0, 1, -1],

        # Space diagonals (4)
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
    ]

    total_contribution = 0.0

    for dz, dy, dx in directions_13:
        # Shift volume by -direction (to compare neighbors)
        shifted = np.roll(volume, shift=(-dz, -dy, -dx), axis=(0, 1, 2))

        # Detect crossings (where voxel value changes between neighbors)
        crossing = (volume != shifted).astype(np.uint8) # this detects crossings on both sides,
        # so doing every direction automatically counts the opposite direction, for 26 total directions

        count = np.sum(crossing)

        # Compute length of the direction vector in physical units
        vec = np.array([dz, dy, dx])
        dir_category = np.sum(np.abs(vec))

        # Weight: number of lines per unit area in that direction
        # Approximated as 2 / length (like in the 3-axis example)
        total_contribution += count * weights[dir_category - 1] / line_densities[dir_category - 1]

    # Normalization constant for 3D Crofton formula (approx)
    surface_area = 4 * total_contribution

    return surface_area

def not_norm_std_curvature_from_mesh(verts, faces, Krad=2, curv_type='max'):
    """
    Same metric as std_curvature_from_mesh except it doesn't normalize the result.
    Only used for calculating raw values for spheres to compute the normalization constant
    for std_curvature_from_mesh()

    Inputs:
        verts (np.ndarray): (N, 3) float array of vertex coordinates.
        faces (np.ndarray): (M, 3) int array of triangle vertex indices.
        Krad (int/float): K-ring radius for curvature estimation.
        curv_type (str): curvature type
            Options: 'max', 'min', 'gaussian', 'mean'. Normalization is only valid for 'max'.
    Output:
        float: normalized weighted std of curvature (un-normalized, with a printed
        warning, if curv_type != 'max').
    """

    # compute curvatures and areas at each vertex
    K = meshes.compute_curvatures(verts, faces, Krad=Krad, curv_type=curv_type)
    A = meshes.compute_vertex_areas(verts, faces)

    wstd_curv = weighted_std(K, A)
    
    return wstd_curv
