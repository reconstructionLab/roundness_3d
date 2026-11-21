"""
synthetic_grains.py
Authors: Henry Yuan and Akshay Mehra

Description: A set of functions that construct synthetic sedimentary grains
loosely based on detrital zircon morphologies
"""
import numpy as np
import trimesh
import pymeshlab as ml
import meshes
import igl
import scipy.sparse.linalg as spla
from scipy.spatial.transform import Rotation

# Note: All meshes for synthetic grains are scaled so that their length = 1.

# global variables
TARGET_EDGE_FRACTION = 0.03  # Fraction of the fitted target bounding-box diagonal.
REMESH_ITERATIONS = 25
FEATURE_ANGLE = 30.0  # Degrees; edges above this dihedral angle are features.

def rotate_mesh(mesh,rotation_matrix):
    """
    Rotates a mesh about the origin (not about its own center) and returns a new mesh.
    Faces are unchanged.

    Inputs:
        mesh (trimesh.Trimesh): mesh to rotate.
        rotation_matrix (numpy.ndarray, shape (3,3)): rotation matrix.
    Output:
        trimesh.Trimesh: new rotated mesh.
    """

    # don't need to center first because voxelizing later will shift to it
    verts = mesh.vertices

    rotated_verts = rotation_matrix @ verts.T

    rotated_mesh = trimesh.Trimesh(vertices=rotated_verts.T,faces=mesh.faces, process=False)

    return rotated_mesh

def starting_zircon_mesh(aspect_ratio, roundness, pass_rng=None, rotate=True):
    """
    Creates the starting (pre-voxelization) mesh for a synthetic zircon grain: an
    ellipsoid (roundness == 1), a euhedral bipyramidal prism (roundness == 0), or a
    semi-round grain made by smoothing the euhedral shape (0 < roundness < 1). Optionally
    applies a random rotation. The grain's long axis (z) has length 1 before rotation.

    Inputs:
        aspect_ratio (float): length / width of the grain, must be >= 1.
        roundness (float): 0 (euhedral) to 1 (round).
        pass_rng (np.random.Generator or None): random generator for reproducible rotations.
            If None, a new unseeded generator is used (not reproducible).
        rotate (bool): whether to apply a random rotation.
    Outputs:
        rotated_mesh (trimesh.Trimesh): the (possibly rotated) grain mesh.
    """
    
    assert aspect_ratio >= 1

    if roundness == 1:
        mesh = initial_round_zircon_mesh(aspect_ratio)
    elif roundness == 0:
        # make minimal euhedral zircon volume
        mesh = initial_euhedral_zircon_mesh(aspect_ratio)
    elif roundness == 0.5:
        mesh = initial_semiround_zircon_mesh(aspect_ratio)
    else:
        raise ValueError(f"roundness must be 0, 0.5, or 1, got {roundness}.")

    # rotate mesh before voxelizing (rotate vertices exactly)
    if rotate:
        rot_matrix = Rotation.random(rng=pass_rng).as_matrix()
        rotated_mesh = rotate_mesh(mesh,rot_matrix)
    else:
        rotated_mesh = mesh

    return rotated_mesh

def initial_euhedral_zircon_mesh(aspect):
    """
    Builds the minimal-vertex euhedral zircon: a square-section prism with pyramidal
    terminations at both ends (16 faces / 10 vertices; for aspect == 1, a bipyramid
    with 8 faces / 6 vertices). Total length (prism + both pyramids) is 1 and width is 1/aspect.

    Inputs:
        aspect (float): length / width, >= 1.
    Output:
        trimesh.Trimesh: closed mesh with outward-facing normals.
    """

    # set l + 2h = 1
    w = 1.0 / aspect
    d = w
    h = w / 2
    l = 1 - w

    if aspect == 1:
        # define 6 vertices
        A = [w/2,d/2,0]
        B = [w,0,h]
        C = [0,0,h]
        D = [0,d,h]
        E = [w,d,h]
        J = [w/2,d/2,2*h]
        verts = np.array([A,B,C,D,E,J])
        # mesh faces
        faces = np.vstack(
            [
                [0,1,2],
                [0,4,1],
                [0,3,4],
                [0,2,3],
                [4,5,1],
                [4,3,5],
                [2,5,3],
                [2,1,5]
            ]
        )

    else:
        # define 10 vertices
        A = [w/2,d/2,0]
        B = [w,0,h]
        C = [0,0,h]
        D = [0,d,h]
        E = [w,d,h]
        F = [w,d,l+h]
        G = [w,0,l+h]
        H = [0,0,l+h]
        I = [0,d,l+h]
        J = [w/2,d/2,2*h + l]

        verts = np.array([A,B,C,D,E,F,G,H,I,J])

        # mesh faces
        faces = np.vstack(
            [
                [0,1,2],
                [0,4,1],
                [0,3,4],
                [0,2,3],
                [1,6,2],
                [2,6,7],
                [1,4,5],
                [1,5,6],
                [3,8,4],
                [4,8,5],
                [3,7,8],
                [2,7,3],
                [5,9,6],
                [5,8,9],
                [7,9,8],
                [7,6,9]
            ]
        )

    mesh = trimesh.Trimesh(vertices=verts,faces=faces,process=True)
    mesh.fix_normals()

    return mesh

def initial_round_zircon_mesh(aspect):
    """
    Builds a round zircon as an ellipsoid (a subdivided icosphere scaled along its axes,
    ~10,000 vertices) with long axis (z) of length 1 and width 1/aspect.

    Inputs:
        aspect (float): length / width, >= 1.
    Output:
        trimesh.Trimesh: ellipsoid mesh centered at the origin.
    """
    c = 0.5
    a = c / aspect
    b = a
    mesh = trimesh.creation.icosphere(subdivisions=5) # 5 subdivisions gives ~10,000 verts
    mesh.apply_scale([a, b, c])
    mesh.fix_normals()
    return mesh

def euhedral_zircon_mesh(aspect, target_vertex_count):
    """
    Builds a euhedral zircon mesh and isotropically remeshes it (preserving sharp edges)
    to approximately the requested number of vertices.

    Inputs:
        aspect (float): length / width, >= 1.
        target_vertex_count (int): desired number of vertices (>= 4).
    Output:
        trimesh.Trimesh: remeshed, watertight euhedral mesh (raises ValueError if not closed).
    """

    assert aspect >= 1

    mesh = initial_euhedral_zircon_mesh(aspect)
    new_mesh = remesh(mesh,target_vertex_count, TARGET_EDGE_FRACTION * np.linalg.norm(mesh.extents))
    
    if not new_mesh.is_watertight:
        raise ValueError("Mesh is not closed")
    
    return new_mesh

def mean_curvature_flow(mesh, c=5e-3):
    """
    Rounds a mesh by implicit mean curvature flow using 20 backward-Euler steps with the
    cotangent Laplacian. Total flow time scales with the grain's volume^(2/3), so
    that c controls rounding strength independent of grain size. The result is rescaled
    so that its length is 1.

    Inputs:
        mesh (trimesh.Trimesh): closed mesh to smooth
        c (float): dimensionless rounding strength (flow time = c * volume^(2/3)).
            Default value is calibrated for this paper.
    Output:
        trimesh.Trimesh: smoothed mesh, same connectivity, long axis length 1.
    """
    # copy mesh to not modify the passed one
    mesh = mesh.copy()

    # first merge vertices (probably not needed but just in case)
    mesh.merge_vertices()

    v = mesh.vertices.astype(np.float64)
    f = mesh.faces.astype(np.int64)

    # calculate mesh volume
    V = mesh.volume
    length_scale = V**(1./3)

    T = c*length_scale**2 # total flow time should scale as length_scale**2 according to Claude; simple dimensional analysis of MCF equation agrees
    n_steps = 20
    dt = T / n_steps
    for _ in range(n_steps):
        L = igl.cotmatrix(v, f)
        M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
        v = spla.spsolve((M - dt * L).tocsc(), M @ v)

    # rescale so that length remains 1
    # get current length (last axis is long axis)
    length = v[:,2].max() - v[:,2].min()
    scalar = 1.0 / length
    v *= scalar

    return trimesh.Trimesh(vertices=v,faces=f)

def initial_semiround_zircon_mesh(aspect):
    """
    Builds a partially rounded zircon: creates a ~10,000-vertex euhedral mesh, then
    applies mean curvature flow.

    Inputs:
        aspect (float): length / width, >= 1.
    Output:
        trimesh.Trimesh: semi-round grain mesh, long axis length 1.
    """

    assert aspect >= 1

    starting_vertex_count = 10000

    # make euhedral zircon mesh
    euhedral_mesh = euhedral_zircon_mesh(aspect, starting_vertex_count)

    # round with mean curvature flow
    rounded_mesh = mean_curvature_flow(euhedral_mesh)

    return rounded_mesh

def voxelize_mesh_at_length(length, mesh, pad_margin=5):
    """
    Converts a mesh (long axis length 1) into a filled binary volume in which the grain is
    about `length` voxels long, then zero-pads the volume.

    Inputs:
        length (float): desired grain length in voxels (voxel pitch = 1/length).
        mesh (trimesh.Trimesh): grain mesh, ideally watertight.
        pad_margin (int): number of background voxels added on every side.
    Output:
        np.ndarray: 3D uint8 array (1 = grain, 0 = background).
    """
    # voxelize to volume
    vol = mesh.voxelized(pitch=1.0/length).fill().matrix.astype(np.uint8)
    # pad
    vol = np.pad(vol, pad_margin)

    return vol

def synthetic_zircon(target_vertex_count,aspect_ratio, roundness, pass_rng=None, rotate=True, MIN_VERTS=1000, MAX_VERTS=10000, MIN_LENGTH=10, MAX_LENGTH=100):
    """
    Generates a synthetic zircon as a binary volume plus its smoothed mesh. Builds a starting
    mesh, then bisects on the voxelized grain length (in voxels) until the mesh
    produced by meshes.from_volume has a vertex count near the target and within
    [MIN_VERTS, MAX_VERTS).

    Inputs:
        target_vertex_count (int): desired number of mesh vertices.
        aspect_ratio (float): length / width, >= 1.
        roundness (float): 0 (euhedral), 0.5 (semi-round), or 1 (round).
        pass_rng (np.random.Generator or None): generator for reproducible rotation.
        rotate (bool): apply a random rotation to the grain.
        MIN_VERTS, MAX_VERTS (int): allowed vertex count range.
        MIN_LENGTH, MAX_LENGTH (float): search range for grain length in voxels.
    Outputs:
        bv (np.ndarray): padded 3D uint8 binary volume.
        mesh (trimesh.Trimesh): mesh from meshes.from_volume(bv).
        nv (int): number of vertices in mesh.
    """
    assert target_vertex_count >= MIN_VERTS and target_vertex_count < MAX_VERTS
    
    base_mesh = starting_zircon_mesh(aspect_ratio,roundness,pass_rng,rotate)

    # find length that yields closest to target vertex count with bisection algorithm
    hi = MAX_LENGTH
    lo = MIN_LENGTH

    # keep track of best option
    best, best_err = None, float("inf")
    while hi - lo > 1:
        l = (lo + hi) / 2

        bv = voxelize_mesh_at_length(l, base_mesh)
        try:
            verts, faces = meshes.from_volume(bv)
        # except if verts or faces is None
        except ValueError:
            # grain too small at this length, search larger
            lo = l
            continue
        nv = verts.shape[0]

        # calculate error and set best if applicable
        err = abs(nv - target_vertex_count)
        if MIN_VERTS <= nv < MAX_VERTS and err < best_err:
            best, best_err = (bv, verts, faces, nv), err

        # is nv larger than the target?
        if nv > target_vertex_count:
            hi = l
        # converse
        elif nv < target_vertex_count:
            lo = l
        else:
            # reached target vertex count
            break

    if best is None:
        raise RuntimeError(
            f"No grain with {MIN_VERTS} <= nv < {MAX_VERTS} found "
            f"(target {target_vertex_count}, aspect {aspect_ratio}, roundness {roundness}).")

    # take best option
    bv, verts, faces, nv = best

    # make mesh
    # Create a simple mesh from vertices and faces
    mesh = trimesh.Trimesh(vertices=verts,faces=faces, process=False)
    return bv, mesh, nv

def _remesh_at_edge_length(mesh, edge_length):
    """
    Cleans a mesh (duplicate vertices/faces, null faces, unreferenced vertices) and runs
    PyMeshLab's isotropic explicit remeshing at a fixed target edge length, preserving
    edges sharper than FEATURE_ANGLE.

    Inputs:
        mesh (trimesh.Trimesh): input mesh.
        edge_length (float): target edge length, in mesh units.
    Output:
        trimesh.Trimesh: remeshed mesh with normals fixed.
    """
    meshset = ml.MeshSet()
    meshset.add_mesh(ml.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    ))
    # Clean up degenerate geometry (duplicate vertices, zero-area/duplicate faces,
    # unreferenced vertices) before remeshing.
    meshset.apply_filter("meshing_remove_duplicate_vertices")
    meshset.apply_filter("meshing_remove_null_faces")
    meshset.apply_filter("meshing_remove_duplicate_faces")
    meshset.apply_filter("meshing_remove_unreferenced_vertices")
    meshset.apply_filter(
        "meshing_isotropic_explicit_remeshing",
        iterations=REMESH_ITERATIONS,
        targetlen=ml.PureValue(float(edge_length)),
        featuredeg=FEATURE_ANGLE,
        checksurfdist=True,
    )
    current = meshset.current_mesh()
    result = trimesh.Trimesh(
        vertices=current.vertex_matrix().copy(),
        faces=current.face_matrix().copy(),
        process=True,
    )
    result.fix_normals()
    return result

def remesh(mesh, target_vertex_count, initial_edge_length, tolerance=0.05, max_attempts=10):
    """
    Remeshes to a target vertex count by iteratively adjusting the target edge length
    (edge length scales with sqrt(vertex_count / target)) until the vertex count is
    within a relative tolerance of the target. Returns the closest result if none
    meet the tolerance.

    Inputs:
        mesh (trimesh.Trimesh): input mesh.
        target_vertex_count (int): desired number of vertices (>= 4).
        initial_edge_length (float): starting guess for the target edge length.
        tolerance (float): acceptable relative error in vertex count.
        max_attempts (int): maximum number of remeshing iterations.
    Output:
        trimesh.Trimesh: remeshed mesh closest to the target vertex count.
    """
    if not isinstance(target_vertex_count, (int, np.integer)) or target_vertex_count < 4:
        raise ValueError("target_vertex_count must be an integer of at least 4.")
    edge_length = float(initial_edge_length)
    best_mesh = None
    best_error = float("inf")
    for _ in range(max_attempts):
        result = _remesh_at_edge_length(mesh, edge_length)
        vertex_count = len(result.vertices)
        if vertex_count == 0:
            raise ValueError("Remeshing produced an empty mesh.")
        error = abs(vertex_count - target_vertex_count) / target_vertex_count
        if error < best_error:
            best_mesh, best_error = result, error
        if error <= tolerance:
            return result
        
        edge_length *= np.sqrt(vertex_count / target_vertex_count)
    
    return best_mesh
