"""
analyze_synthetics.py
Authors: Henry Yuan

Description: A set of functions for building random synthetic grains,
calculating different roundness metrics of them, and analyzing the effectiveness of those metrics
"""

import numpy as np
import os
import synthetic_grains as sg
import roundness_3d as r3d
import pandas as pd
from fast_rs import rs as fast_rs
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits

# GLOBAL VARIABLES
ALL_METRICS = ['std_curv','wadell','bg2013','sphericity_crofton','sphericity_naive','fast_rs']


def generate_random_synthetic_and_compute_roundness(n_per_round=1000, round_stages=(0,0.5,1), metrics=ALL_METRICS, save_dir=None, save_stls=False,
                                        save_vols=False, random_rotation=True, rng_seed=42, MIN_VERTS=1000, MAX_VERTS=10000, MIN_ASPECT=1,
                                        MAX_ASPECT=3, MIN_LENGTH=10, MAX_LENGTH=100):
    """
    Generates random synthetic zircon grains at each roundness stage (random aspect ratio,
    random target mesh vertex count, optional random rotation), computes the requested
    roundness metrics for each grain, and optionally saves each grain's binary volume
    and/or STL mesh. Aspect ratio and target vertex count are drawn uniformly from
    [MIN_ASPECT, MAX_ASPECT) and [MIN_VERTS, MAX_VERTS).

    Inputs:
        n_per_round (int): number of grains generated per roundness stage.
        round_stages (list of float): roundness values to generate; each must be 0, 0.5, or 1.
        metrics (list of str): metrics to compute, from ALL_METRICS ('std_curv', 'wadell',
            'bg2013', 'sphericity_crofton', 'sphericity_naive', 'fast_rs', 'std_curv_not_norm').
            Pass [] to skip. all metrics if only saving volumes or mesh .stls. Only use
            'std_curv_not_norm' for computing normalization factor for
            roundness_3d.std_curvature().
        save_dir (str or None): directory prefix for saved files. Required if save_stls or save_vols is True.
        save_stls (bool): save each mesh as mesh_{index}.stl.
        save_vols (bool): save each binary volume as vol_{index}.npy.
        random_rotation (bool): apply a random rotation to each grain before voxelization.
        rng_seed (int): seed for the random number generator (reproducibility).
        MIN_VERTS, MAX_VERTS (int): allowed range of mesh vertex counts.
        MIN_ASPECT, MAX_ASPECT (float): range of aspect ratios (length / width).
        MIN_LENGTH, MAX_LENGTH (float): grain length range (in voxels) searched when matching
            the target vertex count; calibrated to MIN_VERTS and MAX_VERTS.
    Output:
        pd.DataFrame: one row per grain (stage-major order, row = stage_index * n_per_round
            + grain_index) with columns 'roundness', 'aspect_ratio', 'vertex_count', plus
            one column per metric.
    """
    
    if save_stls or save_vols:
        assert save_dir is not None

    if save_dir is not None:
        # make save_dir
        os.makedirs(save_dir, exist_ok=True)

    # get ranges for vertex count and aspect ratios
    nv_range = MAX_VERTS - MIN_VERTS
    ar_range = MAX_ASPECT - MIN_ASPECT

    n_round_stages = len(round_stages)

    # make pandas dataframe with index and columns for: round_stage, aspect ratio, vertex count, and any roundness metrics
    n_rows = n_round_stages * n_per_round
    cols = ['roundness', 'aspect_ratio', 'vertex_count'] + metrics
    synthetic_stats_df = pd.DataFrame(
        np.full((n_rows, len(cols)), np.nan),
        columns=cols)

    # define rng
    rng = np.random.default_rng(seed=rng_seed)

    # start loop
    for i in range(n_round_stages):
        r = round_stages[i]
        print(f"Starting r = {r} synthetics...")

        for j in range(n_per_round):
            global_ind = i*n_per_round + j
            if j % 100 == 0:
                print(f'Completed {j}/{n_per_round}')

            # select random aspect and target vertex count
            aspect = (rng.random()*ar_range) + MIN_ASPECT
            target_nv = int((rng.random()*nv_range) + MIN_VERTS)

            bv, mesh, nv = sg.synthetic_zircon(target_nv,aspect, r, pass_rng=rng, rotate=random_rotation, MIN_VERTS=MIN_VERTS, MAX_VERTS=MAX_VERTS, MIN_LENGTH=MIN_LENGTH, MAX_LENGTH=MAX_LENGTH)
            
            # extract vertices and faces
            verts = mesh.vertices
            faces = mesh.faces

            # record parameters / basic stats
            synthetic_stats_df.at[global_ind, "roundness"] = r
            synthetic_stats_df.at[global_ind, "aspect_ratio"] = aspect
            synthetic_stats_df.at[global_ind, "vertex_count"] = nv

            # compute and record roundness metrics
            for m in metrics:
                synthetic_stats_df.at[global_ind, m] = _compute_metric(m, bv, verts, faces)

            # save binary volume by index
            if save_vols:
                np.save(os.path.join(save_dir, f'vol_{global_ind}.npy'),bv)
            # Export to STL by index
            if save_stls:
                mesh.export(os.path.join(save_dir, f'mesh_{global_ind}.stl'))

    # return dataframe with stats
    return synthetic_stats_df

def _compute_metric(m, bv, verts, faces):
    """
    Helper function that computes metric m for grain defined by binary volume bv and
    mesh (verts, faces)
    
    Inputs:
    m (str): metric (see above for list of options)
    bv (np.ndarray): binary volume representation of grain
    verts (np.ndarray): shape (N, 3) float array of vertex coordinates for mesh
    faces (np.ndarray): shape (M, 3) integer array of faces for mesh
    
    Output:
    metric_value (float): value of the metric
    """
    if m == 'std_curv':
        return r3d.std_curvature_from_mesh(verts, faces)
    elif m == 'std_curv_not_norm':
        return r3d.not_norm_std_curvature_from_mesh(verts, faces)
    elif m == 'wadell':
        mis = r3d.max_inscribed_sphere_radius(bv)
        return r3d.wadell_curvature_estimate_from_mesh(verts, faces, mis)
    elif m == 'bg2013':
        return r3d.bullard_garboczi_2013_from_mesh(verts, faces)
    elif m == 'sphericity_crofton':
        return r3d.sphericity_crofton(bv)
    elif m == 'sphericity_naive':
        return r3d.sphericity_naive_from_mesh(verts, faces)
    elif m == 'fast_rs':
        roundness, _, _ = fast_rs.rs_calculate(bv, lt_scale=1)
        return roundness[0]
    raise ValueError(f"Roundness metric {m} not recognized")

def compute_gmm_overlap_score(grouped_data, n_init=20, seed=42):
    """
    Measures how well a roundness metric separates known groups (e.g. euhedral,
    semi-round, round). Fits a k-component Gaussian mixture model to the pooled values
    (k = number of groups), matches clusters to groups with the best one-to-one
    assignment, and returns the fraction of grains that land in the wrong cluster.
    0 = perfectly separable; larger = more overlap.

    Inputs:
        grouped_data (sequence of np.ndarray): one 1D float array of metric values per
            group, e.g. (data_euhedral, data_semiround, data_round).
        n_init (int): number of GMM initializations (best fit is kept).
        seed (int): random seed for the GMM.
    Output:
        float: misclassification fraction.
    """
    
    k = len(grouped_data)
    labels = np.repeat(np.arange(k), [len(g) for g in grouped_data])
    X = np.concatenate(grouped_data).reshape(-1, 1)

    # train a Gaussian Mixture Model and use it to make k clusters
    pred = GaussianMixture(n_components=k, n_init=n_init, random_state=seed).fit_predict(X)

    # contingency[i, j] = number of points from group i assigned to cluster j
    contingency = np.zeros((k, k), dtype=int)
    np.add.at(contingency, (labels, pred), 1)

    # best one-to-one assignment of clusters to groups
    rows, cols = linear_sum_assignment(-contingency)
    return 1 - contingency[rows, cols].sum() / len(labels)

def compute_gmm_overlap_score_in_ranges(all_round_data, vertex_counts, ranges):
    """
    Computes the GMM overlap score for each metric within each range of mesh vertex
    counts, to explore how a metric's ability to separate groups varies with mesh resolution.
    For each range, only grains with lo <= vertex count < hi are used.

    Inputs:
        all_round_data (np.ndarray): shape (n_metrics, n_groups, n_grains_per_group),
            metric value for each grain.
        vertex_counts (np.ndarray): shape (n_groups, n_grains_per_group), mesh vertex
            count of each grain (same grain ordering as all_round_data).
        ranges (array-like): shape (n_ranges, 2), (lo, hi) vertex count bounds; lo inclusive,
            hi exclusive.
    Output:
        np.ndarray: shape (n_metrics, n_ranges) overlap score for each metric and range.
    """
    
    # get dimensions of data
    n_metrics, n_groups, n_grains_per_group = all_round_data.shape
    n_ranges = len(ranges)

    # make array for overlap scores to return
    overlap_scores = np.zeros((n_metrics,n_ranges))

    # compute overlap scores for each range, for each metric
    for i,rg in enumerate(ranges):
        # unpack range min and max
        lo, hi = rg

        # find all grains with meshes within this range of vertex counts
        is_within_range = np.logical_and(vertex_counts >= lo, vertex_counts < hi) # boolean array, same shape as vertex_counts

        # loop over metrics
        for j in range(n_metrics):
            # make list for data within the range
            grouped_data = [None] * n_groups

            # populate list
            for k in range(n_groups):
                grouped_data[k] = all_round_data[j,k,is_within_range[k]]

            # calculate overlap score
            overlap_scores[j,i] = compute_gmm_overlap_score(grouped_data)

    return overlap_scores
