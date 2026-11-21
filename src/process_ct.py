"""
process_ct.py

"""

import numpy as np
import cv2
import os
from scipy.ndimage import label, binary_opening, binary_closing, distance_transform_edt, center_of_mass, binary_fill_holes, binary_dilation, binary_erosion, generate_binary_structure
from skimage.morphology import ball, h_maxima
from skimage.segmentation import watershed
from sklearn.cluster import DBSCAN
import time

def process_ct_data(dir):
    print("Processing ct scan data")

    # List all items in the directory
    all_files = os.listdir(dir)
    all_files = sorted(all_files)

    # Filter only files
    imgs = [os.path.join(dir, item) for item in all_files if os.path.isfile(os.path.join(dir, item))]
    
    # process slices and stack into 3D volume
    slices = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in imgs]
    vol = np.stack(slices, axis=0)

    print("Done processing ct scan data")
    return vol

# see commented out segmentation function (pre removing standards) for detailed description of workflow
def segment_grains(volume, threshold, max_radius):
    binary = (volume > threshold).astype(np.uint8)

    # reduce volume edges
    binary, offsets = reduce_volume(binary, max_radius)

    # Create a 3D structuring element
    structure = generate_binary_structure(rank=3, connectivity=1)

    # Perform morphological opening to remove small white spots
    opened = binary_opening(binary, structure=structure).astype(np.uint8)

    # Perform morphological closing to fill small holes
    cleaned = binary_closing(opened, structure=structure).astype(np.uint8)

    # step 2
    print("Eroding ...")
    binary = binary_erosion(cleaned, structure=structure, iterations=max_radius)

    # step 3
    print("Applying watershed algorithm ...")
    # Compute the distance transform
    s = time.time()
    distance_transform = distance_transform_edt(binary)
    print(time.time()-s)
    s = time.time()
    # Identify local maxima as markers
    # h-maxima suppresses small peaks in the distance transform to ensure markers are distinct
    local_maxi = h_maxima(distance_transform, h=1)  # Adjust `h` based on your data
    print(time.time()-s)
    s = time.time()
    coordinates = np.argwhere(local_maxi)  # Extract coordinates of maxima
    print(time.time()-s)
    s = time.time()
    # Cluster maxima using DBSCAN
    # eps: maximum distance between points to be in the same cluster
    # min_samples: minimum points in a cluster (set to 1 for single-linkage behavior)
    db = DBSCAN(eps=10, min_samples=1).fit(coordinates)  # Adjust eps based on voxel spacing
    print(time.time()-s)
    s = time.time()
    # Assign a unique label to each cluster
    cluster_labels = db.labels_
    unique_clusters = np.unique(cluster_labels)

    # Create new markers
    new_markers = np.zeros_like(local_maxi, dtype=int)
    for cluster_id in unique_clusters:
        cluster_coords = coordinates[cluster_labels == cluster_id]
        centroid = np.mean(cluster_coords, axis=0).astype(int)  # Compute centroid
        new_markers[tuple(centroid)] = cluster_id + 1  # Assign unique label (starting from 1)

    print(time.time()-s)

    s = time.time()

    # Apply the watershed algorithm
    # Use the negative distance transform as the input to watershed
    labeled = watershed(-distance_transform, new_markers, mask=binary)

    print(time.time()-s)

    # reassign labels so that they are sequential
    print("Relabeling ...")
    labeled = _relabel_volume(labeled)
    labeled = labeled.astype(np.uint16)

    # step 4
    print("Dilating back ...")
    binary = labeled > 0 # is this necessary?
    for i in range(max_radius):
        new_binary = binary_dilation(binary, structure=structure)
        added_pixels = np.logical_xor(new_binary,binary)
        coords = np.argwhere(added_pixels)
        binary = new_binary
        for j in range(np.shape(coords)[0]):
            z, x, y = coords[j]
            # assuming we're not on the edge !! because no grains should be on the edge !!
            adjs = labeled[z-1:z+2,x-1:x+2,y-1:y+2][structure.astype(bool)]
            uniques, counts = np.unique(adjs,return_counts=True)
            # if all zeros, assign to 0
            if uniques.size == 1:
                labeled[z,x,y] = uniques[0]
            else: # the following if else can probably be reduced to just what's in the if
                if uniques[0] == 0:
                    uniques = uniques[1:] # slice off the zero
                    counts = counts[1:] # slice off the zero
                else:
                    print('no zeros')
                # assign label that has most adjacent squares. If there is a tie between labels,
                # assign 0
                # not editing the binary volume if it's 0!!!
                # Should be handled by the above code when surrounded by all 0's
                # there should only be 1 or 2 labels in unique now
                # because I don't think there are triple grain collisions
                if uniques.size == 1: # have edited uniques since above, so different.
                    # This is just to determine if there are multiple labels in the running
                    labeled[z,x,y] = uniques[0]
                elif counts[0] == counts[1]:
                    labeled[z,x,y] = 0
                else:
                    labeled[z,x,y] = uniques[np.argmax(counts)]

    # reduce volume again with smaller buffer
    labeled_reduced, new_offsets = reduce_volume(labeled, 1)

    total_offsets = np.array(offsets) + np.array(new_offsets)
    return labeled_reduced, total_offsets


# manually merges grain pairs defined by labels in pairs (array of tuples of length 2)
# in labeled_vol. For now, assumes never joining more than two labels into 1
# returns new labeled volume
def join_grains(labeled_vol, pairs):

    for pair in pairs:
        max = np.maximum(pair[0],pair[1])
        min = np.minimum(pair[0],pair[1])

        labeled_vol = labeled_vol.astype(int)

        # change all voxels labeled max to label min
        labeled_vol += (min - max)*((labeled_vol == max).astype(int))

    # relabel so that sequential again
    relabeled = _relabel_volume(labeled_vol.astype(np.uint16))
    return relabeled.astype(np.uint16)

def _relabel_volume(labeled_volume):
    # Get unique labels, excluding background (0)
    unique_labels = np.unique(labeled_volume)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude label 0
    
    # Create a mapping from old labels to new sequential labels
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}
    
    # Create a new volume for relabeled data
    relabeled_volume = np.zeros_like(labeled_volume, dtype=np.uint16)
    
    # Apply the mapping
    for old_label, new_label in label_mapping.items():
        relabeled_volume[labeled_volume == old_label] = new_label
    
    return relabeled_volume

# *** PRIVATE HELPER FUNCTION ***
# given a 3D volume and threshold, clusters connected components using depth-first search
# and labels each cluster with indices starting at 1 (0 is reserved for voxels < threshold).
# Returns the labeled volume and number of distinct clusters found
def _find_grains(volume, threshold):
    binary_volume = (volume > threshold).astype(np.uint16)

    # Label clusters in the 3D boolean array
    structure = ball(1)
    labeled_volume, n_grains = label(binary_volume, structure=structure)
    return labeled_volume, n_grains

# *** PRIVATE HELPER FUNCTION ***
# Returns status (True if no more dilating can be done given binary, False otherwise),
# the updated labeled array, and the updated dilating array
# Given a labeled array of grains, a binary array representing the voxels above a certain
# threshold, dilating, a binary array tracking the voxels that have been labeled, and a
# structuring element, dilate the grains a width of 1 pixel.
def _dilate(labeled, dilating, binary, struct):
    old_dilating = np.copy(dilating)
    dilating = binary_dilation(old_dilating, footprint=struct)
                    
    # remove dilated pixels that were not above the threshold
    dilating = binary*dilating

    added_pixels = np.logical_xor(dilating,old_dilating)
    coords = np.array(np.nonzero(added_pixels))
    if coords.size == 0:
        return True, labeled, dilating

    # else
    # assign labels to new pixels
    for i in range(np.shape(coords)[1]):
        z, x, y = coords[0,i],coords[1,i],coords[2,i]
        # assuming we're not on the edge !! because no grains should be on the edge !!
        adjs = struct*labeled[z-1:z+2,x-1:x+2,y-1:y+2]
        uniques, counts = np.unique(adjs,return_counts=True)
        uniques = uniques[1:] # slice off the zero
        counts = counts[1:] # slice off the zero
        labeled[z,x,y] = uniques[np.argmax(counts)] # biases the smaller label number
            
    return False, labeled, dilating

# *** PRIVATE HELPER FUNCTION ***
# returns updated labeled array, new number of grains (old number + n_added).
# given a labeled volume, a binary volume, and the current number of grains, searches for new
# distinct grains in binary
def _add_grains(labeled, binary, n_grains):
    reduced_vol = np.logical_xor(binary, (labeled > 0))
    new_labeled, n_added = _find_grains(reduced_vol, 0)
    labeled = ((new_labeled > 0).astype(np.uint16) * n_grains) + new_labeled + labeled
    n_total = n_added + n_grains
    return labeled, n_total

# volume should be binary or labeled volume, axes order: z,x,y
# offsets can be 0
# function name is misleading; reshapes volume so that there is exactly buffer voxels of value 0 on every side
def reduce_volume(volume, buffer):
    original_dims = np.shape(volume)

    # get bounding box of volume
    nonzero_coords = np.argwhere(volume != 0)
    zmin,xmin,ymin = nonzero_coords.min(axis=0)
    zmax,xmax,ymax = nonzero_coords.max(axis=0)

    # calculate new shape
    new_shape = (zmax-zmin+1+(2*buffer),xmax-xmin+1+(2*buffer),ymax-ymin+1+(2*buffer))

    # make new volume
    reshaped_volume = np.zeros(new_shape,dtype=volume.dtype)

    # paste in nonzero component of volume
    zlo = buffer
    zhi = zmax-zmin+1+buffer
    xlo = buffer
    xhi = xmax-xmin+1+buffer
    ylo = buffer
    yhi = ymax-ymin+1+buffer
    reshaped_volume[zlo:zhi,xlo:xhi,ylo:yhi] = volume[zmin:zmax+1,xmin:xmax+1,ymin:ymax+1]

    offsets = (zmin - buffer,xmin - buffer,ymin - buffer)
    
    return reshaped_volume, offsets


# *** OLD FUNCTIONS ***
# KEEP THIS DESCRIPTION!

# this is segmentation used pre removing standards
# segments grains using the following steps:
# 1. reduce volume with buffer of max_radius on each side
# 2. fill holes in the volume
# 3. erode max_radius times
# 4. apply watershed
# 5. Relabel volume so labels are sequential
# 6. Dilate back max_radius times
# returns final labeled volume
# def segment_grains(volume, threshold, max_radius):
#     binary = (volume > threshold).astype(np.uint8)

#     # reduce volume edges
#     binary = reduce_volume(binary, max_radius)

#     # step 1
#     print("Filling holes ...")
#     struct_3d = ball(1)
    
#     binary = binary_fill_holes(binary, structure=struct_3d).astype(np.uint8)

#     # step 2
#     print("Eroding ...")
#     binary = binary_erosion(binary, structure=struct_3d, iterations=max_radius)

#     # step 3
#     print("Applying watershed algorithm ...")
#     # Compute the distance transform
#     s = time.time()
#     distance_transform = distance_transform_edt(binary)
#     print(time.time()-s)
#     s = time.time()
#     # Identify local maxima as markers
#     # h-maxima suppresses small peaks in the distance transform to ensure markers are distinct
#     local_maxi = h_maxima(distance_transform, h=1)  # Adjust `h` based on your data
#     print(time.time()-s)
#     s = time.time()
#     coordinates = np.argwhere(local_maxi)  # Extract coordinates of maxima
#     print(time.time()-s)
#     s = time.time()
#     # Cluster maxima using DBSCAN
#     # eps: maximum distance between points to be in the same cluster
#     # min_samples: minimum points in a cluster (set to 1 for single-linkage behavior)
#     db = DBSCAN(eps=18, min_samples=1).fit(coordinates)  # Adjust eps based on voxel spacing
#     print(time.time()-s)
#     s = time.time()
#     # Assign a unique label to each cluster
#     cluster_labels = db.labels_
#     unique_clusters = np.unique(cluster_labels)

#     # Create new markers
#     new_markers = np.zeros_like(local_maxi, dtype=int)
#     for cluster_id in unique_clusters:
#         cluster_coords = coordinates[cluster_labels == cluster_id]
#         centroid = np.mean(cluster_coords, axis=0).astype(int)  # Compute centroid
#         new_markers[tuple(centroid)] = cluster_id + 1  # Assign unique label (starting from 1)

#     print(time.time()-s)
#     s = time.time()
    
#     # Apply the watershed algorithm
#     # Use the negative distance transform as the input to watershed
#     labeled = watershed(-distance_transform, new_markers, mask=binary)

#     print(time.time()-s)

#     # reassign labels so that they are sequential
#     print("Relabeling ...")
#     labeled = _relabel_volume(labeled)
#     labeled = labeled.astype(np.uint16)
    
#     # step 4
#     print("Dilating back ...")
#     binary = labeled > 0 # is this necessary?
#     for i in range(max_radius):
#         new_binary = binary_dilation(binary, structure=struct_3d)
#         added_pixels = np.logical_xor(new_binary,binary)
#         coords = np.argwhere(added_pixels)
#         binary = new_binary
#         for j in range(np.shape(coords)[0]):
#             z, x, y = coords[j]
#             # assuming we're not on the edge !! because no grains should be on the edge !!
#             adjs = labeled[z-1:z+2,x-1:x+2,y-1:y+2][struct_3d.astype(bool)]
#             uniques, counts = np.unique(adjs,return_counts=True)
#             # if all zeros, assign to 0
#             if uniques.size == 1:
#                 labeled[z,x,y] = uniques[0]
#             else: # the following if else can probably be reduced to just what's in the if
#                 if uniques[0] == 0:
#                     uniques = uniques[1:] # slice off the zero
#                     counts = counts[1:] # slice off the zero
#                 else:
#                     print('no zeros')
#                 # assign label that has most adjacent squares. If there is a tie between labels,
#                 # assign 0
#                 # not editing the binary volume if it's 0!!!
#                 # Should be handled by the above code when surrounded by all 0's
#                 # there should only be 1 or 2 labels in unique now
#                 # because I don't think there are triple grain collisions
#                 if uniques.size == 1: # have edited uniques since above, so different.
#                     # This is just to determine if there are multiple labels in the running
#                     labeled[z,x,y] = uniques[0]
#                 elif counts[0] == counts[1]:
#                     labeled[z,x,y] = 0
#                 else:
#                     labeled[z,x,y] = uniques[np.argmax(counts)]

#     # reduce volume again with smaller buffer
#     labeled_reduced = reduce_volume(labeled, 1)
    
#     return labeled_reduced

# # Segments grains in a 3D volume. Returns a labeled array and the number of grains identified
# # If to_file is set to True, labeled is saved as binary .npy file at path.
# # This function includes 4 main steps:
# # 1. find grains using hi as threshold in volume
# # 2. Dilate grains incrementally while decreasing threshold in increments of 1000,
# #    and find new grains after each incremental dilation.
# # 3. Reduce the volume to minimize blank space at edges of volume, leaving buffer on each side
# # 4. [NOT YET IMPLEMENTED] Combine grains that are touching enough
# def segment_grains_old(volume, hi, lo, threshold_step=1000, buffer=1, to_file=False, path=None):
    
#     # Step 1: find initial grains
#     print("Finding grains ...")
#     labeled, n_grains = _find_grains(volume, hi)

#     # Step 2: dilate grains and find new grains
#     print("Growing grains ...")
#     dilating_bin_vol = (volume > hi)
#     threshes = np.arange(hi-threshold_step, lo-threshold_step, -threshold_step)
#     struct_elem = ball(1)

#     for t in threshes:
#         if t > threshes[int((threshes.size) / 4)]:
#             print("25%")
#         if t > threshes[int((threshes.size) / 2)]:
#             print("50%")
#         if t > threshes[int((threshes.size)*3 / 4)]:
#             print("75%")
#         if t == threshes[-1]:
#             print("100%")
#         bin_vol = (volume > t)
#         # dilate current grains
#         status = False
#         while not status:
#             status, labeled, dilating_bin_vol = _dilate(labeled, dilating_bin_vol, bin_vol, struct_elem)
        
#         # find new grains
#         labeled, n_grains = _add_grains(labeled, bin_vol, n_grains)
#         # update dilating_bin_vol to be equal to bin_vol
#         dilating_bin_vol = bin_vol

#     # Step 3: reduce volume
#     print("Reducing volume ...")
#     labeled_reduced = reduce_volume(labeled, buffer)

#     # Step 4: Join connected grains

#     # Save to file
#     if to_file:
#         np.save(path,labeled_reduced)

#     return labeled_reduced, n_grains
