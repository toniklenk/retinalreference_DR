import numpy as np
from typing import List, Tuple

def find_clusters(
        radial_bin_significances,
        radial_bin_bs_significance,
        closest_3_position_idcs,
        sign_radial_bin_threshold: int = 1):
    bootstrap_num = radial_bin_bs_significance.shape[0]
    # Trace clusters in original signal
    _, full_indices, unique_indices = create_clusters(radial_bin_significances > 0,
                                                      closest_3_position_idcs,
                                                      sign_radial_bin_threshold)
    # Trace clusters in bootstrapped signals
    bs_cluster_full_indices = []
    bs_cluster_unique_indices = []
    for bs_idx in range(bootstrap_num):
        _, bs_full_indices, bs_unique_indices = create_clusters(radial_bin_bs_significance[bs_idx] > 0,
                                                                closest_3_position_idcs,
                                                                sign_radial_bin_threshold)
        bs_cluster_full_indices.append(bs_full_indices)
        bs_cluster_unique_indices.append(bs_unique_indices)

    return full_indices, unique_indices, bs_cluster_full_indices, bs_cluster_unique_indices

def trace_cluster(
        current_patch_idx: int,
        significant_bins: np.ndarray[bool],
        closest_indices: np.ndarray[int],
        sign_radial_bin_threshold: int,
        cluster_map: np.ndarray[bool],
        visited_patch_indices: List[int]):
    """Recursively go through all spatially connected patches
    which share the same significant direction bins
    """

    # Select bins
    current_sign_bins = significant_bins[current_patch_idx]

    # Check if enough significant bins in current patch
    if not sum(current_sign_bins) >= sign_radial_bin_threshold:
        return

    # print(f'> {current_patch_idx}')
    # Add patch data
    visited_patch_indices.append(current_patch_idx)

    # Go through adjacent patches
    for new_patch_idx in closest_indices[current_patch_idx]:

        new_sign_bins = significant_bins[new_patch_idx]
        sign_bins_in_common = current_sign_bins & new_sign_bins

        # At least <sign_radial_bin_threshold> significant direction bins in common
        if not (sum(sign_bins_in_common) >= sign_radial_bin_threshold) \
                or (new_patch_idx in visited_patch_indices):
            continue

        # Set cluster map bins for matching significant bins with this neighbor to true
        cluster_map[current_patch_idx, sign_bins_in_common] = True
        cluster_map[new_patch_idx, sign_bins_in_common] = True

        # Continue tracing for next patch
        trace_cluster(new_patch_idx,
                      significant_bins,
                      closest_indices,
                      sign_radial_bin_threshold,
                      cluster_map,
                      visited_patch_indices)

def create_clusters(
        significant_bins: np.ndarray[bool],
        closest_indices: np.ndarray[int],
        sign_radial_bin_threshold: int
) -> Tuple[np.ndarray, list, list]:
    """
    significant_bins: bool array with shape (patches x radial bins)
    closest_indices: int array with shape (patches x 3) containing indices of the three closest patches for each patch
    sign_radial_bin_threshold: int threshold for how many significant bins adjacent neighbors need to have in common
        in order to be considered "connected"

    """

    # Build cluster_idcs starting at each individual patch
    patch_num, radial_bin_num = significant_bins.shape
    cluster_maps = np.zeros((patch_num, patch_num, radial_bin_num), dtype=bool)
    visited_patch_indices = []
    for patch_start_idx in range(significant_bins.shape[0]):
        # print(f'Start at patch {patch_start_idx}')

        # Skip patch, if it is already part of another cluster
        if patch_start_idx in visited_patch_indices:
            continue

        trace_cluster(patch_start_idx,
                      significant_bins,
                      closest_indices,
                      sign_radial_bin_threshold,
                      cluster_maps[patch_start_idx],
                      visited_patch_indices)

    # Get [patch, bin] indices for all possible cluster maps
    cluster_indices = [np.argwhere(_map) for _map in cluster_maps if _map.sum() > 0]

    # Get unique patch indices
    unique_patch_indices = list({tuple(np.unique(_idcs[:, 0])): None for _idcs in cluster_indices}.keys())

    return cluster_maps, cluster_indices, unique_patch_indices

def calc_cluster_signif(
        cluster_full_indices,
        bs_cluster_full_indices,
        radial_bin_significances,
        radial_bin_bs_significances,
        cluster_alpha: float = 0.05):
    """
        Calculate 2nd order cluster statistic based on bootstrapped cluster sized
        and then select significant clusters in original data based on their
        empirical pvalue from that statistic.
        Parameters:
            cluster_full_indices, bs_cluster_full_indices:: list of np.arrays
                each array in the list represents the indices of one cluster in original data
                in 2D coordinates. len of list is (n_clusters), shape of each array is
                (cluster_size x 2D)
            bs_cluster_full_indices, bs_cluster_unique_indices:: list of lists of np.arrays
                nested list of cluster_full_indices for each bootstrapping.


    """

    radial_bin_significances = (radial_bin_significances > 0).astype(np.float64)
    radial_bin_bs_significance = (radial_bin_bs_significances > 0).astype(np.float64)

    bootstrap_num = radial_bin_bs_significance.shape[0]
    # Calculate maximum cluster scores in bootstrapped ETAs
    bs_max_cluster_scores = np.zeros(bootstrap_num)
    for bs_idx in range(bootstrap_num):
        bs_indices = bs_cluster_full_indices[bs_idx]
        bs_significances = radial_bin_bs_significance[bs_idx]

        _scores = [bs_significances[tuple(_idcs.T)].sum() for _idcs in bs_indices]
        if len(_scores) > 0:
            bs_max_cluster_scores[bs_idx] = np.max(_scores)
        else:
            bs_max_cluster_scores[bs_idx] = 0

    # Check significance of cluster scores in original ETA
    original_cluster_scores = np.array(
        [radial_bin_significances[tuple(_idcs.T)].sum() for _idcs in cluster_full_indices])

    cluster_significances = (original_cluster_scores >= bs_max_cluster_scores[:, None]).sum(
        axis=0) / bootstrap_num > 1 - cluster_alpha

    cluster_significant_indices = np.where(cluster_significances)[0]

    return original_cluster_scores, bs_max_cluster_scores, cluster_significant_indices