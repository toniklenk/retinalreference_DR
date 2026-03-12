"""
    Functions for processing of 2-photon calcium imaging data.
    Based on previous analysis code but rewritten for improved code readability.
"""
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple
from datetime import date, datetime
import os, time, scipy
import numpy as np
from tqdm import tqdm
from main_functions import create_clusters, project_to_local_2d_vectors

from multiprocessing.shared_memory import SharedMemory
import concurrent.futures

# def detect_events_with_derivative(recording, test_neuron_dff, excluded_percentile: int = 25,
#                                   kernel_sd: float = 0.5):
def detect_events_with_derivative_generalAPI(
        cmn_selection, # timepoints with CMN stimulus
        dff,
        sample_rate,
        excluded_percentile: int = 25,
        kernel_sd: float = 0.5):
    """
        Detect calcium-events in raw flourescence trace. Done with a different, more simple method
         than in the 2019 paper.

        Parameters:
            cmn_selection: np.array (timepoints x 1)
                Boolean mask of timepoints with CMN stimulus
            dff: np.array (timepoints x 1)
                raw fluorescence trace of one neuron
            sample_rate: float
                sample rate of the fluorescence trace
            excluded_percentile: int
            kernel_sd: float
                determines smoothing of signal
        Returns:
            signal_selection: np.array (timepoints x 1)
            signal_length: int
            signal_proportion: float
            signal_dff_mean: float
    """
    # init Gaussian kernel with mean=0 and standard deviation 0.5 (adjustable)
    kernel_dts = 10 * sample_rate # time accuracy of kernel
    kernel_t = np.linspace(-5, 5, kernel_dts)
    norm_kernel = scipy.stats.norm.pdf(kernel_t, scale=kernel_sd)

    # pad, then Smoothen DFF with Gaussian kernel
    dff_plot_pad = np.zeros(dff.shape[0] + kernel_dts - 1)
    dff_plot_pad[kernel_dts // 2:-kernel_dts // 2 + 1] = dff
    dff_conv = scipy.signal.convolve(dff_plot_pad, norm_kernel, mode='valid', method='fft')

    # Calculate derivative
    diff = np.diff(dff_conv, append=[0])

    # pad, then Smoothen derivative
    diff1_pad = np.zeros(diff.shape[0] + kernel_dts - 1)
    diff1_pad[kernel_dts // 2:-kernel_dts // 2 + 1] = diff
    diff1_conv = scipy.signal.convolve(diff1_pad, norm_kernel, mode='valid', method='fft')

    # Calculate signal based on smoothened derivative
    signal_selection = (diff1_conv > 0) & cmn_selection
    # Exclude bottom 25% percentile
    signal_selection &= dff >= np.percentile(dff[signal_selection], excluded_percentile)

    return signal_selection, sum(signal_selection), sum(signal_selection)/sum(cmn_selection), np.mean(dff[signal_selection])

def generate_eyepos_masks(
        eyepos_left,
        eyepos_right,
        eyepos_time,
        time_resampled,
        q1_min_left,
        q1_min_right,
        q3_max_left,
        q3_max_right):
    """
        Generate maks for selecting data that corresponds to eyepositions to the left and right, respectively.
        Eye positions are resampled to given timeline before selection.
        Parameters:
            eyepos_left, eyepos_right: np.array (timepoints x 1)
                Eye positions at each timepoint.
            eyepos_time: np.array (timepoints x 1)
                Original time axis of eye positions.
            time_resampled:
                Timeline to resample eyepositions to.
            q1_min_left, q1_min_right: float
                threshold for left and right eye to be on one side
            q3_max_left, q3_max_right: float
                threshold for left and right eye to be on the other side.
    """
    eye_pos = np.column_stack((
        np.array(eyepos_left).squeeze(),
        np.array(eyepos_right).squeeze()))

    eye_pos_resampled = scipy.interpolate.interp1d(
        np.array(eyepos_time).squeeze(),
        eye_pos.T,
        kind='nearest')(
        time_resampled
    ).T
    # select data in quadrants
    q1_mask = np.logical_and(
        eye_pos_resampled[:, 0] > q1_min_left,
        eye_pos_resampled[:, 1] > q1_min_right)
    q3_mask = np.logical_and(
        eye_pos_resampled[:, 0] < q3_max_left,
        eye_pos_resampled[:, 1] < q3_max_right)

    return q1_mask, q3_mask


def calculate_local_directions_generalAPI(motion_vectors: np.ndarray, bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
        Calculate calcium-event-triggered averages (ETA), as described in Zhang Y., Huang R., et. at (2022).
        Motion velocities are binned in to n (default=16) bins by their corresponding motion angles,
        and then averaged to yield ETAs.
        Parameters:
            motion_vectors: np.array
                Motion vectors at timepoints with calcium-event.

    """
    angles = np.arctan2(motion_vectors[:, :, 1], motion_vectors[:, :, 0])
    velocities = np.linalg.norm(motion_vectors, axis=2)
    # Backwards transform (keep for reference):
    # motion_vectors_2d = np.zeros((*motion_angles.shape[:2], 2))
    # motion_vectors_2d[:,:,0] = motion_velocities * np.cos(motion_angles)
    # motion_vectors_2d[:,:,1] = motion_velocities * np.sin(motion_angles)

    # Calculate bin vectors for each patch and frame weighted by the local velocities
    bin_norms = velocities[:, :, None] * np.logical_and(
        angles[:, :, None] >= bin_edges[:-1],
        angles[:, :, None] < bin_edges[1:])

    # norms, ETAs
    return bin_norms, bin_norms.mean(axis=0)

def bootstrap_shm(event_train, ang_name, ang_shape, ang_dtype, vel_name, vel_shape, vel_dtype, bins):
    """
        Worker for one bootstrap repetition used in calculate_reverse_correlations_shm
        in parallel processing. Uses shared memory for optimizing runtime.
        This function implements the same as def calculate_local_directions(..): but adapted for this cause.
    """
    # access shared memory
    shm_ang = SharedMemory(name=ang_name)
    ang = np.ndarray(ang_shape, dtype=ang_dtype, buffer=shm_ang.buf)[event_train]

    shm_vel = SharedMemory(name=vel_name)
    vel = np.ndarray(vel_shape, dtype=vel_dtype, buffer=shm_vel.buf)[event_train]

    # calculate calcium-event-triggered averge (ETA)
    ETA = np.mean(vel[:, :, None] * np.logical_and(bins[:-1] <= ang[:, :, None], ang[:, :, None] <= bins[1:]),
            axis=0)
    # release shared memory (but dont unlink?)
    shm_ang.close()
    shm_vel.close()
    return ETA

def calculate_radial_bin_bs_etas(
        motion_vectors: np.ndarray,
        signal: np.ndarray,
        cmn_phases,
        sample_rate,
        radial_bin_edges,
        bootstrap_num: int = 1024,
        num_workers: int = 12
):
    """
        Calculate bootstrapped distribution of calcium event triggered averages (ETA's).
        - precompute all angles and velocities once
        - shared memory for all parallel processes, no problem bc its only read by bootstrapping function
        - np.float32 because it is the fastest float type on common processor architectures

        Parameters:
            motion_vectors: np.array (float) (time x n_radial_bins x 2)
                typical dimensions; (~30 000, 320, 2)
                cmn_motion_vectors_2d. CMN motion vectors projected to 2D. ETA calculation is done fully in 2D.
            signal: np.array (bool) (time x 1)
                indicates timepoints with calcium events during CMN stimulation periods.
            cmn_phases: np.array (bool) (time x 1)
                indicates timepoints with cmn stimulation.
            sample_rate: float
                display rate of the CMN stimulus, sampling rate for all time axes used in this function.
            radial_bin_edges: np.array (n_bins, 1)
                angles for bin edges of binning used in ETA calculations. note: these are angular bins (not spatial).
            bootstrap_num: int
                number of bootstrapping iterations.
            num_workers: int
                number of parallel processes in computing bootstrapping repetitions. 12 should be fast
                while still leaving enough resources to keep using PC while script is running on a 14 core cpu.
        Returns:
            radial_bin_etas:
                true ETAs
            radial_bin_bs_etas:
                bootstrapped ETAs

    """
    motion_vectors_cmn = motion_vectors[cmn_phases]
    signal_within_cmn_selection = signal[cmn_phases]

    angles = np.arctan2(
        motion_vectors_cmn[:, :, 0],
        motion_vectors_cmn[:, :, 1]
    ).astype(np.float32)
    velocities = np.linalg.norm(motion_vectors_cmn, axis=2).astype(np.float32)
    radial_bin_edges = radial_bin_edges.astype(np.float32)

    ang_shm = SharedMemory(create=True, size=angles.nbytes)
    ang_shared = np.ndarray(angles.shape, dtype=angles.dtype, buffer=ang_shm.buf)
    ang_shared[:] = angles

    vel_shm = SharedMemory(create=True, size=velocities.nbytes)
    vel_shared = np.ndarray(velocities.shape, dtype=velocities.dtype, buffer=vel_shm.buf)
    vel_shared[:] = velocities

    # calculate permutations
    frame_no = cmn_phases.sum()
    min_frame_shift = 4 * sample_rate # TODO; this was only 2x in original paper
    max_frame_shift = int(frame_no - min_frame_shift)
    frame_shifts = np.random.randint(min_frame_shift, max_frame_shift, size=(bootstrap_num))
    signal_indices = signal_within_cmn_selection.nonzero()[0][:, None]
    event_trains = np.mod(signal_indices + frame_shifts, signal_within_cmn_selection.size).T

    start_time = time.time() # time parallel computation
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = [exe.submit(
            bootstrap_shm,
            event_trains[i],
            ang_shm.name,
            angles.shape,
            angles.dtype,
            vel_shm.name,
            velocities.shape,
            velocities.dtype,
            radial_bin_edges,
        )
            for i in range(bootstrap_num)]

        # Calculate vector ETAs for each local radial bin
        radial_bin_bs_etas = np.array([f.result() for f in futures])
    print("--- %s seconds ---" % (time.time() - start_time)) # time parallel computation
    # release shared memory
    ang_shm.close()
    vel_shm.close()
    ang_shm.unlink()
    vel_shm.unlink()
    return radial_bin_bs_etas

def calculate_directional_significance_generalAPI(
        radial_bin_etas,
        radial_bin_bs_etas,
        bernoulli_alpha: float = 0.05):
    bootstrap_num = radial_bin_bs_etas.shape[0]

    cdf_values = ((radial_bin_etas > radial_bin_bs_etas).sum(axis=0) / bootstrap_num)
    greater_than = cdf_values > 1 - bernoulli_alpha / 2
    less_than = cdf_values < bernoulli_alpha / 2
    radial_bin_p_values = cdf_values.copy()
    radial_bin_p_values[greater_than] = 1 - cdf_values[greater_than]

    radial_bin_significances = np.zeros_like(radial_bin_p_values)
    radial_bin_significances[greater_than] = 1
    radial_bin_significances[less_than] = -1

    return radial_bin_significances, radial_bin_p_values

def calculate_directional_significance_permutations_generalAPI(
        radial_bin_bs_etas: np.ndarray,):
    radial_bin_bs_significances = np.zeros_like(radial_bin_bs_etas, dtype=np.int64)
    radial_bin_bs_p_values = np.zeros(radial_bin_bs_etas.shape)
    for i, bs_etas in enumerate(radial_bin_bs_etas):
        significances, p_values = calculate_directional_significance_generalAPI(
            bs_etas,
            radial_bin_bs_etas,
        )
        radial_bin_bs_p_values[i] = p_values
        radial_bin_bs_significances[i] = significances
    return radial_bin_bs_significances, radial_bin_bs_p_values


def calc_preferred_directions_generalAPI(
        bin_etas: np.ndarray,
        bin_significances: np.ndarray,
        bin_centers: np.ndarray) -> np.ndarray:
    """
        Calculate estimated RF.
        Calculates preferred direction of each radial patch/bin as weighted sum of significant directions.
        Weigh each direction by its calcium-event-triggered average (ETA).
        Influence of ETA absolute values is normed out.
        (note: this is a different binning than the one done during calculation of the ETA.)

        Parameters:
            bin_etas: shape (patch_num, bin_num)
                calcium-event-triggered averages
            bin_significances: shape (patch_num, bin_num)
                1 if bin is significantly excitatory
                -1 if bin is significantly suppressive
                0 if bin is not significant
            bin_centers: shape (bin_num,)
                radial bin centers. centers of the radial bins in which angles are divided in ETA calculation.
                bin_num = 16 in the default configuration

        Returns:
            population_vectors: shape (patch_num,)
    """

    # Calculate direction vectors for given angles
    direction_vectors = np.array([[np.cos(a), np.sin(a)] for a in bin_centers])

    # Calculate population vector for each patch based on significant direction bins
    population_vectors = np.zeros(bin_etas.shape[:1] + (2,))
    for idx, (etas, signs) in enumerate(zip(bin_etas, bin_significances)):

        if np.any(signs):

            # Select excitatory bins
            idcs = np.where(signs)[0]

            # Calculate
            # normed in terms of ETA, this means that large vectors arise not from high ETA,
            # but from very coherent ETAs in one bin (weird a bit)
            # this means that vecs_pop are not unitary vectors!
            vecs = etas[idcs][:, None] * direction_vectors[idcs]
            vec_pop = np.sum(vecs, axis=0) / np.sum(etas)  # TODO: might need to do etas[idcs] instead

        else:
            vec_pop = np.array([0, 0])

        # Append
        population_vectors[idx] = vec_pop

    return population_vectors

def find_clusters_generalAPI(
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

def calculate_cluster_significances_generalAPI(
        cluster_full_indices,
        bs_cluster_full_indices,
        readial_bin_significances,
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
                nested list of cluster_full_indices for each bootsrapping.


    """

    radial_bin_significances = (readial_bin_significances > 0).astype(np.float64)
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

def FE_similarity(F, E):
    """
        Calculate similarity between F and E according to the definition in Zhanng et. at. 2022
        Parameters:
            E: Estimated receptive field
            F: Optic flow field fitted to E
    """
    return ((np.sum(F*E, axis=1)/
            (np.linalg.norm(np.clip(E, 0.0000001, 1.), axis=1) *
             np.linalg.norm(F, axis=1)))
            .mean())


def tof(angle_azimuth: float, angle_elevation: float, P: np.ndarray):
    """
        Translational optic flow field.
        Returns a translational optic flow field for a given translation axis.

        Parameters:
            angle_azimuth (float):
                angle in radians of the azimuth of the translation axis.
            angle_elevation (float):
                angle in radians of the elevation of the translation axis.
            P (np.array):
                3D positions to sample flow field at.

        Returns:
            F (np.array):
                Return values of optic flow field (3D vector) at each point
                given in positions.
            FoC, FoE (np.array):
                Focus of Expansion and Focus of Contraction of the optic flow field.
                Given as 3D unit vector.
    """
    from numpy import sin, cos
    axis, a, b = np.array([1,0,0]), angle_azimuth, angle_elevation
    # yaw and pitch rotation matrices
    yaw=np.array([[cos(a),-sin(a),0],[sin(a), cos(a), 0],[0,0,1]])
    pitch=np.array([[cos(b), 0, sin(b)],[0,1,0],[-sin(b), 0, cos(b)]])
    T=axis @ yaw @ pitch
    return T - np.dot(P, T)[:,None] * P

def rof(angle_azimuth: float, angle_elevation: float, P: np.ndarray):
    """
        Rotational optic flow field.
        Returns a rotation optic flow field for a given translation axis.

        Parameters:
            angle_azimuth (float):
                angle in radians of the azimuth of the rotation axis.
            angle_elevation (float):
                angle in radians of the elevation of the rotation axis.
            P (np.array):
                3D positions to sample flow field at.

        Returns:
            F (np.array):
                Return values of optic flow field (3D vector) at each point
                given in positions.
            FoC, FoE (np.array):
                Focus of Expansion and Focus of Contraction of the optic flow field.
                Given as 3D unit vector.
    """
    from numpy import sin, cos
    axis, a, b = np.array([1,0,0]), angle_azimuth, angle_elevation
    # yaw and pitch rotation matrices
    yaw=np.array([[cos(a),-sin(a),0],[sin(a), cos(a), 0],[0,0,1]])
    pitch=np.array([[cos(b), 0, sin(b)],[0,1,0],[-sin(b), 0, cos(b)]])
    return -np.cross(P, axis @ yaw @ pitch)

def RSSangle(F, E):
    """
        Calculate RSS (residual sum of squared) of angles between two vector fields.
        Used in this analysis to calculate RSS between estimated RF and optic flow field,
        as an error function to fit the latter to the former.
        Parameters:
            F (np.array):
                Optic flow field to fit.
            E (np.array):
                Estimated RF.
        Returns:
            RSS (float)
    """
    #print(E)
    coeffcients = np.sum(F*E, axis=1)/ (np.linalg.norm(np.clip(E, 0.0000001, 1.), axis=1)* np.linalg.norm(F, axis=1))
    angles = np.arccos(np.clip(coeffcients, -1.0, 1.0))
    return np.sum(angles**2)

def RSSangle_Fto2D(F, E, pos):
    """
        Wraps around def RSSangle(F, E) to transform F from 3D to 2D,
        for convenience in using it with scipy.optimize.minimize
    """
    F=project_to_local_2d_vectors(pos, F[None,:,:]).squeeze()
    return RSSangle(F, E)
