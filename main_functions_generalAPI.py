"""
    Functions for processing of 2-photon calcium imaging data.
    Based on previous analysis code but rewritten for improved code readability.
"""
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple
from datetime import date, datetime
import os
import h5py
import numpy as np
import scipy
from tqdm import tqdm
import time

from multiprocessing.shared_memory import SharedMemory
import concurrent.futures

def calculate_local_directions(motvecs: np.ndarray, bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
        Calculate calcium-event-triggered averages (ETA), as described in Zhang Y., Huang R., et. at (2022).
        Directional angles of motion vectors are binned in to n (default=16) bins, then motion velocities
        are averaged for each bin.
        Parameters:
            motvecs...
    """
    # Convert to angle and velocity
    motion_angles = np.arctan2(motvecs[:, :, 1], motvecs[:, :, 0])
    motion_velocities = np.linalg.norm(motvecs, axis=2)
    # Backwards transform (keep for reference):
    # motion_vectors_2d = np.zeros((*motion_angles.shape[:2], 2))
    # motion_vectors_2d[:,:,0] = motion_velocities * np.cos(motion_angles)
    # motion_vectors_2d[:,:,1] = motion_velocities * np.sin(motion_angles)

    # Calculate bin vectors for each patch and frame weighted by the local velocities
    bin_norms = motion_velocities[:, :, None] * np.logical_and(bin_edges[:-1] <= motion_angles[:, :, None],
                                                               motion_angles[:, :, None] <= bin_edges[1:])

    # Calculate ETAs
    bin_etas = np.mean(bin_norms, axis=0)

    return bin_norms, bin_etas

def bootstrap_shm(idx, ang_name, ang_shape, ang_dtype, vel_name, vel_shape, vel_dtype, bins):
    """
        Worker for one bootstrap repetition used in calculate_reverse_correlations_shm
        in parallel processing. Uses shared memory for optimizing runtime.
        This function implements the same as def calculate_local_directions(..): but adapted for this cause.
    """
    # access shared memory
    shm_ang = SharedMemory(name=ang_name)
    ang = np.ndarray(ang_shape, dtype=ang_dtype, buffer=shm_ang.buf)[idx]

    shm_vel = SharedMemory(name=vel_name)
    vel = np.ndarray(vel_shape, dtype=vel_dtype, buffer=shm_vel.buf)[idx]

    # calculate calcium-event-triggered averge (ETA)
    return np.mean(vel[:, :, None] * np.logical_and(bins[:-1] <= ang[:, :, None], ang[:, :, None] <= bins[1:]),
                   axis=0)

def calculate_reverse_correlations_shm_generalAPI(
        motion_vectors: np.array,
        signal: np.array,
        cmn_phases,
        sample_rate,
        radial_bin_edges,
        bootstrap_num: int = 1024,
        num_workers: int = 12
):
    """
        Calculate original and bootstrapped calcium event triggered averages (ETA's).

        Parameters:
            motion_vectors: np.array (float) (time x n_radial_bins x 2)
                typical dimensions; (~30 000, 320, 2)
                cmn_motion_vectors_2d. CMN motion vectors projected to 2D. ETA calculation is done fully in 2D.
            signal: np.arrary (bool) (time x 1)
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
                while still leaving enough ressources to keep using PC while script is running on a 14 core cpu.
        Returns:
            radial_bin_etas:
                true ETAs
            radial_bin_bs_etas:
                bootstrapped ETAs

    """
    # calculate true ETAs =============================
    radial_bin_norms, radial_bin_etas = calculate_local_directions(
        motion_vectors[signal, :, :],
        radial_bin_edges)
    # =================================================

    """
      Calculate bootstrapped ETAs
        - precompute all angles and velocities once
        - shared memory for all parallel processes, no problem bc its only read by bootstrapping function.
        - np.float32 because it's fastest float type on common processor architectures
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
    min_frame_shift = 4 * sample_rate
    max_frame_shift = int(frame_no - min_frame_shift)
    frame_shifts = np.random.randint(min_frame_shift, max_frame_shift, size=(bootstrap_num))
    signal_indices = signal_within_cmn_selection.nonzero()[0][:, None]
    idcs = np.mod(signal_indices + frame_shifts, signal_within_cmn_selection.size).T
    start_time = time.time() # time parallel computation

    # adjust maximum number of processes as needed, leave blank to use all kernels
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = [exe.submit(
            bootstrap_shm,
            idcs[i],
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
    # =================================================

    return radial_bin_etas, radial_bin_norms, radial_bin_bs_etas
