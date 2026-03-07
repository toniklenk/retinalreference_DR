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



def animal_id_from_path(path: Union[Path, str]) -> str:
    return Path(path).as_posix().split('/')[-2]


def recording_id_from_path(path: Union[Path, str]) -> Tuple[str, str]:
    return Path(path).as_posix().split('/')[-1].split('_')  # type: ignore


def parse_date(_date: Union[str, datetime, date]):
    if isinstance(_date, str):
        _date = datetime.strptime(_date, '%Y-%m-%d')

    if isinstance(_date, datetime):
        _date = _date.date()

    return _date

def calculate_ca_frame_times(mirror_position: np.ndarray, mirror_time: np.ndarray, imaging_rate: float):
    mirror_position = np.squeeze(mirror_position)
    mirror_time = np.squeeze(mirror_time)

    # Find all the indices where the value is 1 (these are the frame trigger points)
    frame_indices = mirror_time[np.where(mirror_position == 1)[0]]

    first_time = frame_indices[0]
    end_time = frame_indices[-1]

    # Get corresponding times
    frame_times = np.arange(first_time, end_time, 1 / imaging_rate)
    return frame_times\

def calculate_ca_frame_times_experimental(mirror_position: np.ndarray, mirror_time: np.ndarray, imaging_rate: float):
    mirror_position = np.squeeze(mirror_position)
    mirror_time = np.squeeze(mirror_time)

    # Find all the indices where the value is 1 (these are the frame trigger points)
    # version 1: uncomment this if data is from downstairs MOM microscope setup
    #peak_idcs = np.where(np.diff(mirror_position,prepend=mirror_position[0]) == True)[0] # np.diff of boolean array stays boolean
    frame_indices = mirror_time[np.where(mirror_position == True)[0]]
    first_time = frame_indices[0]
    end_time = frame_indices[-1]

    # Find all the indices where the value is 1 (these are the frame trigger points)
    # version 2: uncomment this if data is from upstairs microscope setup
    # peak_prominence = (mirror_position.max() - mirror_position.min()) / 4
    # peak_idcs, _ = scipy.signal.find_peaks(mirror_position, prominence=peak_prominence)

    # calculate manually
    #first_peak = peak_idcs[0]
    #end_peak = peak_idcs[-1] # -2 in downstairs microscope, -1 in upstairs

    #first_time = mirror_time[first_peak]
    #end_time = mirror_time[end_peak]

    # Get corresponding times
    frame_times = np.arange(first_time, end_time, 1 / imaging_rate)
    return frame_times


def digest_folder(recording_path: Union[Path, str], plane: int = 0):
    """
    Reads in data from suite2p's F.npy file and the Display.h5py and Io.h5py file

    Parameters:
        recording_path (Union[Path, str])

        plane (int): index of recorded plane to be digested

    Returns:
        flourescence: flourescence traces (ROIs x Timepoints)

        recording (dict): contains
                    stimulus data defining the CMN
                    time axis of whole recording resampled to 10Hz

        phase (dict): contains
                    timing information for each phase of the recording, such as
                        indices of phase in time array of whole recording
                        start_idx, end_idx, size
                    type information, about what stimulus was shown during that phase
                        CMN_no_foreground, CMN1/2/3/4, Translation/Rotation grating, SphereUniformBackground

        ca_rec_group_id_fun (function): calculates
                    1d interpolation of group id at given time from nearest neighbor in recording timeline
                    (scipy.interpolate.interp1d(record_group_ids_time, record_group_ids, kind='nearest'))

    """
    animal_id = animal_id_from_path(recording_path)
    rec_date, rec_id, *_ = recording_id_from_path(recording_path)
    rec_date = parse_date(rec_date)

    recording = {}
    recording['animal_id'] = animal_id
    recording['rec_date'] = rec_date
    recording['rec_id'] = rec_id

    s2p_path = os.path.join(recording_path, 'suite2p', 'plane' + str(plane))
    print(s2p_path)

    print('Calculate frame timing of Fluorescence')
    with h5py.File(os.path.join(recording_path, 'Io.hdf5'), 'r') as io_file:
        #mirror_position = np.squeeze(io_file['ai_y_mirror_in'])[:]
        #mirror_time = np.squeeze(io_file['ai_y_mirror_in_time'])[:]

        mirror_position = np.squeeze(io_file['di_frame_sync'])[:]
        mirror_time = np.squeeze(io_file['di_frame_sync_time'])[:]

        # Calculate frame timing
        ca_times = calculate_ca_frame_times(mirror_position, mirror_time, imaging_rate=1.9989) # example data=2.1798
        record_group_ids = io_file['__record_group_id'][:].squeeze()
        record_group_ids_time = io_file['__time'][:].squeeze()

        ca_rec_group_id_fun = scipy.interpolate.interp1d(record_group_ids_time, record_group_ids,
                                                         kind='nearest')

    print('Load ROI data')
    fluorescence = np.load(os.path.join(s2p_path, 'F.npy'), allow_pickle=True)

    # Check if frame times and fluorescence length match
    if ca_times.shape[0] != fluorescence.shape[1]:
        print(f'Detected frame times\' length doesn\'t match frame count. '
              f'Detected frame times ({ca_times.shape[0]}) / Frames ({fluorescence.shape[1]})')
        exit()

    imaging_rate = 1. / np.mean(np.diff(ca_times))  # Hz # average of the time intervals between adjacent frames
    print(f'Estimated imaging rate {imaging_rate:.2f}Hz')

    # Save to recording
    recording['roi_num'] = fluorescence.shape[0]
    recording['signal_length'] = fluorescence.shape[1]
    recording['imaging_rate'] = imaging_rate
    recording['ca_times'] = ca_times
    record_group_ids = ca_rec_group_id_fun(
        ca_times)
    recording['record_group_ids'] = record_group_ids

    print('Add ROI stats and signals')
    # upper sample
    sample_rate = 10
    time_resampled = np.arange(ca_times[0], ca_times[-1], 1 / sample_rate)
    recording[f'time_resampled'] = time_resampled
    recording[f'sample_rate'] = sample_rate
    frame_relative_time_whole_trace = np.zeros_like(ca_times) * np.nan

    print('Add display phase data')
    with h5py.File(os.path.join(recording_path, 'Display.hdf5'), 'r') as disp_file:
        # Get attributes
        recording.update({f'display/attrs/{k}': v for k, v in disp_file.attrs.items()})
        phase = {}
        for key1, member1 in tqdm(disp_file.items()):

            # If dataset, write to file
            if isinstance(member1, h5py.Dataset):
                recording[f'display/{key1}'] = member1[:]
                continue

            if 'phase' in key1:
                phase_id = int(key1.replace('phase', ''))

                # Add calcium start/end indices
                in_phase_idcs = np.where(recording['record_group_ids'] == phase_id)[0] #indices of data from that phase
                phase[f'in_phase_idcs_{phase_id}'] = in_phase_idcs

                # i think this is just some corrections for in_phase_idcs[0] or [-1]
                start_index = np.argmin(np.abs(
                    ca_times - ca_times[in_phase_idcs[0]]))
                end_index = np.argmin(np.abs(ca_times - ca_times[in_phase_idcs[-1]]))
                phase[f'ca_start_index_{phase_id}'] = start_index
                phase[f'ca_end_index_{phase_id}'] = end_index

                # Write attributes
                phase[f'phase_{phase_id}'] = {}
                phase[f'phase_{phase_id}'].update({k: v for k, v in member1.attrs.items()})

                # Write datasets
                for key2, member2 in member1.items():
                    if isinstance(member2, h5py.Dataset):
                        phase[f'{phase_id}/{key2}'] = member2[:]

                if 'CMN' in phase[f'phase_{phase_id}']['__visual_name']:
                    if 'CMN_no_foreground' in phase[f'phase_{phase_id}']['__visual_name']:
                        phase[f'frame_relative_time_phase_{phase_id}'] = np.zeros_like(in_phase_idcs) * np.nan
                        for k, index in enumerate(in_phase_idcs):
                            relative_time_index = np.argmin(np.abs(phase[f'{phase_id}/__time'] - ca_times[index]))
                            phase[f'frame_relative_time_phase_{phase_id}'][k] = (phase[f'{phase_id}/time'][relative_time_index])
                        frame_relative_time_whole_trace[in_phase_idcs] = phase[f'frame_relative_time_phase_{phase_id}']

            # Add other data
            # CMN data
            else:
                # Write attributes
                recording.update({f'display/{key1}/{k}': v for k, v in member1.attrs.items()})

                # Get datasets
                for key2, member2 in member1.items():
                    if isinstance(member2, h5py.Dataset):
                        recording[f'display/{key1}/{key2}'] = member2[:]

    return fluorescence, recording, phase, ca_rec_group_id_fun


def process_recording(recording, phase, radial_bin_num: int = 16):
    """
        Adapt CMN stimuli to resampling.
    """
    # Go through all CMN phases and add CMN data to resampled time domain
    ca_times = recording['ca_times']
    time_resampled = recording['time_resampled']
    phase_num = recording['display/protocol0/__target_phase_count']
    cmn_phase_selection_original = np.zeros_like(recording['ca_times'], dtype=bool)
    cmn_phase_selection = np.zeros_like(recording['time_resampled'], dtype=bool)
    cmn_motion_vectors_3d = None
    cmn_motion_vectors_3d_original = None
    positions = None
    patch_corners = None
    patch_indices = None
    print('Pick from motion vectors matrix to form cmn_motion_vectors_3d')
    for n in tqdm(range(phase_num)):

      if 'CMN' not in phase[f'phase_{n}']['__visual_name']:
        continue
      if 'CMN3D20240606Vel140Scale7Long' in phase[f'phase_{n}']['__visual_name']:
        cmn_start_time = phase[f'phase_{n}']['__start_time']
        cmn_end_time = cmn_start_time + phase[f'phase_{n}']['__target_duration']

        # Add CMN phase to selection
        selection_original = (cmn_start_time <= ca_times) & (ca_times <= cmn_end_time)
        cmn_phase_selection_original |= selection_original
        selection = (cmn_start_time <= time_resampled) & (time_resampled <= cmn_end_time)
        cmn_phase_selection |= selection

        # Get CMN base data
        cmn_name = phase[f'phase_{n}']['__visual_name']
        positions = recording[f'display/{cmn_name}/centers_0'][:].squeeze()
        patch_corners = recording[f'display/{cmn_name}/vertices_0'][:].squeeze()
        patch_indices = recording[f'display/{cmn_name}/indices_0'][:].squeeze()

        # Get CMN phase data
        frame_indices = phase[f'{n}/frame_index'].squeeze()
        frame_times = phase[f'{n}/__time'].squeeze()
        motion_vectors_full = recording[f'display/{cmn_name}/motion_vectors_0'][:].squeeze()

        # Find corresponding CMN indices
        indices = [np.argmin(np.abs(frame_times - t)) for t in time_resampled[selection]]

        # Update motion vectors
        if cmn_motion_vectors_3d is None:
            cmn_motion_vectors_3d = np.nan * np.ones(time_resampled.shape + (motion_vectors_full.shape[1:]))
        cmn_motion_vectors_3d[selection] = motion_vectors_full[frame_indices[indices]]

        # Find corresponding CMN indices
        indices_original = [np.argmin(np.abs(frame_times - t)) for t in ca_times[selection_original]]

        # Update motion vectors
        if cmn_motion_vectors_3d_original is None:
            cmn_motion_vectors_3d_original = np.nan * np.ones(ca_times.shape + (motion_vectors_full.shape[1:]))
        cmn_motion_vectors_3d_original[selection_original] = motion_vectors_full[frame_indices[indices_original]]

    recording[f'positions'] = positions # center of CMN stimulus
    recording[f'patch_corners'] = patch_corners
    recording[f'patch_indices'] = patch_indices
    recording[f'cmn_phase_selection_original'] = cmn_phase_selection_original
    recording[f'cmn_phase_selection'] = cmn_phase_selection
    recording[f'cmn_motion_vectors_3d'] = cmn_motion_vectors_3d # (time_resampled, motion_vectors.shape[1:])
    recording[f'cmn_motion_vectors_2d'] = project_to_local_2d_vectors(positions, cmn_motion_vectors_3d)
    recording[f'cmn_motion_vectors_3d_original'] = cmn_motion_vectors_3d_original
    recording[f'cmn_motion_vectors_2d_original'] = project_to_local_2d_vectors(positions,
                                                                               cmn_motion_vectors_3d_original)

    # Set radial bins
    radial_bin_edges = np.linspace(-np.pi, np.pi, radial_bin_num + 1)
    radial_bin_centers = radial_bin_edges[:-1] + (radial_bin_edges[1] - radial_bin_edges[0]) / 2
    recording['radial_bin_num'] = radial_bin_centers.shape[0]
    recording['radial_bin_edges'] = radial_bin_edges
    recording['radial_bin_centers'] = radial_bin_centers

    # Calculate pairwise distances between all patches and select the three adjacent ones for each patch
    patch_num = positions.shape[0]
    pairwise_distances = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :],
                                        axis=-1)
    closest_3_position_indices = np.zeros((patch_num, 3), dtype=np.int64)
    for patch_idx in range(patch_num):
        closest_3_position_indices[patch_idx] = np.argsort(pairwise_distances[patch_idx])[
                                                1:4]
    recording['closest_3_position_indices'] = closest_3_position_indices


def crossproduct(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Workaround because of NoReturn situation in numpy.cross"""
    return np.cross(v1, v2)


def project_to_local_2d_vectors(normals: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
        Parameters:
                normals: np.array
                    centers of CMN stimuli

                vectors: np.array
                    3D CMN motion vectors
    """
    vnorms = np.array([0, 0, 1]) - normals * np.dot(normals, np.array([0, 0, 1]))[:, None]
    vnorms /= np.linalg.norm(vnorms, axis=1)[:, None]

    hnorms = -crossproduct(vnorms, normals)
    hnorms /= np.linalg.norm(hnorms, axis=1)[:, None]

    vectors_2d = np.zeros((*vectors.shape[:2], 2))
    for i, v in enumerate(vectors):
        # Calculate 2d motion vectors in coordinate system defined by local horizontal and vertical norms
        motvecs_2d = np.array([np.sum(v * hnorms, axis=1),
                               np.sum(v * vnorms, axis=1)])
        vectors_2d[i] = motvecs_2d.T

    return vectors_2d


def calculate_dff(recording, fluorescences, imaging_rate, window_size: int = 120, percentile: int = 10):
    """
        Calculate the delF/F signal at each timepoint. (Normalized fluorescence change relative to a baseline.)

        Parameters:
            recording (dict):
                    Contains arrays with original and resampled times.
                    Resampled dff is saved to this object.
            fluorescences(array):
                    Array of fluorescence traces (ROIs/neurons x timepoints)
            imaging_rate (float):
                    original sample rate from microscope as saved in recording object.
            window_size (int):
                    window size in time for moving average over individual fluorescence traces.
            percentile (int):
                    percentile from which mean is calculated as baseline.

        Returns:
            Dff_all (array):
                    Array with shape (Neurons x Timepoints, same as fluorescence).
                    delF/F normalized measure of fluorescence.
            Dff_resampled (array):
                    Same as Dff_all, but resampled to new sampling rate (10Hz for now).
    """
    print('Calculate dff')
    # Calculate DFF with local low baseline
    window_size = int(window_size * imaging_rate)
    if window_size % 2 == 0:
        window_size += 1
    half_window_size = int((window_size - 1) // 2)

    # Calculate for each signal datapoint
    Dff_all = np.zeros_like(fluorescences)
    for n in tqdm(range(Dff_all.shape[0])):
        single_neuron = fluorescences[n]
        # Pad fluorescence data with the median of the first and last segments
        f_padded = np.pad(single_neuron, (half_window_size, half_window_size), mode='constant', constant_values=(
            np.median(single_neuron[:half_window_size]), np.median(single_neuron[-half_window_size:])))
        single_neuron_dff = np.zeros(single_neuron.shape[0])
        baseline = np.zeros(single_neuron.shape[0])
        for i in range(single_neuron.shape[0]):
            subfluores = f_padded[i:i + window_size]
            baseline[i] = np.mean(subfluores[subfluores < np.percentile(subfluores, percentile)])
            single_neuron_dff[i] = (single_neuron[i] - baseline[i]) / baseline[i]
        Dff_all[n] = single_neuron_dff
    Dff_resampled = scipy.interpolate.interp1d(recording['ca_times'], Dff_all, kind='nearest')(
        recording['time_resampled'])
    recording['Dff_resampled'] = Dff_resampled
    return Dff_all, Dff_resampled

def calculate_dff_vectorized(recording, fluorescences, imaging_rate, window_size: int = 120, percentile: int = 10):
    """
        Vectorized version of calculate_dff.
    """
    print('Calculate dff')
    from numpy.lib.stride_tricks import sliding_window_view

    window_size = int(window_size * imaging_rate)
    if window_size % 2 == 0:
        window_size += 1
    half_window_size = int((window_size - 1) // 2)

    n_neurons, n_timepoints = fluorescences.shape

    # padding
    pad_left  = np.median(fluorescences[:, :half_window_size], axis=1, keepdims=True) \
                    * np.ones((1, half_window_size))
    pad_right = np.median(fluorescences[:, -half_window_size:], axis=1, keepdims=True) \
                    * np.ones((1, half_window_size))
    f_padded = np.concatenate([pad_left, fluorescences, pad_right], axis=1)

    Dff_all = np.zeros_like(fluorescences)

    for n in tqdm(range(n_neurons)):
        windows = sliding_window_view(f_padded[n], window_size)
        thresholds = np.percentile(windows, percentile, axis=1, keepdims=True)
        masked   = np.where(windows < thresholds, windows, np.nan)
        baseline = np.nanmean(masked, axis=1)  # shape: (n_timepoints,)
        Dff_all[n] = (fluorescences[n] - baseline) / baseline

    Dff_resampled = scipy.interpolate.interp1d(
        recording['ca_times'], Dff_all, kind='nearest'
    )(recording['time_resampled'])

    recording['Dff_resampled'] = Dff_resampled
    return Dff_all, Dff_resampled

def detect_events_with_derivative(recording, test_neuron_dff, excluded_percentile: int = 25,
                                  kernel_sd: float = 0.5):
    cmn_selection = recording[f'cmn_phase_selection'] # timepoints with CMN stimulus
    dff = test_neuron_dff
    sample_rate = recording['sample_rate']

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

    # Final selection
    recording['signal_selection'] = signal_selection

    signal_length = sum(signal_selection)
    recording['signal_length'] = signal_length
    recording['signal_proportion'] = signal_length / sum(cmn_selection)
    recording['signal_dff_mean'] = np.mean(dff[signal_selection])

def calculate_reverse_correlations(recording, bootstrap_num: int = 1000):
    """
        Calculate original and bootstrapped calcium event triggered averages (ETA's).
    """
    print('CMN_pipeline')
    # ROI data
    test_neuron_signal_selection = recording['signal_selection'] # timepoints mask giving events
    # Recording data
    sample_rate = recording['sample_rate']
    radial_bin_edges = recording['radial_bin_edges']
    cmn_phase_selection = recording['cmn_phase_selection'] # timepoints mask giving CMN stim
    motion_vectors_2d = recording['cmn_motion_vectors_2d']

    motion_vectors_2d_filtered = motion_vectors_2d[test_neuron_signal_selection, :, :]
    radial_bin_norms, radial_bin_etas = calculate_local_directions(motion_vectors_2d_filtered, radial_bin_edges)

    recording[f'radial_bin_etas'] = radial_bin_etas # ETA's
    recording[f'radial_bin_norms'] = radial_bin_norms

    min_frame_shift = 4 * sample_rate
    max_frame_shift = int(cmn_phase_selection.sum() - min_frame_shift)
    frame_shifts = np.random.randint(min_frame_shift, max_frame_shift, size=(bootstrap_num))

    radial_bin_bs_etas = np.zeros((bootstrap_num,) + radial_bin_etas.shape)
    signal_within_cmn_selection = test_neuron_signal_selection[cmn_phase_selection]

    for s in tqdm(range(bootstrap_num)):
        # Circularly permutate signal
        perm_signal_selection = np.roll(signal_within_cmn_selection, frame_shifts[s])

        # Get motion vectors of permutated signal
        bs_motion_vectors = motion_vectors_2d[cmn_phase_selection][perm_signal_selection]

        # Calculate vector ETAs for each local radial bin
        radial_bin_bs_etas[s] = calculate_local_directions(bs_motion_vectors, radial_bin_edges)[1]
    recording[f'radial_bin_bs_etas'] = radial_bin_bs_etas


def bootstrap_shm(idx, ang_name, ang_shape, ang_dtype, vel_name, vel_shape, vel_dtype, bins):
    """
        Worker one bootstrap repetition used in calculate_reverse_correlations_shm
        in parallel processing. Uses shared memory for optimizing runtime.
    """
    # access shared memory
    shm_ang = SharedMemory(name=ang_name)
    ang = np.ndarray(ang_shape, dtype=ang_dtype, buffer=shm_ang.buf)[idx]

    shm_vel = SharedMemory(name=vel_name)
    vel = np.ndarray(vel_shape, dtype=vel_dtype, buffer=shm_vel.buf)[idx]

    # calculate calcium-event-triggered averge (ETA)
    return np.mean(vel[:, :, None] * np.logical_and(bins[:-1] <= ang[:, :, None], ang[:, :, None] <= bins[1:]),
                   axis=0)

def calculate_reverse_correlations_shm(recording, bootstrap_num: int = 1024):
    """
        Calculate original and bootstrapped calcium event triggered averages (ETA's).
    """
    print('CMN_pipeline')
    # ROI data
    test_neuron_signal_selection = recording['signal_selection']  # timepoints mask giving events
    # Recording data
    sample_rate = recording['sample_rate']
    radial_bin_edges = recording['radial_bin_edges']
    cmn_phase_selection = recording['cmn_phase_selection']  # timepoints mask giving CMN stim
    motion_vectors_2d = recording['cmn_motion_vectors_2d']

    motion_vectors_2d_filtered = motion_vectors_2d[test_neuron_signal_selection, :, :]
    radial_bin_norms, radial_bin_etas = calculate_local_directions(motion_vectors_2d_filtered, radial_bin_edges)

    recording['radial_bin_etas'] = radial_bin_etas  # ETA's
    recording['radial_bin_norms'] = radial_bin_norms

    min_frame_shift = 4 * sample_rate
    max_frame_shift = int(cmn_phase_selection.sum() - min_frame_shift)
    frame_shifts = np.random.randint(min_frame_shift, max_frame_shift, size=(bootstrap_num))

    radial_bin_bs_etas = np.zeros((bootstrap_num,) + radial_bin_etas.shape)
    signal_within_cmn_selection = test_neuron_signal_selection[cmn_phase_selection]
    signal_indices = signal_within_cmn_selection.nonzero()[0][:, None]
    idcs = np.mod(signal_indices + frame_shifts, signal_within_cmn_selection.size).T
    mv2d = motion_vectors_2d[cmn_phase_selection]

    # np.float32 is fastest type on common processor architectures
    angles = np.arctan2(mv2d[:, :, 0], mv2d[:, :, 1]).astype(np.float32)
    velocities = np.linalg.norm(mv2d, axis=2).astype(np.float32)
    radial_bin_edges = radial_bin_edges.astype(np.float32)

    # precompute all angles and velocities once and write to shared memory for all parallel processes
    # shm=SharedMemory(create=True, size=angles.nbytes+velocities.nbytes+radial_bin_edges.nbytes)
    ang_shm = SharedMemory(create=True, size=angles.nbytes)
    ang_shared = np.ndarray(angles.shape, dtype=angles.dtype, buffer=ang_shm.buf)
    ang_shared[:] = angles

    vel_shm = SharedMemory(create=True, size=velocities.nbytes)
    vel_shared = np.ndarray(velocities.shape, dtype=velocities.dtype, buffer=vel_shm.buf)
    vel_shared[:] = velocities

    start_time = time.time() # time parallel computation

    # adjust maximum number of processes as needed, leave blank to use all kernels
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as exe:
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

    recording[f'radial_bin_bs_etas'] = radial_bin_bs_etas



def calculate_directional_significance(recording, bernoulli_alpha: float = 0.05):
    radial_bin_etas = recording['radial_bin_etas']
    radial_bin_bs_etas = recording['radial_bin_bs_etas']
    bootstrap_num = radial_bin_bs_etas.shape[0]

    # Calculate p-values/significances of original ETAs
    cdf_values = ((radial_bin_etas > radial_bin_bs_etas).sum(axis=0) / bootstrap_num)
    greater_than = cdf_values > 1 - bernoulli_alpha / 2
    less_than = cdf_values < bernoulli_alpha / 2
    radial_bin_p_values = cdf_values.copy()
    radial_bin_p_values[greater_than] = 1 - cdf_values[greater_than]

    radial_bin_significances = np.zeros_like(radial_bin_p_values)
    radial_bin_significances[greater_than] = 1
    radial_bin_significances[less_than] = -1

    recording['radial_bin_p_values'] = radial_bin_p_values
    recording['radial_bin_significances'] = radial_bin_significances

    # Calculate p-value and significance for each bootstrapped norm within bootstrapped distribution
    radial_bin_bs_significance = np.zeros_like(radial_bin_bs_etas, dtype=np.int64)
    radial_bin_bs_p_values = np.zeros(radial_bin_bs_etas.shape)
    for bs_idx in range(bootstrap_num):
        bs_etas = radial_bin_bs_etas[bs_idx]

        cdf_values = (bs_etas > radial_bin_bs_etas).sum(axis=0) / bootstrap_num
        greater_than = cdf_values > 1 - bernoulli_alpha / 2
        less_than = cdf_values < bernoulli_alpha / 2
        p_values = cdf_values.copy()
        p_values[greater_than] = 1 - cdf_values[greater_than]

        significances = np.zeros_like(p_values)
        significances[greater_than] = 1
        significances[less_than] = -1

        # Save results
        radial_bin_bs_p_values[bs_idx] = p_values
        radial_bin_bs_significance[bs_idx] = significances

    recording['radial_bin_bs_p_values'] = radial_bin_bs_p_values
    recording['radial_bin_bs_significances'] = radial_bin_bs_significance


def calculate_directional_preference(recording):
    radial_bin_centers = recording['radial_bin_centers']
    radial_bin_etas = recording['radial_bin_etas']
    radial_bin_significances_greater = recording['radial_bin_significances'] > 0

    vecs = calc_preferred_directions(radial_bin_etas, radial_bin_significances_greater, radial_bin_centers)
    recording['preferred_vectors'] = vecs


def create_clusters(significant_bins: np.ndarray[bool],
                    closest_indices: np.ndarray[int],
                    sign_radial_bin_threshold: int) -> Tuple[np.ndarray, list, list]:
    """
    significant_bins: bool array with shape (patches x radial bins)
    closest_indices: int array with shape (patches x 3) containing indices of the three closest patches for each patch
    sign_radial_bin_threshold: int threshold for how many significant bins adjacent neighbors need to have in common
        in order to be considered "connected"

    returns
        np.ndarray: shape (patches x patches x radial bins) of bool values, marking valid parts of the cluster
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


def trace_cluster(current_patch_idx: int,
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


def find_clusters(recording, sign_radial_bin_threshold: int = 1):
    # Get data
    radial_bin_significances = recording['radial_bin_significances']
    closest_3_position_idcs = recording['closest_3_position_indices']
    # BS data
    radial_bin_bs_significance = recording['radial_bin_bs_significances']
    bootstrap_num = radial_bin_bs_significance.shape[0]

    # Trace clusters in original signal
    _, full_indices, unique_indices = create_clusters(radial_bin_significances > 0,
                                                      closest_3_position_idcs,
                                                      sign_radial_bin_threshold)

    recording['cluster_full_indices'] = full_indices
    recording['cluster_unique_patch_indices'] = unique_indices

    # Trace clusters in bootstrapped signals
    bs_cluster_full_indices = []
    bs_cluster_unique_indices = []
    for bs_idx in range(bootstrap_num):
        _, bs_full_indices, bs_unique_indices = create_clusters(radial_bin_bs_significance[bs_idx] > 0,
                                                                closest_3_position_idcs,
                                                                sign_radial_bin_threshold)
        bs_cluster_full_indices.append(bs_full_indices)
        bs_cluster_unique_indices.append(bs_unique_indices)
    recording['bs_cluster_full_indices'] = bs_cluster_full_indices
    recording['bs_cluster_unique_patch_indices'] = bs_cluster_unique_indices


def calculate_local_directions(motvecs: np.ndarray, bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def calc_preferred_directions(bin_etas: np.ndarray, bin_significances: np.ndarray,
                              bin_centers: np.ndarray) -> np.ndarray:
    """
        Calculates preferred direction of each radial bin as weighted sum of significant directions.
        Weigh each direction by its calcium-event-triggered average.
        influence of ETA absolute values is normed out.

        Parameters:
            bin_etas: shape (patch_num, bin_num)
                calium-event-triggered averages
            bin_significances: shape (patch_num, bin_num)
                1 if bin is significantly excitatory
                -1 if bin is significantly suppressive
                0 if bin is not significant
            bin_centers: shape (bin_num,)
                radial bin centers. centers of the radias bins in which angles are divided in ETA calculation.
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
            vec_pop = np.sum(vecs, axis=0) / np.sum(etas)

        else:
            vec_pop = np.array([0, 0])

        # Append
        population_vectors[idx] = vec_pop

    return population_vectors


def calculate_cluster_significances(recording, cluster_alpha: float = 0.05):
    cluster_full_indices = recording['cluster_full_indices']
    bs_cluster_full_indices = recording['bs_cluster_full_indices']

    radial_bin_significances = (recording[f'radial_bin_significances'] > 0).astype(np.float64)
    radial_bin_bs_significance = (recording[f'radial_bin_bs_significances'] > 0).astype(np.float64)

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

    recording[f'cluster_bs_max_scores'] = bs_max_cluster_scores
    recording[f'cluster_original_scores'] = original_cluster_scores
    recording[f'cluster_significant_indices'] = cluster_significant_indices
