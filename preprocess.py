import os, h5py, scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import date, datetime
from typing import Union, Tuple
from utils import project_to_local_2d_vectors
from numpy.lib.stride_tricks import sliding_window_view
from plot import plot_eyepositions_mask

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

def calc_ca_times(
        mirror_position: np.ndarray,
        mirror_time: np.ndarray,
        imaging_rate: float):
    mirror_position = np.squeeze(mirror_position)
    mirror_time = np.squeeze(mirror_time)

    # Find all the indices where the value is 1 (these are the frame trigger points)
    frame_indices = mirror_time[np.where(mirror_position == 1)[0]]

    first_time = frame_indices[0]
    end_time = frame_indices[-1]

    # Get corresponding times
    frame_times = np.arange(first_time, end_time, 1 / imaging_rate)
    return frame_times\

# currently unused
def calc_ca_times_experimental(
        mirror_position: np.ndarray,
        mirror_time: np.ndarray,
        imaging_rate: float):
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

def digest_folder(
        recording_path: Union[Path, str],
        imaging_rate: float,
        plane: int = 0):
    """
    Reads in data from suite2p's F.npy file and the Display.h5py and Io.h5py file

    Parameters:
        recording_path (Union[Path, str])
        imaging_rate (float):
                    imaging rate of the ca signal
        plane (int):
                    index of recorded plane to be digested

    Returns:
        fluorescence: fluorescence traces (ROIs x Timepoints)

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

    recording = {'animal_id': animal_id, 'rec_date': rec_date, 'rec_id': rec_id}

    s2p_path = os.path.join(recording_path, 'suite2p', 'plane' + str(plane))
    print(s2p_path)

    print('Calculate frame timing of Fluorescence')
    with h5py.File(os.path.join(recording_path, 'Io.hdf5'), 'r') as io_file:
        #mirror_position = np.squeeze(io_file['ai_y_mirror_in'])[:]
        #mirror_time = np.squeeze(io_file['ai_y_mirror_in_time'])[:]

        mirror_position = np.squeeze(io_file['di_frame_sync'])[:]
        mirror_time = np.squeeze(io_file['di_frame_sync_time'])[:]

        # Calculate frame timing
        ca_times = calc_ca_times(mirror_position, mirror_time, imaging_rate=imaging_rate) # example data=2.1798, 1.9989
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

                # I think this is just some corrections for in_phase_idcs[0] or [-1]
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

def process_recording(
        recording,
        phase,
        radial_bin_num: int = 16):
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

def calc_dff(
        recording,
        fluorescences,
        imaging_rate,
        window_size: int = 120,
        percentile: int = 10):
    """
        Calculate_dff.
    """
    print('Calculate dff')
    window_size = int(window_size * imaging_rate)
    if window_size % 2 == 0:
        window_size += 1
    half_window_size = int((window_size - 1) // 2)

    # padding
    pad_left  = np.median(fluorescences[:, :half_window_size], axis=1, keepdims=True) \
                    * np.ones((1, half_window_size))
    pad_right = np.median(fluorescences[:, -half_window_size:], axis=1, keepdims=True) \
                    * np.ones((1, half_window_size))
    f_padded = np.concatenate([pad_left, fluorescences, pad_right], axis=1)

    # vectorized version of nested loop
    windows = sliding_window_view(f_padded, [1, window_size])
    thresholds = np.percentile(windows, percentile, axis=3, keepdims=True)
    windows_masked = np.where(windows < thresholds, windows, np.nan)
    baselines = np.nanmean(windows_masked, axis=3).squeeze()
    Dff_all = fluorescences - baselines

    Dff_resampled = scipy.interpolate.interp1d(
        recording['ca_times'], Dff_all, kind='nearest'
    )(recording['time_resampled'])

    recording['Dff_resampled'] = Dff_resampled
    return Dff_all, Dff_resampled

def detect_events(
        cmn_selection, # timepoints with CMN stimulus
        dff,
        sample_rate,
        excluded_percentile: int = 25,
        kernel_sd: float = 0.5):
    """
        Detect calcium-events in raw fluorescence trace. Done with a different, more simple method
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


# EYETRACKING
def calc_eyepos_masks(
        eyepos_left,
        eyepos_right,
        eyepos_time,
        time_resampled,
        q1_min_left,
        q1_min_right,
        q3_max_left,
        q3_max_right,
        verbose=False):
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

    eye_pos = (eye_pos - eye_pos.mean(axis=0)) / eye_pos.std(axis=0)

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

    # scatterplot to check quadrants
    if verbose:
        plot_eyepositions_mask(eye_pos_resampled, q1_mask, q3_mask)

    return q1_mask, q3_mask