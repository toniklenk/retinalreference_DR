import pickle
from main_functions import *
from main_functions_generalAPI import *


def main():
    # recording_name = '2026-03-04_mb_fish8_rec2'
    recording_name = '2026-03-04_mb_fish8_rec2'
    plane = 0
    imaging_rate = 1.9988 # fish1_rec2
    params=(.1,.1,-.1,-.1) # fish1_rec2



    recording_path = os.path.join('data', recording_name)
    save_path = os.path.join('results', recording_name)

    # set flags for saving intermediate results
    compute_dff=False # False skip recalculating dff

    # load eyetracking data
    camera = h5py.File(os.path.join(recording_path, 'Camera.hdf5'), 'r')
    fluorescence, rec, phase, ca_rec_group_id_fun = digest_folder(recording_path, imaging_rate=1.9957, plane=0)
    # add resampled CMN data to resampled time domain for each phase
    # (info: time domain is resampled from ~2.1Hz to 10Hz in digest_folder())
    process_recording(rec, phase, radial_bin_num=16)
    if  compute_dff:
        dff_original, dff_resampled = calculate_dff_vectorized(rec, fluorescence, rec['imaging_rate'])
        np.save(os.path.join(save_path, "dff_original.npy"), dff_original)
    else:
        dff_original = np.load(os.path.join(save_path, "dff_original.npy"))
        dff_resampled = scipy.interpolate.interp1d(rec['ca_times'], dff_original, kind='nearest')(
                rec['time_resampled'])

    # Eye tracking
    q1_mask, q3_mask=generate_eyepos_masks(
        camera['eyepos_ang_le_pos_0'],
        camera['eyepos_ang_re_pos_0'],
        camera['fish_embedded_frame_time'],
        rec['time_resampled'],
        q1_min_left = params[0],
        q1_min_right = params[1],
        q3_max_left = params[2],
        q3_max_right = params[3])

    # iterated individual neurons, draws receptive field of each neuron to .png and .pdf
    for i, dff_i in tqdm(enumerate(dff_resampled)):
        # determine calcium events
        # do this before subselecting data with eye positions to avoid jumps in the first derivative
        rec['signal_selection'], rec['signal_length'], rec['signal_proportion'], rec['signal_dff_mean']\
            =detect_events_with_derivative_generalAPI(
            rec['cmn_phase_selection'],
            dff_i,
            rec['sample_rate'])

        # STEP 1 of 2-step bootstrap test
        # Calcium-event-triggered-averages calculation
        # calculate bootstrapping distribution of ETAs in each spatial position and direction bin
        #calculate_reverse_correlations_shm(rec, bootstrap_num=1024)
        # radial bin etas
        q1_rbe_bootstrapped = calculate_radial_bin_bs_etas(
            rec['cmn_motion_vectors_2d'][q1_mask],
            rec['signal_selection'][q1_mask],
            rec['cmn_phase_selection'][q1_mask],
            rec['sample_rate'],
            rec['radial_bin_edges'],
            bootstrap_num=1024,
            num_workers=22,)

        # radial bin etas
        q3_rbe_bootstrapped = calculate_radial_bin_bs_etas(
            rec['cmn_motion_vectors_2d'][q3_mask],
            rec['signal_selection'][q3_mask],
            rec['cmn_phase_selection'][q3_mask],
            rec['sample_rate'],
            rec['radial_bin_edges'],
            bootstrap_num=1024,
            num_workers=22,)

        _path=os.path.join(save_path, 'bootstrapped RBEs',)
        Path(_path).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(_path, f'neuron_{str(i)}_bsRBE_q1.npy'), q1_rbe_bootstrapped)
        np.save(os.path.join(_path, f'neuron_{str(i)}_bsRBE_q3.npy'), q3_rbe_bootstrapped)

        #break

if __name__ == '__main__':
    main()
