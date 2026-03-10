import pickle
from main_functions import *
from main_functions_generalAPI import calculate_local_directions_generalAPI, generate_eyepos_masks, \
    calc_preferred_directions_generalAPI
from plotting_functions import *



def main():
    recording_path = os.path.join('data','2026-02-25_mb_fish1_rec2')
    save_path = os.path.join('results','2026-02-25_mb_fish1_rec2_run2')

    # set flags for saving intermediate results
    save_results=False
    save_etas=True
    compute_dff=False # False skip recalculating dff

    # load eyetracking data
    camera = h5py.File(os.path.join(recording_path, 'Camera.hdf5'), 'r')
    fluorescence, rec, phase, ca_rec_group_id_fun = digest_folder(recording_path, plane=0)

    # modifies recording object directly
    # add resampled CMN data to resampled time domain for each phase
    # (info: time domain is resampled from ~2.1Hz to 10Hz in digest_folder())
    process_recording(rec, phase, radial_bin_num=16)
    if  compute_dff:
        dff_original, dff_resampled = calculate_dff_vectorized(rec, fluorescence, rec['imaging_rate'])
        np.save(os.path.join(save_path, "deconvolved_Dff_original_fast.npy"), dff_original)
    else:
        dff_original = np.load(os.path.join(save_path, "deconvolved_Dff_original.npy"))
        dff_resampled = scipy.interpolate.interp1d(rec['ca_times'], dff_original, kind='nearest')(
                rec['time_resampled'])

    # Eye tracking
    params=(10,-5, 6, -12) # parameters_fish1_rec2
    q1_mask, q3_mask=generate_eyepos_masks(
        camera['eyepos_ang_le_pos_0'],
        camera['eyepos_ang_re_pos_0'],
        camera['fish_embedded_frame_time'],
        rec['time_resampled'],
        q1_min_left = params[0],
        q1_min_right = params[1],
        q3_max_left = params[2],
        q3_max_right = params[3])

    n_neurons=dff_resampled.shape[0]
    # iterated individual neurons, draws receptive field of each neuron to .png and .pdf
    for i in tqdm(range(n_neurons)):
        # determine calcium events
        # do this before subselecting data with eye positions to avoid jumps in the first derivative
        rec['signal_selection'], rec['signal_length'], rec['signal_proportion'], rec['signal_dff_mean']\
            =detect_events_with_derivative(
            rec['signal_selection'],
            dff_resampled[i, :],
            rec['sample_rate'])

        # calculate calcium event triggered averages (of CMN motion vectors), without bootstrapping
        # 1. based on data from all eye positions
        radial_bin_norms, radial_bin_etas = calculate_local_directions_generalAPI(
            rec['cmn_motion_vectors_2d'][rec['signal_selection'], :, :],
            rec['radial_bin_edges'])
        #rec['radial_bin_norms'], rec['radial_bin_etas']=radial_bin_norms, radial_bin_etas

        # 2. based on data from one side/quadrant of eye positions
        radial_bin_norms_q1, radial_bin_etas_q1 = calculate_local_directions_generalAPI(
            rec['cmn_motion_vectors_2d'][rec['signal_selection'] & q1_mask, :, :],
            rec['radial_bin_edges'])

        # 3. based on data from other side/quadrant of eye positios
        radial_bin_norms_q3, radial_bin_etas_q3 = calculate_local_directions_generalAPI(
            rec['cmn_motion_vectors_2d'][rec['signal_selection'] & q3_mask, :, :],
            rec['radial_bin_edges'])




        # STEP 1 of 2-step bootstrap test
        # Calcium-event-triggered-averages calculation
        # calculate bootstrapping distribution of ETAs in each spatial position and direction bin
        calculate_reverse_correlations_shm(rec, bootstrap_num=1024)
        # calculate p-values for both true and bootstrapped ETAs
        calculate_directional_significance(rec)


        # STEP 2 of 2-step bootstrap test
        # find clusters in original and bootstrapped etas p-values (these represent the first order statistic)
        find_clusters(rec)
        # calculate the second order statistics for the original/true cluster sizes
        # from the distribution of the cluster sizes in the bootstrapped values
        calculate_cluster_significances(rec)

        if save_results:
            path=os.path.join(recording_path, 'recording_dicts',)
            Path(path).mkdir(parents=True, exist_ok=True) # create dir if needed
            with open(os.path.join(path, f'recording_neuron{str(i)}.pkl'),'wb') as f:
                pickle.dump(rec, f)

        # estimate neurons RFs (RF: 'preferred_vectors')
        # weighted sum of ETAs of SIGNIFICANT radial bins in each spatial path of the visual field
        # this is needed for plotting to work
        # calculate_directional_preference(rec)
        RF_est=calc_preferred_directions_generalAPI(
            radial_bin_etas,
            rec['radial_bin_significances'] > 0,
            rec['radial_bin_centers'])

        RF_est=calc_preferred_directions_generalAPI(
            radial_bin_etas,
            rec['radial_bin_significances'] > 0,
            rec['radial_bin_centers'])

        RF_est=calc_preferred_directions_generalAPI(
            radial_bin_etas,
            rec['radial_bin_significances'] > 0,
            rec['radial_bin_centers'])


        rec['preferred_vectors']=RF_est

        # fit optic flow fields
        # to estimated RF from all data


        # to estimated RF from first quadrant data

        # to estimated RF from third quadrant data

        # later: only to motion vectors from significant patches.

        # calculate difference between Focus of Contraction

        # statistical testing of difference between Focus of Contraction: 1-step nonparametric permutation test
        # permutation distribution: random shifting of eye positions in relation to motion vectors.
        # calculate distance

        # create directories if needed
        Path(os.path.join(save_path, 'pdf')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(save_path, 'png')).mkdir(parents=True, exist_ok=True)
        plot_rf_overview(rec, i, save_path)


        print(1)

if __name__ == '__main__':
    main()
