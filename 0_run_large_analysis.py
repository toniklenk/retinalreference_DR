"""
    Script for running analysis on whole dataset.
"""
from os.path import join
import multiprocessing, pickle
from scipy.interpolate import interp1d
from preprocess import *
from rf_estimate import *
from cluster import *
from plot import *

def run_bootstrapping(REC_NAME: str, PLANE: int, IMG_RATE: float):
    LOAD_DFF = False  # if dff has been calculated before

    # EYETRACKING DATA
    EYEPOS_QUADRANTS = (.1, .1, -.1, -.1)


    # NEURON SELECTION
    # run script for selected neurons, set 'all' for whole dataset, otherwise list of ints
    NEURON_SELECTION = 'all'
    # NEURON_SELECTION = range(200, 480+1)
    # ============================================
    # avoid crashes with multiprocessing

    # LOAD DATA
    rec_path = join('data', REC_NAME)
    save_path = join('results', REC_NAME, 'plane' + str(PLANE))
    camera = h5py.File(join(rec_path, 'Camera.hdf5'), 'r')
    fluorescence, rec, phase, _ = digest_folder(rec_path, IMG_RATE, PLANE)

    # PREPROCESSING CMN STIMULI
    # add resampled CMN data to resampled time domain for each phase
    # (info: time domain is resampled from ~2.1Hz to 10Hz in digest_folder())
    process_recording(rec, phase, radial_bin_num=16)

    # PREPROCESSING FLUORESCENCE TRACE
    if LOAD_DFF:
        dff_original = np.load(join(save_path, "dff_original.npy"))
        dff_resampled = (interp1d(
            rec['ca_times'],
            dff_original,
            kind='nearest')(rec['time_resampled']))
    else:
        dff_original, dff_resampled = calc_dff(
            rec,
            fluorescence,
            rec['imaging_rate'])
        Path(save_path).mkdir(parents=True, exist_ok=True)
        np.save(join(save_path, "dff_original.npy"), dff_original)

    # EYE POSITION SELECTION
    q1_mask, q3_mask=calc_eyepos_masks(
        camera['eyepos_ang_le_pos_0'],
        camera['eyepos_ang_re_pos_0'],
        camera['fish_embedded_frame_time'],
        rec['time_resampled'],
        q1_min_left = EYEPOS_QUADRANTS[0],
        q1_min_right = EYEPOS_QUADRANTS[1],
        q3_max_left = EYEPOS_QUADRANTS[2],
        q3_max_right = EYEPOS_QUADRANTS[3],
        verbose=False)

    # NEURON SELECTION
    if NEURON_SELECTION == 'all':
        n_neurons = dff_resampled.shape[0]
        selected_neurons = np.arange(n_neurons)
    else:
        selected_neurons = NEURON_SELECTION

    # RUN FOR SELECTED NEURONS
    for i in tqdm(selected_neurons):
        # calcium events
        signal, _, _, _ = detect_events(rec['cmn_phase_selection'], dff_resampled[i], rec['sample_rate'])

        etas_q1_bs = calc_etas_bs(
            rec['cmn_motion_vectors_2d'][q1_mask],
            signal[q1_mask],
            rec['cmn_phase_selection'][q1_mask],
            rec['sample_rate'],
            rec['radial_bin_edges'],
            bootstrap_num=1024,
            num_workers=22,)
        etas_q3_bs = calc_etas_bs(
            rec['cmn_motion_vectors_2d'][q3_mask],
            signal[q3_mask],
            rec['cmn_phase_selection'][q3_mask],
            rec['sample_rate'],
            rec['radial_bin_edges'],
            bootstrap_num=1024,
            num_workers=22,)

        # WRITE TO DISC
        _path=join(save_path, 'bootstrapped etas',)
        Path(_path).mkdir(parents=True, exist_ok=True)
        np.save(join(_path, f'neuron_{str(i)}_bsRBE_q1.npy'), etas_q1_bs)
        np.save(join(_path, f'neuron_{str(i)}_bsRBE_q3.npy'), etas_q3_bs)

def run_clustering(REC_NAME:str , PLANE: int, IMG_RATE: float):
    LOAD_DFF = False  # if dff has been calculated before

    # EYETRACKING DATA
    EYEPOS_QUADRANTS = (.1, .1, -.1, -.1)

    # PERMUTATION TESTING
    ALPHA = .05  # / (320 * 16)

    # LOAD DATA
    rec_path = join('data', REC_NAME)
    save_path = join('results', REC_NAME, 'plane' + str(PLANE))
    camera = h5py.File(join(rec_path, 'Camera.hdf5'), 'r')
    fluorescence, rec, phase, _ = digest_folder(rec_path, IMG_RATE, PLANE)

    # PREPROCESSING CMN STIMULI
    process_recording(rec, phase, radial_bin_num=16)

    # PREPROCESSING FLUORESCENCE TRACE
    if LOAD_DFF:
        dff_original = np.load(join(save_path, "dff_original.npy"))
        dff_resampled = (interp1d(
            rec['ca_times'],
            dff_original,
            kind='nearest')(rec['time_resampled']))
    else:
        dff_original, dff_resampled = calc_dff(
            rec,
            fluorescence,
            rec['imaging_rate'])
        Path(save_path).mkdir(parents=True, exist_ok=True)
        np.save(join(save_path, "dff_original.npy"), dff_original)

    # EYE POSITION SELECTION
    q1_mask, q3_mask = calc_eyepos_masks(
        camera['eyepos_ang_le_pos_0'],
        camera['eyepos_ang_re_pos_0'],
        camera['fish_embedded_frame_time'],
        rec['time_resampled'],
        q1_min_left=EYEPOS_QUADRANTS[0],
        q1_min_right=EYEPOS_QUADRANTS[1],
        q3_max_left=EYEPOS_QUADRANTS[2],
        q3_max_right=EYEPOS_QUADRANTS[3],
        verbose=False)

    # SAVE RESULTS
    clusters_list_q1, clusters_list_q3 = [], []
    rf_estimates_list_q1, rf_estimates_list_q3 = [], []

    for i in tqdm(np.arange(dff_resampled.shape[0])):
        signal, _, _, _ = detect_events(rec['cmn_phase_selection'], dff_resampled[i], rec['sample_rate'])

        # ETAs
        radial_bin_norms_q1, etas_q1 = calc_etas(
            rec['cmn_motion_vectors_2d'][q1_mask],
            signal[q1_mask],
            rec['cmn_phase_selection'][q1_mask],
            rec['radial_bin_edges'])
        radial_bin_norms_q3, etas_q3 = calc_etas(
            rec['cmn_motion_vectors_2d'][q3_mask],
            signal[q3_mask],
            rec['cmn_phase_selection'][q3_mask],
            rec['radial_bin_edges'])

        # bootstrapped ETAs
        _path = join(save_path, 'bootstrapped RBEs', )
        etas_q1_bs = np.load(join(_path, f'neuron_{str(i)}_bsRBE_q1.npy'))
        etas_q3_bs = np.load(join(_path, f'neuron_{str(i)}_bsRBE_q3.npy'))

        # bin significances
        significances_q1, pvalues_q1 = calc_perm_statistic(etas_q1, etas_q1_bs, bernoulli_alpha=ALPHA)
        significances_q3, pvalues_q3 = calc_perm_statistic(etas_q3, etas_q3_bs, bernoulli_alpha=ALPHA)

        # bin significances bootstrapped ETAs
        significances_bs_q1, pvalues_bs_q1 = calc_perm_statistic_bs(etas_q1_bs)
        significances_bs_q3, pvalues_bs_q3 = calc_perm_statistic_bs(etas_q3_bs)

        # find clusters |                                               bs_cluster_unique_indices_q1
        full_indices_q1, unique_indices_q1, bs_cluster_full_indices_q1, _ = (
            find_clusters(significances_q1, significances_bs_q1, rec['closest_3_position_indices'], ))
        full_indices_q3, unique_indices_q3, bs_cluster_full_indices_q3, _ = (
            find_clusters(significances_q3, significances_bs_q3, rec['closest_3_position_indices'], ))

        # cluster statistics
        # TODO rewrite: take all clusters, output significant clusters
        cluster_significant_indices_q1 = calc_cluster_signif(
            full_indices_q1, bs_cluster_full_indices_q1, significances_q1, significances_bs_q1)
        cluster_significant_indices_q3 = calc_cluster_signif(
            full_indices_q3, bs_cluster_full_indices_q3, significances_q3, significances_bs_q3)

        # RF estimates (clusters not used)
        E1 = estimate_rf(etas_q1, significances_q1 > 0, rec['radial_bin_centers'])
        E3 = estimate_rf(etas_q3, significances_q3 > 0, rec['radial_bin_centers'])

        # SAVE RESULTS
        rf_estimates_list_q1.append(E1)
        rf_estimates_list_q3.append(E3)
        clusters_list_q1.append([unique_indices_q1[c] for c in cluster_significant_indices_q1])
        clusters_list_q3.append([unique_indices_q3[c] for c in cluster_significant_indices_q3])

    cluster_path = join(save_path, 'clusters')
    Path(cluster_path).mkdir(parents=True, exist_ok=True)
    with open(join(cluster_path, 'clusters_q1.pickle'), 'wb') as f:
        pickle.dump(clusters_list_q1, f)
    with open(join(cluster_path, 'clusters_q3.pickle'), 'wb') as f:
        pickle.dump(clusters_list_q3, f)

    np.save(join(cluster_path, 'rf_estimates_q1.npy'), np.stack(rf_estimates_list_q1))
    np.save(join(cluster_path, 'rf_estimates_q3.npy'), np.stack(rf_estimates_list_q3))


def main():
    multiprocessing.set_start_method('spawn')

    REC_NAME = '2026-02-25_mb_fish1_rec2'
    IMG_RATE = 1.9988  # fish1_rec2

    run_bootstrapping(REC_NAME, 0, IMG_RATE)
    run_bootstrapping(REC_NAME, 1, IMG_RATE)
    run_bootstrapping(REC_NAME, 2, IMG_RATE)
    run_bootstrapping(REC_NAME, 3, IMG_RATE)
    run_bootstrapping(REC_NAME, 4, IMG_RATE)

    REC_NAME = '2026-03-04_mb_fish8_rec2'
    IMG_RATE = 1.9957 # fish8_rec2
    run_bootstrapping(REC_NAME, 0, IMG_RATE)
    run_bootstrapping(REC_NAME, 1, IMG_RATE)
    run_bootstrapping(REC_NAME, 2, IMG_RATE)
    run_bootstrapping(REC_NAME, 3, IMG_RATE)
    run_bootstrapping(REC_NAME, 4, IMG_RATE)

    REC_NAME = '2026-02-25_mb_fish1_rec2'
    IMG_RATE = 1.9988  # fish1_rec2
    run_clustering(REC_NAME, 0, IMG_RATE)
    run_clustering(REC_NAME, 1, IMG_RATE)
    run_clustering(REC_NAME, 2, IMG_RATE)
    run_clustering(REC_NAME, 3, IMG_RATE)
    run_clustering(REC_NAME, 4, IMG_RATE)

    REC_NAME = '2026-03-04_mb_fish8_rec2'
    IMG_RATE = 1.9957
    run_clustering(REC_NAME, 0, IMG_RATE)
    run_clustering(REC_NAME, 1, IMG_RATE)
    run_clustering(REC_NAME, 2, IMG_RATE)
    run_clustering(REC_NAME, 3, IMG_RATE)
    run_clustering(REC_NAME, 4, IMG_RATE)
    print('all analysis ran successfully')


if __name__ == '__main__':
    main()