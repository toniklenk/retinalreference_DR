"""
    Base analysis consists of the following steps:
        1. Preprocessing of recording data and fluorescence traces
        2. Definition of eye position quadrants
        3. RF estimation
        4. Cluster analysis in RFs
"""

from os.path import join
import multiprocessing, pickle
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from preprocess import *
from rf_estimate import *
from cluster import *
from utils import *
from plot import *
def main():
    # manually defined Parameters+++==============
    # FISH AND RECORDING
    REC_NAME = '2026-02-25_mb_fish1_rec2'
    # REC_NAME = '2026-03-04_mb_fish8_rec2'

    # CALCIUM DATA
    PLANE = 0
    IMG_RATE = 1.9988  # fish1_rec2
    # IMG_RATE = 1.9957 # fish8_rec2
    LOAD_DFF = True  # if dff has been calculated before

    # EYETRACKING DATA
    EYEPOS_QUADRANTS = (.1, .1, -.1, -.1)  # fish1_rec2
    # EYEPOS_QUADRANTS = (.1, .1, -.1, -.1)  # fish8_rec2

    # PERMUTATION TESTING
    ALPHA = .05 # / (320 * 16)
    # if permutation ETAs have been computed before
    LOAD_PERMUTATION_RESULTS = False
    # run permutation test only and save results
    RUN_ONLY_PERMUTATION_TEST = False

    # NEURON SELECTION
    # run script for selected neurons, set 'all' for whole dataset, otherwise list of ints
    NEURON_SELECTION = 'all'
    # NEURON_SELECTION = [6, 7, 8]
    # ============================================
    # avoid crashes with multiprocessing
    multiprocessing.set_start_method('spawn')

    rec_path = join('data', REC_NAME)
    save_path = join('results', REC_NAME, 'plane' + str(PLANE))
    camera = h5py.File(join(rec_path, 'Camera.hdf5'), 'r')
    fluorescence, rec, phase, _ = digest_folder(rec_path, IMG_RATE, PLANE)

    # add resampled CMN data to resampled time domain for each phase
    # (info: time domain is resampled from ~2.1Hz to 10Hz in digest_folder())
    process_recording(rec, phase, radial_bin_num=16)

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


    q1_mask, q3_mask=calc_eyepos_masks(
        camera['eyepos_ang_le_pos_0'],
        camera['eyepos_ang_re_pos_0'],
        camera['fish_embedded_frame_time'],
        rec['time_resampled'],
        q1_min_left = EYEPOS_QUADRANTS[0],
        q1_min_right = EYEPOS_QUADRANTS[1],
        q3_max_left = EYEPOS_QUADRANTS[2],
        q3_max_right = EYEPOS_QUADRANTS[3],
        verbose=True)

    # save intermediate results
    E_list_q1, E_list_q3 = [], []
    F_list_q1, F_list_q3 = [], []
    FoC_list_q1, FoC_list_q3 = [], []
    FE1_sim_list, FE3_sim_list = [], []
    clusters_list_q1, clusters_list_q3 = [], []
    rf_estimates_list_q1, rf_estimates_list_q3 = [], []

    # default init to run for all neurons in dataset
    if NEURON_SELECTION == 'all':
        n_neurons = dff_resampled.shape[0]
        selected_neurons = np.arange(n_neurons)
    else:
        selected_neurons = NEURON_SELECTION


    for i in tqdm(selected_neurons):
        # calcium events
        rec['signal_selection'], rec['signal_length'], rec['signal_proportion'], rec['signal_dff_mean'] \
            = detect_events(rec['cmn_phase_selection'], dff_resampled[i], rec['sample_rate'])

        # bootstrap ETAs
        if LOAD_PERMUTATION_RESULTS:
            # load
            _path = join(save_path, 'bootstrapped RBEs', )
            etas_q1_bs = np.load(join(_path, f'neuron_{str(i)}_bsRBE_q1.npy'))
            etas_q3_bs = np.load(join(_path, f'neuron_{str(i)}_bsRBE_q3.npy'))
        else:
            # calculate
            etas_q1_bs = calc_etas_bs(
                rec['cmn_motion_vectors_2d'][q1_mask],
                rec['signal_selection'][q1_mask],
                rec['cmn_phase_selection'][q1_mask],
                rec['sample_rate'],
                rec['radial_bin_edges'],
                bootstrap_num=1024,
                num_workers=24,)
            etas_q3_bs = calc_etas_bs(
                rec['cmn_motion_vectors_2d'][q3_mask],
                rec['signal_selection'][q3_mask],
                rec['cmn_phase_selection'][q3_mask],
                rec['sample_rate'],
                rec['radial_bin_edges'],
                bootstrap_num=1024,
                num_workers=24,)
            # save bootstrapped ETAs
            _path=join(save_path, 'bootstrapped RBEs',)
            Path(_path).mkdir(parents=True, exist_ok=True)
            np.save(join(_path, f'neuron_{str(i)}_bsRBE_q1.npy'), etas_q1_bs)
            np.save(join(_path, f'neuron_{str(i)}_bsRBE_q3.npy'), etas_q3_bs)
        if RUN_ONLY_PERMUTATION_TEST:
            continue

        # ETAs for CMN and eyepos quadrants
        radial_bin_norms_q1, etas_q1 = calc_etas(
            rec['cmn_motion_vectors_2d'][rec['signal_selection'] & q1_mask, :, :],
            rec['radial_bin_edges'])
        radial_bin_norms_q3, etas_q3 = calc_etas(
            rec['cmn_motion_vectors_2d'][rec['signal_selection'] & q3_mask, :, :],
            rec['radial_bin_edges'])

        # bin significances
        significances_q1, pvalues_q1 = calc_perm_statistic(etas_q1, etas_q1_bs, bernoulli_alpha=ALPHA)
        significances_q3, pvalues_q3 = calc_perm_statistic(etas_q3, etas_q3_bs, bernoulli_alpha=ALPHA)
        # bin significances for bootstrapped ETAs
        significances_bs_q1, pvalues_bs_q1 = calc_perm_statistic_bs(etas_q1_bs)
        significances_bs_q3, pvalues_bs_q3 = calc_perm_statistic_bs(etas_q3_bs)

        # find clusters |                                               bs_cluster_unique_indices_q1
        full_indices_q1, unique_indices_q1, bs_cluster_full_indices_q1, _  = (
            find_clusters(significances_q1, significances_bs_q1, rec['closest_3_position_indices'], ))

        full_indices_q3, unique_indices_q3, bs_cluster_full_indices_q3, _ = (
            find_clusters(significances_q3, significances_bs_q3, rec['closest_3_position_indices'], ))

        # cluster statistics
        # TODO rewrite: take all clusters, output significant clusters
        cluster_significant_indices_q1 = calc_cluster_signif(
            full_indices_q1, bs_cluster_full_indices_q1, significances_q1, significances_bs_q1)
        cluster_significant_indices_q3 = calc_cluster_signif(
            full_indices_q3, bs_cluster_full_indices_q3, significances_q3, significances_bs_q3)

        clusters_list_q1.append([unique_indices_q1[c] for c in cluster_significant_indices_q1])
        clusters_list_q3.append([unique_indices_q3[c] for c in cluster_significant_indices_q3])
        # RF estimates (clusters not used)
        E1 = estimate_rf(etas_q1, significances_q1 > 0, rec['radial_bin_centers'])
        E3 = estimate_rf(etas_q3, significances_q3 > 0, rec['radial_bin_centers'])
        rf_estimates_list_q1.append(E1)
        rf_estimates_list_q3.append(E3)

        # continue
        # determine mode center for neurons with unimodal rf in both eye positions
        # determine_mode_center(unimodal_RF_significant, positions) \
        #                                             -> np.array((ModeCenter_azimuth, ModeCenter_elevation))
        if len(cluster_significant_indices_q1 == 1 and len(cluster_significant_indices_q3 == 1)):
            pass

        # determine FoC for neurons with bimodal rf in both eye positions
        if len(cluster_significant_indices_q1 == 2 and len(cluster_significant_indices_q3 == 2)):
            pass

        # calc_FoC(bimodal_RF_significant, positions) -> np.array((FoC_azimuth, FoC_elevation))
        # skip rest

        # select only parts of RF in significant clusters for estimation
        E1_si=np.zeros_like(E1)
        # TODO continue editing here

        # fit E
        mini_F1=minimize(
            lambda angles: RSSangle_Fto2D(tof(*angles, rec['positions']), E1, rec['positions']),
            [0., 0.],)
        mini_F3=minimize(
            lambda angles: RSSangle_Fto2D(tof(*angles, rec['positions']), E3, rec['positions']),
            [0., 0.],)
        # F from fit
        F1_3D=tof(*mini_F1.x, rec['positions'])[None,:,:]
        F1=project_to_local_2d_vectors(rec['positions'], F1_3D).squeeze()
        F3_3D=tof(*mini_F3.x, rec['positions'])[None,:,:]
        F3=project_to_local_2d_vectors(rec['positions'], F3_3D).squeeze()
        E_list_q1.append(E1); F_list_q1.append(F1)
        FoC_list_q1.append(mini_F1.x); FoC_list_q3.append(mini_F3.x)
        E_list_q3.append(E3); F_list_q3.append(F3)
        FE1_sim_list.append(FE_similarity(F1, E1))
        FE3_sim_list.append(FE_similarity(F3, E3))

        # # PLOTTING
        # move this to a notebook fully
        # plot_rf_overview_generalAPI(
        #     etas_q1,
        #     rec['radial_bin_edges'],
        #     significances_q1,
        #     rec['positions'],
        #     rec['patch_corners'],
        #     rec['patch_indices'],
        #     cluster_significant_indices_q1,
        #     E1,
        #     unique_indices_q1,
        #     i, save_path=save_path, q=1)
        # plot_rf_overview_generalAPI(
        #     etas_q3,
        #     rec['radial_bin_edges'],
        #     significances_q3,
        #     rec['positions'],
        #     rec['patch_corners'],
        #     rec['patch_indices'],
        #     cluster_significant_indices_q3,
        #     E3,
        #     unique_indices_q3,
        #     i, save_path=save_path, q=3)
        #
        # plot_v1(
        #     E1, E3,
        #     F1, F3,
        #     rec['positions'],
        #     alpha_E=.9, alpha_F=.55,
        #     save_path_=save_path,
        #     neuron_num=i)

    # SAVE RESULTS
    cluster_path = join(save_path, 'clusters')
    Path(cluster_path).mkdir(parents=True, exist_ok=True)
    with open(join(cluster_path, 'clusters_q1.pickle'), 'wb') as f:
        pickle.dump(clusters_list_q1, f)
    with open(join(cluster_path, 'clusters_q3.pickle'), 'wb') as f:
        pickle.dump(clusters_list_q3, f)
    np.save(join(cluster_path, 'rf_estimates_q1.npy'), np.concatenate(rf_estimates_list_q1))
    np.save(join(cluster_path, 'rf_estimates_q3.npy'), np.concatenate(rf_estimates_list_q3))

    # FoC_array_q1=np.concatenate([FoC[:,None].T for FoC in FoC_list_q1])
    # FoC_array_q3=np.concatenate([FoC[:,None].T for FoC in FoC_list_q3])
    # FoC_path = join(save_path, 'FoCs')
    # Path(FoC_path).mkdir(parents=True, exist_ok=True)
    # np.save(join(FoC_path, f'FoCq1'), FoC_array_q1)
    # np.save(join(FoC_path, f'FoCq3'), FoC_array_q3)

    print('pipeline ran successfully')

if __name__ == '__main__':
    main()
