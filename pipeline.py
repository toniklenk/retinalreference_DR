from main_functions import *
from plotting_functions import *


def main():
    #better to use \ as raw backslash
    # recording_path = r"example_data\fish1_20250815\2025-08-15_recording1" # example data
    recording_path = r"data\2026-02-25_mb_fish1_rec2"
    save_path = r"results\2026-02-25_mb_fish1_rec2_run2"

    fluorescence, recording, phase, ca_rec_group_id_fun = digest_folder(recording_path, plane=0)

    # modifies recording object directly
    # add resampled CMN data to resampled time domain for each phase
    # (info: time domain is resampled from ~2.1Hz to 10Hz in digest_folder())
    process_recording(recording, phase, radial_bin_num=16)

    # fast=True
    # if fast==True:
    #     Dff_all_neuron, Dff_resampled = calculate_dff_vectorized(recording, fluorescence, recording['imaging_rate'])
    #     np.save(os.path.join(save_path, "deconvolved_Dff_original_fast.npy"), Dff_all_neuron)
    # else:
    #     Dff_all_neuron, Dff_resampled = calculate_dff(recording, fluorescence, recording['imaging_rate'])
    #     np.save(os.path.join(save_path, "deconvolved_Dff_original.npy"), Dff_all_neuron)


    Dff_all_neuron = np.load(os.path.join(save_path, "deconvolved_Dff_original.npy"))
    Dff_resampled = scipy.interpolate.interp1d(recording['ca_times'], Dff_all_neuron, kind='nearest')(
            recording['time_resampled'])

    # iterated individual neurons, draws receptive field of each neuron to .png and .pdf
    for k, i in tqdm(enumerate(range(Dff_all_neuron.shape[0]))):
        test_neuron_dff = Dff_resampled[i, :]

        # einfacher als mit GMMs (wie's in 2019 paper is), robust
        detect_events_with_derivative(recording, test_neuron_dff)

        # ETA berechnung, Abb mit cnts für bewegungsrichtugen im vf
        calculate_reverse_correlations_shm(recording, bootstrap_num=1024)

        # 1. Teil Bootstrap test, bootstrap verteilung berechnungen, welche revcorr significant
        calculate_directional_significance(recording)

        # added by me: 2. Teil
        calculate_directional_preference(recording)

        # clusteranalyse
        find_clusters(recording)

        # cluster based statistics, (2-step NP bootsrap test)
        calculate_cluster_significances(recording)

        plot_rf_overview(recording, i, save_path)
        print(1)

if __name__ == '__main__':
    main()
