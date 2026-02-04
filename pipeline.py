from main_functions import *
from plotting_functions import *

#better to use \ as raw backslash
recording_path = r"example_data\fish1_20250815\2025-08-15_recording1"
save_path = ''

fluorescence, recording, phase, ca_rec_group_id_fun = digest_folder(recording_path)

# modifies recording object directly
# add resampled CMN data to resampled time domain for each phase
# (info: time domain is resampled from ~2.1Hz to 10Hz in digest_folder())
process_recording(recording, phase, radial_bin_num=16)

#Dff_all_neuron, Dff_resampled = calculate_dff(recording, fluorescence, recording['imaging_rate'])
Dff_all_neuron = np.load(os.path.join(save_path, "deconvolved_Dff_original.npy"))
Dff_resampled = scipy.interpolate.interp1d(recording['ca_times'], Dff_all_neuron, kind='nearest')(
        recording['time_resampled'])


for k, i in tqdm(enumerate(range(Dff_all_neuron.shape[0]))):
    test_neuron_dff = Dff_resampled[i, :]

    # einfacher als mit GMMs (wie's in 2019 paper is), robust
    detect_events_with_derivative(recording, test_neuron_dff)

    # ETA berechnung, Abb mit cnts für bewegungsrichtugen im vf
    calculate_reverse_correlations(recording)

    # 1. Teil Bootstrap test, bootstrap verteilung berechnungen, welche revcorr significant
    calculate_directional_significance(recording)

    # clusteranalyse
    find_clusters(recording)

    # cluster based statistics, (2-step NP bootsrap test)
    calculate_cluster_significances(recording)

    plot_rf_overview(recording, i, save_path)
    print(1)
