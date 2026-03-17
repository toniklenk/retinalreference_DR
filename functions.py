"""
    Functions for processing of 2-photon calcium imaging data.
    Based on previous analysis code but rewritten for improved code readability.
"""
import scipy
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def calc_dff(
        recording,
        fluorescences,
        imaging_rate,
        window_size: int = 120,
        percentile: int = 10):
    """
        Vectorized version of calculate_dff.
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


def project_to_local_2d_vectors(
        normals: np.ndarray,
        vectors: np.ndarray
) -> np.ndarray:
    """
        Parameters:
                normals: np.array
                    centers of CMN stimuli

                vectors: np.array
                    3D CMN motion vectors
    """

    def crossproduct(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Workaround because of NoReturn situation in numpy.cross"""
        return np.cross(v1, v2)

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


def FE_similarity(F, E):
    """
        Calculate similarity between F and E according to the definition in Zhang et. at. 2022
        Parameters:
            E: Estimated receptive field
            F: Optic flow field fitted to E
    """
    return ((np.sum(F*E, axis=1)/
            (np.linalg.norm(np.clip(E, 0.0000001, 1.), axis=1) *
             np.linalg.norm(F, axis=1)))
            .mean())

def tof(
        angle_azimuth: float,
        angle_elevation: float,
        P: np.ndarray):
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

def rof(
        angle_azimuth: float,
        angle_elevation: float,
        P: np.ndarray):
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
    coefficients = np.sum(F*E, axis=1)/ (np.linalg.norm(np.clip(E, 0.0000001, 1.), axis=1)* np.linalg.norm(F, axis=1))
    angles = np.arccos(np.clip(coefficients, -1.0, 1.0))
    return np.sum(angles**2)

def RSSangle_Fto2D(F, E, pos):
    """
        Wraps around def RSSangle(F, E) to transform F from 3D to 2D,
        for convenience in using it with scipy.optimize.minimize
    """
    F=project_to_local_2d_vectors(pos, F[None,:,:]).squeeze()
    return RSSangle(F, E)
