from archive.plotting import *
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path

def plot_v1(E1, E3, F1, F3, positions, save_path_=None, neuron_num=None, alpha_E=.5, alpha_F=.5):
    # Plot estimated RF and fitted TOF/ROF together
    # gs_cmn = plt.GridSpec(20, 10)
    fig, (ax1, ax3)=plt.subplots(ncols=1, nrows=2, figsize=(20, 10))
    # Plot preferred local motion vectors quiver plot
    # Get position and local motion preference data
    x, y, _ = cart2sph(*positions.T)

    E1vels=np.linalg.norm(E1, axis=1)
    color = matplotlib.colormaps['tab20'](0)
    ax1.quiver(
        x, y,
        E1[:, 0], E1[:, 1],
        pivot='mid',color=color,width=0.002,scale=E1vels.max() * 30,alpha=alpha_E)

    F1vels=np.linalg.norm(F1, axis=1)
    color = matplotlib.colormaps['tab20'](2)
    ax1.quiver(
        x, y,
        F1[:, 0], F1[:, 1],
        pivot='mid',color=color,width=0.002,scale=F1vels.max() * 30,alpha=alpha_F)

    E3vels=np.linalg.norm(E3, axis=1)
    color = matplotlib.colormaps['tab20'](0)
    ax3.quiver(
        x, y,
        E3[:, 0], E3[:, 1],
        pivot='mid',color=color,width=0.002,scale=E3vels.max() * 30,alpha=alpha_E)

    F3vels=np.linalg.norm(F3, axis=1)
    color = matplotlib.colormaps['tab20'](2)
    ax3.quiver(
        x, y,
        F3[:, 0], F3[:, 1],
        pivot='mid',color=color,width=0.002,scale=F3vels.max() * 30,alpha=alpha_F)

    ax1.set_xlabel('azimuth [deg]')
    ax3.set_xlabel('azimuth [deg]')
    ax1.set_ylabel('elevation [deg]')
    ax3.set_ylabel('elevation [deg]')
    ax1.set_aspect('equal')
    ax3.set_aspect('equal')

    if save_path_ is not None and neuron_num is not None:
        png_path = os.path.join(save_path_, 'png')
        Path(png_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(png_path, f'both2D_{neuron_num}.png'), dpi=300)
    plt.close()

def plot_rf_overview_generalAPI(
        radial_bin_etas,
        radial_bin_edges,
        radial_bin_significance,
        positions,
        patch_corners,
        patch_indices,
        cluster_significant_indices,
        estimatedRF,
        cluster_unique_patch_indices,
        neuron_num, save_path: str = None, q=0):
    """

    """
    print(f'plot {neuron_num} neuron receptive field')
    # Recording data

    # Plot DFF
    gs_cmn = plt.GridSpec(20, 10)
    fig_cmn = plt.figure(figsize=(20, 10))

    # Plot radial histogram figure
    ax_hist = fig_cmn.add_subplot(gs_cmn[:9, :])
    patch_collections = plot_radial_histograms(ax_hist, positions, np.abs(radial_bin_etas), radial_bin_edges)
    plot_radial_significance(radial_bin_significance, patch_collections)
    plot_patch_grid(ax_hist, patch_corners, patch_indices)

    # Plot preferred local motion vectors quiver plot
    ax_quiv = fig_cmn.add_subplot(gs_cmn[11:, :], sharex=ax_hist, sharey=ax_hist)

    # If Roi has a receptive field, plot best egomotion fit
    if len(cluster_significant_indices) > 0:
        # Get position and local motion preference data
        x, y, _ = cart2sph(*positions.T)
        preferred_vectors = estimatedRF
        preferred_velocities = np.linalg.norm(estimatedRF, axis=1)
        cluster_significant_indices = cluster_significant_indices
        selected_clusters = [cluster_unique_patch_indices[_idx] for _idx in cluster_significant_indices]

        for s_c, idcs in enumerate(selected_clusters):
            idcs = np.array(idcs)
            color = matplotlib.colormaps['tab20'](s_c)
            ax_quiv.quiver(x[idcs], y[idcs], preferred_vectors[idcs, 0], preferred_vectors[idcs, 1],
                           pivot='mid', color=color, width=0.002, scale=preferred_velocities[idcs].max() * 30)

    # Add grid
    plot_patch_grid(ax_quiv, patch_corners, patch_indices)

    # Format
    ax_quiv.set_xlabel('azimuth [deg]')
    ax_quiv.set_ylabel('elevation [deg]')
    ax_quiv.set_aspect('equal')

    if save_path is not None:
        _path = os.path.join(save_path, '2Dsigni')
        Path(_path).mkdir(parents=True, exist_ok=True)
        fig_cmn.savefig(os.path.join(_path, f'{neuron_num}_{q}.png'), dpi=300)
    plt.close(fig_cmn)

def plot_eyepositions(
        eyepos,
        q1_min_left,
        q1_min_right,
        q1_width,
        q1_height,
        q3_max_left,
        q3_max_right,
        q3_widht,
        q3_height):
    """
        Scatterplot of eye positions to visualize quadrants for subselecting data.
    """
    fig, ax = plt.subplots(figsize=(20,20))

    q1 = np.logical_and(
        eyepos[:, 0] > q1_min_left,
        eyepos[:, 1] > q1_min_right)
    q3 = np.logical_and(
        eyepos[:, 0] < q3_max_left,
        eyepos[:, 1] < q3_max_right)
    out = np.logical_not(q1 | q3)

    ax.scatter(eyepos[q1,0], eyepos[q1,1], s=1., alpha=0.6, color='red')
    ax.scatter(eyepos[q3,0], eyepos[q3,1], s=1., alpha=0.6, color='blue')
    ax.scatter(eyepos[out,0], eyepos[out,1], s=1., alpha=0.6, color='grey')
    ax.set_xlabel('right eye position (no unit, normalized)', fontsize=20, color='black')
    ax.set_ylabel('left eye position (no unit, normalized)', fontsize=20, color='black')
    ax.tick_params(axis='both', which='major', labelsize=16, color='black')

    # 1st quadrant (defined by lower boundaries)
    plt.hlines((q1_min_right, q1_min_right+q1_height),q1_min_left, q1_min_left+q1_width)
    plt.vlines((q1_min_left, q1_min_left+q1_width),  q1_min_right, q1_min_right+q1_height)

    # 3rd quadrant (define by upper boundaries)
    plt.hlines((q3_max_right, q3_max_right-q3_height), q3_max_left-q3_widht, q3_max_left)
    plt.vlines((q3_max_left, q3_max_left-q3_widht), q3_max_right-q3_height, q3_max_right)

    return q1, q3, out