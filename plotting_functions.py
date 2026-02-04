import numpy as np
import os
import quaternionic
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from typing import List
from vispy.util import transforms


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    elevation = np.arctan2(z, hxy)
    azimuth = -np.arctan2(y, x)
    return np.array([azimuth, elevation, r])


def _add_grid_patch_coordinates(_vertices: np.ndarray, _patches, duplicated):
    p = []
    center = _vertices.mean(axis=0)
    caz, cel, _ = cart2sph(*center)
    for j in range(3):
        v1 = _vertices[j]
        v2 = _vertices[j + 1]

        q1 = quaternionic.array([0, *v1])
        q2 = quaternionic.array([0, *v2])

        for k in np.linspace(0, 1, 20):
            vinterp = quaternionic.slerp(q1, q2, k).ndarray[1:]
            az, el, _ = cart2sph(*vinterp)

            # If there is a sign flip at the back around +/-180deg,
            #  duplicate patch once for an ever so slightly rotated version
            #  and correct azimuth on this patch
            if (np.sign(az) != np.sign(caz)) & (np.abs(caz) > np.pi / 2):
                if not duplicated:
                    M = transforms.rotate(np.sign(caz) / 10 ** 6, (0, 0, 1))[:3, :3]
                    _add_grid_patch_coordinates(np.array([np.dot(v, M) for v in _vertices]), _patches, True)
                    duplicated = True
                az = np.nan

            p.append([az, el])

    _patches.append(p)


def plot_radial_histograms(ax: plt.Axes,
                           positions: np.ndarray, lengths: np.ndarray, bin_edges: np.ndarray,
                           scale: float = 1.0) -> List[PatchCollection]:
    # normlengths = (lengths / lengths.max()) ** 2
    normlengths = lengths / lengths.max()
    normlengths = normlengths ** 2
    normlengths *= 1 / 10
    for pos, lens in zip(positions, normlengths):
        # Calculate spherical coordinates
        az, el, _ = cart2sph(*pos)

        # Create radial histogram patches
        coll = PatchCollection([Polygon(
            np.array([[az, el],
                      [az + _len * scale * np.cos(edge1), el + _len * scale * np.sin(edge1)],
                      [az + _len * scale * np.cos(edge2), el + _len * scale * np.sin(edge2)],
                      [az, el],
                      ])) for _len, edge1, edge2 in zip(lens, bin_edges[:-1], bin_edges[1:])],
            color='gray', edgecolor='black', linewidth=.1)

        # Add to axes
        ax.add_collection(coll)

    xticks = np.linspace(-np.pi, np.pi, 5)
    ax.set_xticks(xticks, [int(v / np.pi * 180) for v in xticks])
    ax.set_xlim(-np.pi - 0.2, np.pi + 0.2)
    ax.set_xlim(-np.pi - 0.2, np.pi + 0.2)
    ax.set_xlabel('azimuth [deg]')
    yticks = np.linspace(-np.pi / 2, np.pi / 2, 5)
    ax.set_yticks(yticks, [int(v / np.pi * 180) for v in yticks])
    ax.set_ylim(-np.pi / 2 + np.radians(1), np.pi / 2 - np.radians(1))
    ax.set_ylabel('elevation [deg]')
    ax.set_aspect('equal')

    return ax._children


colors_inh_exc = [(0 / 255, 0 / 255, 255 / 255), (150 / 255, 150 / 255, 150 / 255), (255 / 255, 0 / 255, 0 / 255)]
nodes_inh_exc = [0.0, 0.5, 1.0]
significance_cmap = LinearSegmentedColormap.from_list('blue_gray__red', list(zip(nodes_inh_exc, colors_inh_exc)))


def plot_radial_significance(radial_bin_significance, patch_collections: List[PatchCollection]):
    # Mark significance
    for i, coll in enumerate(patch_collections):
        coll.set_cmap(significance_cmap)
        coll.set_clim(-1, 1)
        coll.set_array(radial_bin_significance[i])


def plot_patch_grid(ax: plt.Axes, corners, indices):
    x = []
    y = []
    patches = []
    # Iterate through patches
    for i in range(indices.shape[0])[::3]:
        vertices = np.append(corners[i:i + 3], corners[None, i], axis=0)

        # Calculate spherical coordinates
        azims, elevs, _ = cart2sph(*vertices.T)

        _add_grid_patch_coordinates(vertices, patches, False)

    ax.scatter(x, y, s=10, c='black')

    # Create grid patches
    coll = PatchCollection([Polygon(np.array(p)) for p in patches],
                           color='None', edgecolor='black', linewidth=.5)
    ax.add_collection(coll)

    return coll


def plot_rf_overview(recording, neuron_num, save_path: str = None):
    print(f'plot {neuron_num} neuron receptive field')
    # Recording data
    radial_bin_etas = recording[f'radial_bin_etas']
    radial_bin_edges = recording['radial_bin_edges']
    radial_bin_significance = recording['radial_bin_significances']
    positions = recording['positions']
    patch_corners = recording['patch_corners']
    patch_indices = recording['patch_indices']

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

    # If Roi has a recepotive field, plot best egomotion fit
    if len(recording['cluster_significant_indices']) > 0:
        # Get position and local motion preference data
        x, y, _ = cart2sph(*positions.T)
        preferred_vectors = recording['preferred_vectors']
        preferred_velocities = np.linalg.norm(recording['preferred_vectors'], axis=1)
        cluster_significant_indices = recording['cluster_significant_indices']
        cluster_unique_patch_indices = recording['cluster_unique_patch_indices']
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
        png_path = os.path.join(save_path, 'png')
        fig_cmn.savefig(f'{png_path}/{neuron_num}.png', dpi=300)
        pdf_path = os.path.join(save_path, 'pdf')
        fig_cmn.savefig(f'{pdf_path}/{neuron_num}.pdf', dpi=300)
        plt.close(fig_cmn)
    else:
        plt.show()
