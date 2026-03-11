from plotting_functions import *
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path

def plot_v1(E1, E3, F1, F3, positions, save_path_=None, neuron_num=None, alpha_E=.5, alpha_F=.5):
    # Plot estimated RF and fitted TOF/ROF together
    # gs_cmn = plt.GridSpec(20, 10)
    fig_cmn = plt.figure(figsize=(20, 10))
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
        pdf_path = os.path.join(save_path_, 'pdf')
        Path(png_path).mkdir(parents=True, exist_ok=True)
        Path(pdf_path).mkdir(parents=True, exist_ok=True)
        fig_cmn.savefig(f'{png_path}/{neuron_num}.png', dpi=300)
        fig_cmn.savefig(f'{pdf_path}/{neuron_num}.pdf', dpi=300)
        plt.close(fig_cmn)
