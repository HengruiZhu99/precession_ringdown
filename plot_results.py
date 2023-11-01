import os
import json
import numpy as np

import quaternion
import spherical_functions as sf

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import PatchCollection
import matplotlib.colors as mplcolors
import matplotlib.cm as cmx

plt.style.use("paper.mplstyle")

# Colorblind friendly from Nichols
colors = [
    "#000000",
    "#0072B2",
    "#009E73",
    "#E69F00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#D55E00",
]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)

# widths for PRL
onecol_w_in = 3.4
twocol_w_in = 7.0625


# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(
                plt.Rectangle(
                    [
                        width / len(orig_handle.colors) * i - handlebox.xdescent,
                        -handlebox.ydescent,
                    ],
                    width / len(orig_handle.colors),
                    height,
                    facecolor=c,
                    edgecolor="none",
                )
            )

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch


def compute_rotation_factor(input_mode, output_mode, theta):
    v = quaternion.quaternion(*np.array([np.sin(theta), 0, np.cos(theta)])).normalized()
    R = (1 - v * quaternion.z).normalized()

    return sf.Wigner_D_element(R, input_mode[0], output_mode[1], input_mode[1])


# Figure 1
def create_Figure1(
    qs, thetas, ratios_A, ratios_B, inset_fig=True, filename="Figure1.pdf"
):
    fig, axis = plt.subplots(
        2,
        2,
        figsize=(twocol_w_in, twocol_w_in * 0.45),
        sharex=False,
        sharey=False,
        height_ratios=[0.05, 1],
    )
    plt.subplots_adjust(hspace=0.02, wspace=0.3)

    # panel A

    result = axis[1][0].scatter(thetas, qs, s=8, c=np.log10(ratios_A), cmap="viridis")

    axis[1][0].set_xlabel(r"misalignment angle $\theta$", fontsize=12)
    axis[1][0].set_ylabel(r"mass ratio $q$", fontsize=12)

    axis[1][0].set_xticks(
        [
            0.0,
            np.pi / 8,
            2 * np.pi / 8,
            3 * np.pi / 8,
            4 * np.pi / 8,
            5 * np.pi / 8,
            6 * np.pi / 8,
            7 * np.pi / 8,
            np.pi,
        ]
    )
    axis[1][0].set_xticklabels(
        [r"$0$", None, r"$\pi/4$", None, r"$\pi/2$", None, r"$3\pi/4$", None, r"$\pi$"]
    )

    c1 = fig.colorbar(result, cax=axis[0][0], orientation="horizontal")

    c1.ax.xaxis.set_ticks_position("top")
    c1.ax.set_xlabel(
        r"$\log_{10}\left(A_{(+,2,1,0)}/A_{(+,2,2,0)}\right)$",
        fontsize=12,
        labelpad=-36,
    )

    if inset_fig:
        im = plt.imread("CCEFigures/SpinMisalignmentCartoon.jpeg")
        newax = fig.add_axes([0.185, 0.145, 0.26, 0.26], anchor="NE", zorder=1)
        newax.imshow(im)
        newax.get_xaxis().set_ticks([])
        newax.get_yaxis().set_ticks([])
        plt.setp(newax.spines.values(), color="lightgrey")

    # panel B

    result = axis[1][1].scatter(thetas, ratios_B, c=qs, s=8, cmap="magma", vmax=8.5)

    angles = np.linspace(0, np.pi, 100)
    rotation_factors = np.array(
        [
            abs(
                compute_rotation_factor((2, 2), (2, 1), angle)
                + compute_rotation_factor((2, 2), (2, -1), angle)
            )
            / abs(
                compute_rotation_factor((2, 2), (2, 2), angle)
                + compute_rotation_factor((2, 2), (2, -2), angle)
            )
            for angle in angles
        ]
    )

    axis[1][1].plot(
        angles,
        rotation_factors,
        label=r"$\cfrac{\mathfrak{D}_{1,2}^{2,\pm}(\theta)}{\mathfrak{D}_{2,2}^{2,\pm}(\theta)}$",
    )

    xlim = axis[1][1].get_xlim()
    axis[1][1].plot(
        np.arange(-np.pi, 2 * np.pi, 0.01),
        np.ones_like(np.arange(-np.pi, 2 * np.pi, 0.01)),
        ls="--",
        color=colors[0],
        lw=1.4,
        alpha=0.6,
    )
    axis[1][1].set_xlim(xlim)

    x = 3.05
    y = 5.8e-1
    ell_offset = ScaledTranslation(x, y, axis[1][1].transScale)
    ell_tform = ell_offset + axis[1][1].transLimits + axis[1][1].transAxes
    axis[1][1].add_patch(
        Ellipse(
            xy=(0, 0),
            width=0.46,
            height=0.28,
            color=colors[0],
            fill=False,
            lw=1,
            zorder=10,
            transform=ell_tform,
        )
    )

    axis[1][1].set_yscale("log")
    axis[1][1].set_xlim(0 - 0.2, np.pi + 0.2)

    axis[1][1].set_xlabel(r"misalignment angle $\theta$", fontsize=12)
    axis[1][1].set_ylabel(r"$A_{(\pm,2,\pm1,0)}/A_{(\pm,2,\pm2,0)}$", fontsize=12)

    axis[1][1].set_xticks(
        [
            0.0,
            np.pi / 8,
            2 * np.pi / 8,
            3 * np.pi / 8,
            4 * np.pi / 8,
            5 * np.pi / 8,
            6 * np.pi / 8,
            7 * np.pi / 8,
            np.pi,
        ]
    )
    axis[1][1].set_xticklabels(
        [r"$0$", None, r"$\pi/4$", None, r"$\pi/2$", None, r"$3\pi/4$", None, r"$\pi$"]
    )

    axis[1][1].legend(loc="lower right", frameon=True, framealpha=1, fontsize=12)

    c2 = fig.colorbar(result, cax=axis[0][1], orientation="horizontal")

    c2.ax.set_xlim(1, 8)
    c2.ax.xaxis.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
    c2.ax.xaxis.set_ticks_position("top")
    c2.ax.set_xlabel(r"mass ratio $q$", fontsize=12, labelpad=-36)

    plt.savefig(f"CCEFigures/{filename}", bbox_inches="tight")


# Figure 2
def create_Figure2(
    thetas, ratios_L2M1, pro_retro_ratios_L2M2, kick_angles, name_suffix=""
):
    fig, axis = plt.subplots(
        3, 1, figsize=(onecol_w_in, onecol_w_in * 1.4), height_ratios=[0.05, 1, 1]
    )
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    result = axis[1].scatter(thetas, ratios_L2M1, c=kick_angles, s=8, cmap="coolwarm")

    axis[1].set_yscale("log")
    axis[1].set_xlim(0 - 0.2, np.pi + 0.2)

    axis[1].set_xticks(
        [
            0.0,
            np.pi / 8,
            2 * np.pi / 8,
            3 * np.pi / 8,
            4 * np.pi / 8,
            5 * np.pi / 8,
            6 * np.pi / 8,
            7 * np.pi / 8,
            np.pi,
        ]
    )
    axis[1].set_xticklabels([])

    angles = np.linspace(0, np.pi, 100)
    rotation_factors = np.array(
        [
            abs(compute_rotation_factor((2, 2), (2, 1), angle))
            / abs(compute_rotation_factor((2, 2), (2, 2), angle))
            for angle in angles
        ]
    )

    # Change label to just be some WignerD notation that we define in methods?
    axis[1].plot(
        angles,
        rotation_factors,
        label=r"$\cfrac{\mathfrak{D}_{1,2}^{2}(\theta)}{\mathfrak{D}_{2,2}^{2}(\theta)}$",
    )

    xlim = axis[1].get_xlim()
    axis[1].plot(
        np.arange(-np.pi, 2 * np.pi, 0.01),
        np.ones_like(np.arange(-np.pi, 2 * np.pi, 0.01)),
        ls="--",
        color=colors[0],
        lw=1.4,
        alpha=0.6,
    )
    axis[1].set_xlim(xlim)

    axis[1].set_ylabel(r"$A_{(+,2,1,0)}/A_{(+,2,2,0)}$", fontsize=12)

    axis[1].legend(loc="lower right", frameon=True, framealpha=1, fontsize=12)

    axis[1].set_ylim(bottom=6e-3, top=2e1)

    axis[2].scatter(thetas, pro_retro_ratios_L2M2, c=kick_angles, s=8, cmap="coolwarm")

    axis[2].set_yscale("log")
    axis[2].set_xlim(0 - 0.2, np.pi + 0.2)

    axis[2].set_xticks(
        [
            0.0,
            np.pi / 8,
            2 * np.pi / 8,
            3 * np.pi / 8,
            4 * np.pi / 8,
            5 * np.pi / 8,
            6 * np.pi / 8,
            7 * np.pi / 8,
            np.pi,
        ]
    )
    axis[2].set_xticklabels(
        [r"$0$", None, r"$\pi/4$", None, r"$\pi/2$", None, r"$3\pi/4$", None, r"$\pi$"]
    )

    angles = np.linspace(0, np.pi, 100)
    rotation_factors = np.array(
        [
            abs(compute_rotation_factor((2, 2), (2, -2), angle))
            / abs(compute_rotation_factor((2, 2), (2, 2), angle))
            for angle in angles
        ]
    )

    axis[2].plot(
        angles,
        rotation_factors,
        label=r"$\cfrac{\mathfrak{D}_{-2,2}^{2}(\theta)}{\mathfrak{D}_{2,2}^{2}(\theta)}$",
    )

    xlim = axis[2].get_xlim()
    axis[2].plot(
        np.arange(-np.pi, 2 * np.pi, 0.01),
        np.ones_like(np.arange(-np.pi, 2 * np.pi, 0.01)),
        ls="--",
        color=colors[0],
        lw=1.4,
        alpha=0.6,
    )
    axis[2].set_xlim(xlim)

    axis[2].set_xlabel(r"misalignment angle $\theta$", fontsize=12)
    axis[2].set_ylabel(r"$A_{(-,2,2,0)}/A_{(+,2,2,0)}$", fontsize=12)

    axis[2].legend(loc="lower right", frameon=True, framealpha=1, fontsize=12)

    axis[2].set_ylim(bottom=6e-6, top=2e3)

    c = fig.colorbar(result, cax=axis[0], orientation="horizontal", pad=0)

    c.ax.xaxis.set_ticks_position("top")
    c.ax.set_xlabel(r"kick velocity angle $\phi$", fontsize=12, labelpad=-30)

    plt.savefig(f"CCEFigures/Figure2.pdf", bbox_inches="tight")


# Figure 3
def create_Figure3(simulations, mirror_mode_ratios, mirror_mode_ratio_errors):
    fig, axis = plt.subplots(1, 1, figsize=(onecol_w_in, onecol_w_in * 0.88))
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    ratio_spreads = []
    for mirror_mode_ratio in mirror_mode_ratios:
        ratio_spreads.append(max(mirror_mode_ratio) / min(mirror_mode_ratio))

    max_ratio_spreads = sorted(ratio_spreads, reverse=True)[1:5]
    min_ratio_spread = min(ratio_spreads)

    count = 1
    for i, mirror_mode_ratio in enumerate(mirror_mode_ratios):
        if ratio_spreads[i] in max_ratio_spreads:
            if ratio_spreads[i] == max_ratio_spreads[1]:
                axis.errorbar(
                    x=np.arange(len(mirror_mode_ratio)),
                    y=mirror_mode_ratio,
                    yerr=mirror_mode_ratio_errors[i],
                    fmt="o-",
                    color=colors[1],
                )
            else:
                axis.errorbar(
                    x=np.arange(len(mirror_mode_ratio)),
                    y=mirror_mode_ratio,
                    yerr=mirror_mode_ratio_errors[i],
                    fmt="o-",
                    alpha=0.4,
                    lw=0.5,
                    color=colors[1 + count],
                )
                count += 1
        elif ratio_spreads[i] == min_ratio_spread:
            axis.errorbar(
                x=np.arange(len(mirror_mode_ratio)),
                y=mirror_mode_ratio,
                yerr=mirror_mode_ratio_errors[i],
                fmt="o--",
                zorder=np.inf,
                lw=1.4,
            )

    axis.set_yscale("log")

    axis.set_xticks(np.arange(len(mirror_mode_ratios[0])))
    axis.set_xticklabels([r"$(2,2)$", r"$(2,1)$", r"$(3,3)$", r"$(3,2)$", r"$(3,1)$"])

    h, l = axis.get_legend_handles_labels()
    h.append(MulticolorPatch(["black"]))
    l.append("non-precessing")
    h.append(MulticolorPatch(colors[1:5]))
    l.append("precessing")
    axis.legend(
        h,
        l,
        loc="lower left",
        handler_map={MulticolorPatch: MulticolorPatchHandler()},
        frameon=True,
        framealpha=1,
        fontsize=12,
    )

    axis.set_xlabel(r"$(\ell,|m|)$", fontsize=12)
    axis.set_ylabel(r"$A_{(+,\ell,m,0)}/A_{(+,\ell,-m,0)}$", fontsize=12)

    plt.savefig(f"CCEFigures/Figure3.pdf", bbox_inches="tight")


def create_Figure1_supplement(t, time_dependent_thetas, qs):
    fig, axis = plt.subplots(2, 1, figsize=(onecol_w_in, onecol_w_in * 0.88), height_ratios=[0.05, 1.])
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    cm = plt.get_cmap('magma') 
    cNorm = mplcolors.Normalize(vmin=0, vmax=8.5)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    for i, time_dependent_theta in enumerate(time_dependent_thetas):
        idx1 = np.argmin(abs(t - -500))
        idx2 = np.argmin(abs(t - 20)) + 1 
        axis[1].plot(t[idx1:idx2], time_dependent_theta[idx1:idx2], c=scalarMap.to_rgba(qs[i]))

    axis[1].set_yticks(
        [
            0.0,
            np.pi / 8,
            2 * np.pi / 8,
            3 * np.pi / 8,
            4 * np.pi / 8,
            5 * np.pi / 8,
            6 * np.pi / 8,
            7 * np.pi / 8,
            np.pi,
        ]
    )
    axis[1].set_yticklabels(
        [r"$0$", None, r"$\frac{\pi}{4}$", None, r"$\frac{\pi}{2}$", None, r"$\frac{3\pi}{4}$", None, r"$\pi$"]
    )

    c = fig.colorbar(scalarMap, cax=axis[0], orientation="horizontal", pad=0)

    c.ax.xaxis.set_ticks_position("top")
    c.ax.set_xlabel(r"mass ratio $q$", fontsize=12, labelpad=-30)
    c.ax.set_xlim(1, 8)
    c.ax.xaxis.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])

    axis[1].set_xlabel(r"$(t-t_{\mathrm{peak}})/M$", fontsize=12)
    axis[1].set_ylabel(r"misalignment angle $\theta$", fontsize=12)

    plt.savefig(f"CCEFigures/supplement_Figure1.pdf", bbox_inches="tight")

def create_Figure2_supplement(thetas, errors, qs):
    fig, axis = plt.subplots(2, 1, figsize=(onecol_w_in, onecol_w_in * 0.88), height_ratios=[0.05, 1.])
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    result = axis[1].scatter(thetas, np.sqrt(2.*errors), c=qs, s=8, cmap='magma', vmax=8.5)
    axis[1].set_yscale('log')

    axis[1].set_xticks(
        [
            0.0,
            np.pi / 8,
            2 * np.pi / 8,
            3 * np.pi / 8,
            4 * np.pi / 8,
            5 * np.pi / 8,
            6 * np.pi / 8,
            7 * np.pi / 8,
            np.pi,
        ]
    )
    axis[1].set_xticklabels(
        [r"$0$", None, r"$\pi/4$", None, r"$\pi/2$", None, r"$3\pi/4$", None, r"$\pi$"]
    )

    c = fig.colorbar(result, cax=axis[0], orientation="horizontal", pad=0)

    c.ax.xaxis.set_ticks_position("top")
    c.ax.set_xlabel(r"mass ratio $q$", fontsize=12, labelpad=-30)
    c.ax.set_xlim(1, 8)
    c.ax.xaxis.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])

    axis[1].set_xlabel(r"misalignment angle $\theta$", fontsize=12)
    axis[1].set_ylabel(r"relative error", fontsize=12)

    plt.savefig(f"CCEFigures/supplement_Figure2.pdf", bbox_inches="tight")

def create_Figure3_supplement(thetas, chi_ps, qs):
    fig, axis = plt.subplots(4, 1, figsize=(onecol_w_in, onecol_w_in * 0.88), height_ratios=[0.108, 1., 0.1, 1.])
    plt.subplots_adjust(hspace=0.05, wspace=0.02)

    result = axis[1].scatter(thetas, chi_ps, c=qs, s=8, cmap='magma', vmax=8.5)
    axis[1].set_yscale('log')
    axis[1].set_ylim(2e-1, 1e0)

    axis[1].set_yticks([2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 1e0])
    axis[1].set_yticklabels([None, None, None, None, None, None, None, None, r'$10^{0}$'])

    result = axis[3].scatter(thetas, chi_ps, c=qs, s=8, cmap='magma', vmax=8.5)
    axis[3].set_yscale('log')
    axis[3].set_ylim(top=2e-6, bottom=6e-8)

    axis[2].set_visible(False)

    axis[1].spines.bottom.set_visible(False)
    axis[3].spines.top.set_visible(False)
    axis[1].xaxis.tick_top()
    axis[1].tick_params(labeltop=False)
    axis[3].xaxis.tick_bottom()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    axis[1].plot([0, 1], [0, 0], transform=axis[1].transAxes, **kwargs)
    axis[3].plot([0, 1], [1, 1], transform=axis[3].transAxes, **kwargs)

    for i in [1, 3]:
        axis[i].set_xticks(
            [
                0.0,
                np.pi / 8,
                2 * np.pi / 8,
                3 * np.pi / 8,
                4 * np.pi / 8,
                5 * np.pi / 8,
                6 * np.pi / 8,
                7 * np.pi / 8,
                np.pi,
            ]
        )
        axis[i].set_xticklabels(
            [r"$0$", None, r"$\pi/4$", None, r"$\pi/2$", None, r"$3\pi/4$", None, r"$\pi$"]
        )
        
    c = fig.colorbar(result, cax=axis[0], orientation="horizontal", pad=0)

    c.ax.xaxis.set_ticks_position("top")
    c.ax.set_xlabel(r"mass ratio $q$", fontsize=12, labelpad=-30)
    c.ax.set_xlim(1, 8)
    c.ax.xaxis.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])

    axis[3].set_xlabel(r"misalignment angle $\theta$", fontsize=12)
    axis[3].set_ylabel(r"$\chi_{p}$", fontsize=12, y=1.2)

    plt.savefig(f"CCEFigures/supplement_Figure3.pdf", bbox_inches="tight")

def compute_mode_amplitude(data, mode, pro_retro=False, mirror=False):
    L, M, N, S = mode
    if pro_retro and not mirror:
        # prograde
        A_mode_pro = (
            data[str((L, M, N, S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, M, N, S)).replace(" ", "")]["A"][1]
        )
        A_mode_pro_std_re = data[str((L, M, N, S)).replace(" ", "")]["A_std"][0]
        A_mode_pro_std_im = data[str((L, M, N, S)).replace(" ", "")]["A_std"][1]

        # retrograde
        A_mode_retro = (
            data[str((L, M, N, -S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, M, N, -S)).replace(" ", "")]["A"][1]
        )
        A_mode_retro_std_re = data[str((L, M, N, -S)).replace(" ", "")]["A_std"][0]
        A_mode_retro_std_im = data[str((L, M, N, -S)).replace(" ", "")]["A_std"][1]

        return abs(A_mode_pro + A_mode_retro), np.sqrt(
            (A_mode_pro_std_re**2 + A_mode_retro_std_re**2)
            + (A_mode_pro_std_im**2 + A_mode_retro_std_im**2)
        )
    elif not pro_retro and mirror:
        # positive M
        A_mode_pM = abs(
            data[str((L, M, N, S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, M, N, S)).replace(" ", "")]["A"][1]
        )
        A_mode_pM_std = np.sqrt(
            data[str((L, M, N, S)).replace(" ", "")]["A_std"][0] ** 2
            + data[str((L, M, N, S)).replace(" ", "")]["A_std"][1] ** 2
        )

        A_mode_nM = abs(
            data[str((L, -M, N, -S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, -M, N, -S)).replace(" ", "")]["A"][1]
        )
        A_mode_nM_std = np.sqrt(
            data[str((L, -M, N, -S)).replace(" ", "")]["A_std"][0] ** 2
            + data[str((L, -M, N, -S)).replace(" ", "")]["A_std"][1] ** 2
        )

        Q_sum = np.sqrt(A_mode_pM**2 + A_mode_nM**2)

        return Q_sum, np.sqrt(
            (A_mode_pM * A_mode_pM_std**2 + A_mode_nM * A_mode_nM_std**2) / Q_sum
        )
    elif pro_retro and mirror:
        # prograde positive M
        A_mode_pro_pM = (
            data[str((L, M, N, S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, M, N, S)).replace(" ", "")]["A"][1]
        )
        A_mode_pro_pM_std_re = data[str((L, M, N, S)).replace(" ", "")]["A_std"][0]
        A_mode_pro_pM_std_im = data[str((L, M, N, S)).replace(" ", "")]["A_std"][1]

        # retrograde positive M
        A_mode_retro_pM = (
            data[str((L, M, N, -S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, M, N, -S)).replace(" ", "")]["A"][1]
        )
        A_mode_retro_pM_std_re = data[str((L, M, N, -S)).replace(" ", "")]["A_std"][0]
        A_mode_retro_pM_std_im = data[str((L, M, N, -S)).replace(" ", "")]["A_std"][1]

        # prograde negative M
        A_mode_pro_nM = (
            data[str((L, -M, N, -S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, -M, N, -S)).replace(" ", "")]["A"][1]
        )
        A_mode_pro_nM_std_re = data[str((L, -M, N, -S)).replace(" ", "")]["A_std"][0]
        A_mode_pro_nM_std_im = data[str((L, -M, N, -S)).replace(" ", "")]["A_std"][1]

        # retrograde negative M
        A_mode_retro_nM = (
            data[str((L, -M, N, S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, -M, N, S)).replace(" ", "")]["A"][1]
        )
        A_mode_retro_nM_std_re = data[str((L, -M, N, S)).replace(" ", "")]["A_std"][0]
        A_mode_retro_nM_std_im = data[str((L, -M, N, S)).replace(" ", "")]["A_std"][1]

        A_mode_pM = abs(A_mode_pro_pM + A_mode_retro_pM)
        A_mode_nM = abs(A_mode_pro_nM + A_mode_retro_nM)

        Q_sum = np.sqrt(A_mode_pM**2 + A_mode_nM**2)

        A_mode_pM_std = np.sqrt(
            (A_mode_pro_pM_std_re**2 + A_mode_retro_pM_std_re**2)
            + (A_mode_pro_pM_std_im**2 + A_mode_retro_pM_std_im**2)
        )
        A_mode_nM_std = np.sqrt(
            (A_mode_pro_nM_std_re**2 + A_mode_retro_nM_std_re**2)
            + (A_mode_pro_nM_std_im**2 + A_mode_retro_nM_std_im**2)
        )

        return Q_sum, np.sqrt(
            (A_mode_pM * A_mode_pM_std**2 + A_mode_nM * A_mode_nM_std**2) / Q_sum
        )
    else:
        A_mode = abs(
            data[str((L, M, N, S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, M, N, S)).replace(" ", "")]["A"][1]
        )
        A_mode_std = np.sqrt(
            data[str((L, M, N, S)).replace(" ", "")]["A_std"][0] ** 2
            + data[str((L, M, N, S)).replace(" ", "")]["A_std"][1] ** 2
        )

        return A_mode, A_mode_std


def compute_ratio(
    data,
    mode1,
    mode2,
    mode1_pro_retro=False,
    mode2_pro_retro=False,
    mode1_mirror=False,
    mode2_mirror=False,
):
    A_mode1, A_mode1_std = compute_mode_amplitude(
        data, mode1, mode1_pro_retro, mode1_mirror
    )
    A_mode2, A_mode2_std = compute_mode_amplitude(
        data, mode2, mode2_pro_retro, mode2_mirror
    )

    return A_mode1 / A_mode2, (A_mode1 / A_mode2) * np.sqrt(
        (A_mode1_std / A_mode1) ** 2 + (A_mode2_std / A_mode2) ** 2
    )


def compute_asymmetry_statistics(data):
    asymms = []
    for L in range(2, 3 + 1):
        for M in range(1, L + 1):
            # positive M
            A_mode_p_pro = (
                data[str((L, M, 0, 1)).replace(" ", "")]["A"][0]
                + 1j * data[str((L, M, 0, 1)).replace(" ", "")]["A"][1]
            )
            A_mode_p_pro_std_re = data[str((L, M, 0, 1)).replace(" ", "")]["A_std"][0]
            A_mode_p_pro_std_im = data[str((L, M, 0, 1)).replace(" ", "")]["A_std"][1]

            A_mode_p_retro = (
                data[str((L, M, 0, -1)).replace(" ", "")]["A"][0]
                + 1j * data[str((L, M, 0, -1)).replace(" ", "")]["A"][1]
            )
            A_mode_p_retro_std_re = data[str((L, M, 0, -1)).replace(" ", "")]["A_std"][
                0
            ]
            A_mode_p_retro_std_im = data[str((L, M, 0, -1)).replace(" ", "")]["A_std"][
                1
            ]

            A_mode_p = A_mode_p_pro + A_mode_p_retro
            A_mode_p_std_re = np.sqrt(
                (A_mode_p_pro_std_re**2 + A_mode_p_retro_std_re**2)
            )
            A_mode_p_std_im = np.sqrt(
                (A_mode_p_pro_std_im**2 + A_mode_p_retro_std_im**2)
            )

            # negative M
            A_mode_n_pro = (
                data[str((L, -M, 0, -1)).replace(" ", "")]["A"][0]
                + 1j * data[str((L, -M, 0, -1)).replace(" ", "")]["A"][1]
            )
            A_mode_n_pro_std_re = data[str((L, -M, 0, -1)).replace(" ", "")]["A_std"][0]
            A_mode_n_pro_std_im = data[str((L, -M, 0, -1)).replace(" ", "")]["A_std"][1]

            A_mode_n_retro = (
                data[str((L, -M, 0, 1)).replace(" ", "")]["A"][0]
                + 1j * data[str((L, -M, 0, 1)).replace(" ", "")]["A"][1]
            )
            A_mode_n_retro_std_re = data[str((L, -M, 0, 1)).replace(" ", "")]["A_std"][
                0
            ]
            A_mode_n_retro_std_im = data[str((L, -M, 0, 1)).replace(" ", "")]["A_std"][
                1
            ]

            A_mode_n = A_mode_n_pro + A_mode_n_retro
            A_mode_n_std_re = np.sqrt(
                (A_mode_n_pro_std_re**2 + A_mode_n_retro_std_re**2)
            )
            A_mode_n_std_im = np.sqrt(
                (A_mode_n_pro_std_im**2 + A_mode_n_retro_std_im**2)
            )

            asymm = abs(A_mode_p) - abs((-1) ** L * np.conjugate(A_mode_n))
            asymm_std = np.sqrt(
                (A_mode_p_std_re**2 + A_mode_p_std_im**2)
                + (A_mode_n_std_re**2 + A_mode_n_std_im**2)
            )

            asymms.append(asymm)

    return np.array(asymms)


def main():
    # Load data from QNM fits
    with open("QNM_results.json") as input_file:
        data = json.load(input_file)

    # Construct relevant arrays for ratios, parameters, etc.
    qs = []
    chi_ps = []

    thetas = []
    kick_angles = []

    errors = []
    mismatches = []
    t0s = []
    CVs = []

    ratios_L2M1 = []
    ratios_L2M1_pro_retro = []
    ratios_L2M1_mirror = []
    ratios_L2M1_pro_retro_mirror = []
    pro_retro_ratios_L2M2 = []
    pro_retro_ratios_L2M1 = []

    mirror_mode_ratios = []
    mirror_mode_ratio_errors = []

    asymms = []

    time_dependent_thetas = []

    for i, simulation in enumerate(data):
        q = data[simulation]["q"]
        qs.append(q)

        chi1 = data[simulation]["chi1"]
        chi2 = data[simulation]["chi2"]

        # Patricia Schmidt definition
        sin_theta1 = np.sin(np.arccos(np.dot(chi1, [0, 0, 1]) / np.linalg.norm(chi1)))
        sin_theta2 = np.sin(np.arccos(np.dot(chi2, [0, 0, 1]) / np.linalg.norm(chi2)))
        chi_p = max(np.linalg.norm(chi1[:2]), np.linalg.norm(chi2[:2]))
        chi_ps.append(chi_p)

        thetas.append(data[simulation]["theta"])
        kick_angles.append(data[simulation]["kick theta"])

        errors.append(data[simulation]["error"])
        mismatches.append(data[simulation]["mismatch"])
        t0s.append(data[simulation]["best t0"])
        CVs.append(data[simulation]["best CV"])

        ratios_L2M1.append(
            compute_ratio(data[simulation], (2, 1, 0, 1), (2, 2, 0, 1))[0]
        )
        ratios_L2M1_pro_retro.append(
            compute_ratio(
                data[simulation],
                (2, 1, 0, 1),
                (2, 2, 0, 1),
                mode1_pro_retro=True,
                mode2_pro_retro=True,
            )[0]
        )
        ratios_L2M1_mirror.append(
            compute_ratio(
                data[simulation],
                (2, 1, 0, 1),
                (2, 2, 0, 1),
                mode1_mirror=True,
                mode2_mirror=True,
            )[0]
        )
        ratios_L2M1_pro_retro_mirror.append(
            compute_ratio(
                data[simulation],
                (2, 1, 0, 1),
                (2, 2, 0, 1),
                mode1_pro_retro=True,
                mode2_pro_retro=True,
                mode1_mirror=True,
                mode2_mirror=True,
            )[0]
        )
        pro_retro_ratios_L2M2.append(
            compute_ratio(data[simulation], (2, 2, 0, -1), (2, 2, 0, 1))[0]
        )
        pro_retro_ratios_L2M1.append(
            compute_ratio(data[simulation], (2, 1, 0, -1), (2, 1, 0, 1))[0]
        )

        mirror_mode_ratio = []
        mirror_mode_ratio_error = []
        for L, M in [(2, 2), (2, 1), (3, 3), (3, 2), (3, 1)]:
            ratio, ratio_error = compute_ratio(
                data[simulation], (L, M, 0, 1), (L, -M, 0, -1)
            )
            mirror_mode_ratio.append(ratio)
            mirror_mode_ratio_error.append(ratio_error)
        mirror_mode_ratios.append(mirror_mode_ratio)
        mirror_mode_ratio_errors.append(mirror_mode_ratio_error)

        asymms.append(compute_asymmetry_statistics(data[simulation]))

        if len(data[simulation]['thetas']) == len(np.arange(-1000, 250, 0.1)):
            time_dependent_thetas.append(data[simulation]['thetas'])
        else:
            time_dependent_thetas.append([None]*len(np.arange(-1000, 250, 0.1)))
            
    qs = np.array(qs)
    chi_ps = np.array(chi_ps)

    thetas = np.array(thetas)
    kick_angles = np.array(kick_angles)

    errors = np.array(errors)
    mismatches = np.array(mismatches)
    t0s = np.array(t0s)
    CVs = np.array(CVs)

    ratios_L2M1 = np.array(ratios_L2M1)
    ratios_L2M1_pro_retro = np.array(ratios_L2M1_pro_retro)
    ratios_L2M1_mirror = np.array(ratios_L2M1_mirror)
    ratios_L2M1_pro_retro_mirror = np.array(ratios_L2M1_pro_retro_mirror)
    pro_retro_ratios_L2M2 = np.array(pro_retro_ratios_L2M2)
    pro_retro_ratios_L2M1 = np.array(pro_retro_ratios_L2M1)

    mirror_mode_ratios = np.array(mirror_mode_ratios)
    mirror_mode_ratio_errors = np.array(mirror_mode_ratio_errors)

    asymms = np.array(asymms)

    time_dependent_thetas = np.array(time_dependent_thetas)

    create_Figure1(
        qs, thetas, ratios_L2M1, ratios_L2M1_pro_retro_mirror, filename="Figure1.pdf"
    )

    create_Figure2(thetas, ratios_L2M1, pro_retro_ratios_L2M2, kick_angles)

    create_Figure3(list(data.keys()), mirror_mode_ratios, mirror_mode_ratio_errors)
    
    create_Figure1_supplement(np.arange(-1000, 250, 0.1), time_dependent_thetas, qs)

    create_Figure2_supplement(thetas, errors, qs)

    create_Figure3_supplement(thetas, chi_ps, qs)

if __name__ == "__main__":
    main()
