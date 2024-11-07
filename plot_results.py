import os
import json
import numpy as np

import scri
import quaternion
import spherical_functions as sf

from quaternion.calculus import derivative
from quaternion.calculus import indefinite_integral as integrate
from scri.mode_calculations import LLDominantEigenvector

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
    "#F0E442",
    "#56B4E9",
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
    """Compute the WignerD matrices using spherical_functions.

    Parameters
    ----------
    input_mode: tuple
        Mode, i.e., (\ell, m), to be rotated.
    output_mode: tuple
        Mode to be rotated into.
    theta: float

    Returns
    -------
    WignerD_coeff: complex
        WignerD matrix coefficient.

    """
    v = quaternion.quaternion(*np.array([np.sin(theta), 0, np.cos(theta)])).normalized()
    R = (1 - v * quaternion.z).normalized()

    WignerD_coeff = sf.Wigner_D_element(R, input_mode[0], output_mode[1], input_mode[1])

    return WignerD_coeff


def rotation_factor_theory(q, chi_eff=0):
    """Compute the theory prediction for the relative QNM excitation of the (2,1)/(2,2) QNM ratio.

    Parameters
    ----------
    q : float
        Mass ratio.
    chi_eff : float, optional
        Effective spin. [Default: 0.]

    Returns
    -------
    angles : ndarray
        Angles, from 0 to \pi, used to compute the rotation curve.
    rotation_factors: ndarray
        Predictions for relative QNM excitation.

    """
    nu = q / (1 + q) ** 2
    ratio = 0.43 * (np.sqrt(1 - 4 * nu) - chi_eff)

    angles = np.linspace(0, np.pi, 100)
    rotation_factors = np.array(
        [
            np.sqrt(
                (ratio * compute_rotation_factor((2, 1), (2, 1), angle)) ** 2
                + (ratio * compute_rotation_factor((2, 1), (2, -1), angle)) ** 2
                + compute_rotation_factor((2, 2), (2, 1), angle) ** 2
                + compute_rotation_factor((2, 2), (2, -1), angle) ** 2
            )
            / np.sqrt(
                (ratio * compute_rotation_factor((2, 1), (2, 2), angle)) ** 2
                + (ratio * compute_rotation_factor((2, 1), (2, -2), angle)) ** 2
                + compute_rotation_factor((2, 2), (2, 2), angle) ** 2
                + compute_rotation_factor((2, 2), (2, -2), angle) ** 2
            )
            for angle in angles
        ]
    )

    return angles, rotation_factors


# Figure 1
def create_L2M1_and_L2M0_figure(
    qs, thetas, ratios_L2M1_pro_retro_mirror, ratios_L2M0_pro_retro_mirror
):
    fig, axis = plt.subplot_mosaic(
        [["A", "A"], ["B", "C"]],
        figsize=(twocol_w_in, twocol_w_in * 0.4),
        height_ratios=[0.05, 1],
    )
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    # panel B
    result = axis["B"].scatter(
        [None] * len(qs), [None] * len(qs), c=qs, s=8, cmap="magma", vmax=8.5
    )

    cm = plt.get_cmap("magma")
    cNorm = mplcolors.Normalize(vmin=min(qs), vmax=8.5)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    for i in range(len(thetas)):
        markers, caps, bars = axis["B"].errorbar(
            thetas[i],
            ratios_L2M1_pro_retro_mirror[:, 0][i],
            yerr=ratios_L2M1_pro_retro_mirror[:, 1][i],
            fmt="o",
            markersize=np.sqrt(8),
            color=scalarMap.to_rgba(qs[i]),
            alpha=0.8,
        )
        [bar.set_alpha(0.4) for bar in bars]

    angles = np.linspace(0, np.pi, 100)
    rotation_factors = np.array(
        [
            np.sqrt(
                compute_rotation_factor((2, 2), (2, 1), angle) ** 2
                + compute_rotation_factor((2, 2), (2, -1), angle) ** 2
            )
            / np.sqrt(
                compute_rotation_factor((2, 2), (2, 2), angle) ** 2
                + compute_rotation_factor((2, 2), (2, -2), angle) ** 2
            )
            for angle in angles
        ]
    )

    axis["B"].plot(
        angles,
        rotation_factors,
        label="rotation of $q=1$,\n non-spinning\n perturbation",
        lw=1.4,
        alpha=0.6,
        zorder=np.inf,
    )

    _, rotation_factors = rotation_factor_theory(8, -0.6)
    axis["B"].plot(
        angles,
        rotation_factors,
        label="$q=8$, $\chi_{\mathrm{diff}}=-0.6$",
        lw=1.4,
        alpha=0.6,
        zorder=np.inf,
        color=colors[3],
    )

    xlim = axis["B"].get_xlim()
    axis["B"].plot(
        np.arange(-np.pi, 2 * np.pi, 0.01),
        np.ones_like(np.arange(-np.pi, 2 * np.pi, 0.01)),
        ls="--",
        color=colors[0],
        lw=1.4,
        alpha=0.6,
        zorder=np.inf - 1,
    )
    axis["B"].set_xlim(xlim)

    axis["B"].set_yscale("log")
    axis["B"].set_xlim(0 - 0.2, np.pi + 0.2)

    axis["B"].set_xlabel(r"misalignment angle $\theta$", fontsize=10)
    axis["B"].set_ylabel(r"$A_{(\pm,2,\pm1,0)}/A_{(\pm,2,\pm2,0)}$", fontsize=10)

    axis["B"].set_xticks(
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
    axis["B"].set_xticklabels(
        [r"$0$", None, r"$\pi/4$", None, r"$\pi/2$", None, r"$3\pi/4$", None, r"$\pi$"]
    )

    axis["B"].set_ylim(3e-3, 3e0)

    leg = axis["B"].legend(
        loc="lower center",
        ncol=1,
        frameon=True,
        framealpha=1,
        fontsize=8,
        columnspacing=-1.8,
    )
    leg.set_zorder(np.inf)

    # panel C

    result = axis["C"].scatter(
        [None] * len(qs), [None] * len(qs), c=qs, s=8, cmap="magma", vmax=8.5
    )

    cm = plt.get_cmap("magma")
    cNorm = mplcolors.Normalize(vmin=min(qs), vmax=8.5)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    for i in range(len(thetas)):
        markers, caps, bars = axis["C"].errorbar(
            thetas[i],
            ratios_L2M0_pro_retro_mirror[:, 0][i],
            yerr=ratios_L2M0_pro_retro_mirror[:, 1][i],
            fmt="o",
            markersize=np.sqrt(8),
            color=scalarMap.to_rgba(qs[i]),
            alpha=0.8,
        )
        [bar.set_alpha(0.4) for bar in bars]

    angles = np.linspace(0, np.pi, 100)
    rotation_factors = np.array(
        [
            np.sqrt(
                compute_rotation_factor((2, 2), (2, 0), angle) ** 2
                + compute_rotation_factor((2, 2), (2, 0), angle) ** 2
            )
            / np.sqrt(
                compute_rotation_factor((2, 2), (2, 2), angle) ** 2
                + compute_rotation_factor((2, 2), (2, -2), angle) ** 2
            )
            for angle in angles
        ]
    )

    axis["C"].plot(
        angles,
        rotation_factors,
        label="rotation of $q=1$,\n non-spinning\n perturbation",
        lw=1.4,
        alpha=0.6,
        zorder=np.inf,
    )

    xlim = axis["C"].get_xlim()
    axis["C"].plot(
        np.arange(-np.pi, 2 * np.pi, 0.01),
        np.ones_like(np.arange(-np.pi, 2 * np.pi, 0.01)),
        ls="--",
        color=colors[0],
        lw=1.4,
        alpha=0.6,
        zorder=np.inf - 1,
    )
    axis["C"].set_xlim(xlim)

    axis["C"].set_yscale("log")
    axis["C"].set_xlim(0 - 0.2, np.pi + 0.2)

    axis["C"].set_xlabel(r"misalignment angle $\theta$", fontsize=10)
    axis["C"].set_ylabel(
        r"$A_{(\pm,2,0,0)}/A_{(\pm,2,\pm2,0)}$", fontsize=10, rotation=270, labelpad=16
    )
    axis["C"].yaxis.set_label_position("right")
    axis["C"].yaxis.tick_right()

    axis["C"].set_xticks(
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
    axis["C"].set_xticklabels(
        [r"$0$", None, r"$\pi/4$", None, r"$\pi/2$", None, r"$3\pi/4$", None, r"$\pi$"]
    )
    axis["C"].set_yticklabels([])

    axis["C"].set_ylim(3e-3, 3e0)

    # Colorbar
    c = fig.colorbar(result, cax=axis["A"], orientation="horizontal")

    c.ax.xaxis.set_ticks_position("top")
    c.ax.set_xlim(1, 8)
    c.ax.xaxis.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
    c.ax.set_xlabel(r"mass ratio $q$", fontsize=10, labelpad=-30)

    im = plt.imread("CCEFigures/SpinMisalignmentCartoon.jpeg")
    newax = fig.add_axes([-0.003, 0.34, 0.64, 0.28], zorder=1)
    newax.imshow(im)
    newax.get_xaxis().set_ticks([])
    newax.get_yaxis().set_ticks([])
    plt.setp(newax.spines.values(), color="lightgrey")

    plt.savefig(f"CCEFigures/L2M1_and_L2M0_vs_prediction.pdf", bbox_inches="tight")


# Figure 2
def create_kick_velocity_figure(
    thetas, ratios_L2M0, pro_retro_ratios_L2M2, kick_angles, name_suffix=""
):
    fig, axis = plt.subplots(
        3, 1, figsize=(onecol_w_in, onecol_w_in * 1.4), height_ratios=[0.05, 1, 1]
    )
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    result = axis[1].scatter(
        [None] * len(kick_angles),
        [None] * len(kick_angles),
        c=kick_angles,
        s=8,
        cmap="coolwarm",
    )

    cm = plt.get_cmap("coolwarm")
    cNorm = mplcolors.Normalize(vmin=min(kick_angles), vmax=max(kick_angles))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    for i in range(len(thetas)):
        markers, caps, bars = axis[1].errorbar(
            thetas[i],
            ratios_L2M0[:, 0][i],
            yerr=ratios_L2M0[:, 1][i],
            fmt="o",
            markersize=np.sqrt(8),
            color=scalarMap.to_rgba(kick_angles[i]),
            alpha=0.8,
        )
        [bar.set_alpha(0.4) for bar in bars]

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
            abs(compute_rotation_factor((2, 2), (2, 0), angle))
            / abs(compute_rotation_factor((2, 2), (2, 2), angle))
            for angle in angles
        ]
    )

    axis[1].plot(
        angles,
        rotation_factors,
        label=r"$\cfrac{\mathfrak{D}_{0,2}^{2}(\theta)}{\mathfrak{D}_{2,2}^{2}(\theta)}$",
        zorder=np.inf,
    )

    xlim = axis[1].get_xlim()
    axis[1].plot(
        np.arange(-np.pi, 2 * np.pi, 0.01),
        np.ones_like(np.arange(-np.pi, 2 * np.pi, 0.01)),
        ls="--",
        color=colors[0],
        lw=1.4,
        alpha=0.6,
        zorder=np.inf - 1,
    )
    axis[1].set_xlim(xlim)

    axis[1].set_ylabel(r"$A_{(+,2,0,0)}/A_{(+,2,2,0)}$", fontsize=10)

    axis[1].legend(loc="lower right", frameon=True, framealpha=1, fontsize=10)

    axis[1].set_ylim(2e-4, 2e2)

    result = axis[2].scatter(
        [None] * len(kick_angles),
        [None] * len(kick_angles),
        c=kick_angles,
        s=8,
        cmap="coolwarm",
    )

    cm = plt.get_cmap("coolwarm")
    cNorm = mplcolors.Normalize(vmin=min(kick_angles), vmax=max(kick_angles))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    for i in range(len(thetas)):
        markers, caps, bars = axis[2].errorbar(
            thetas[i],
            pro_retro_ratios_L2M2[:, 0][i],
            yerr=pro_retro_ratios_L2M2[:, 1][i],
            fmt="o",
            markersize=np.sqrt(8),
            color=scalarMap.to_rgba(kick_angles[i]),
            alpha=0.8,
        )
        [bar.set_alpha(0.4) for bar in bars]

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
        zorder=np.inf,
    )

    xlim = axis[2].get_xlim()
    axis[2].plot(
        np.arange(-np.pi, 2 * np.pi, 0.01),
        np.ones_like(np.arange(-np.pi, 2 * np.pi, 0.01)),
        ls="--",
        color=colors[0],
        lw=1.4,
        alpha=0.6,
        zorder=np.inf - 1,
    )
    axis[2].set_xlim(xlim)

    axis[2].set_xlabel(r"misalignment angle $\theta$", fontsize=10)
    axis[2].set_ylabel(r"$A_{(-,2,-2,0)}/A_{(+,2,2,0)}$", fontsize=10)

    axis[2].legend(loc="lower right", frameon=True, framealpha=1, fontsize=10)

    axis[2].set_ylim(bottom=6e-6, top=2e3)

    c = fig.colorbar(result, cax=axis[0], orientation="horizontal", pad=0)

    c.ax.xaxis.set_ticks_position("top")
    c.ax.set_xticks(
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
    c.ax.set_xticklabels(
        [r"$0$", None, r"$\pi/4$", None, r"$\pi/2$", None, r"$3\pi/4$", None, r"$\pi$"]
    )
    c.ax.set_xlabel(r"kick velocity angle $\phi$", fontsize=10, labelpad=-30)

    plt.savefig(
        f"CCEFigures/kick_velocity_spread_vs_prediction.pdf", bbox_inches="tight"
    )


# Figure 3
def create_mode_asymmetries_figure(
    simulations, mirror_mode_ratios, mirror_mode_ratio_errors, N_systems=5
):
    fig, axis = plt.subplots(1, 1, figsize=(onecol_w_in, onecol_w_in * 0.88))
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    ratio_spreads = []
    for mirror_mode_ratio in mirror_mode_ratios:
        ratio_spreads.append(max(mirror_mode_ratio) / min(mirror_mode_ratio))

    max_ratio_spreads = sorted(ratio_spreads, reverse=True)[1:N_systems]
    min_ratio_spread = min(ratio_spreads)

    count = 1
    for i, mirror_mode_ratio in enumerate(mirror_mode_ratios):
        if ratio_spreads[i] in max_ratio_spreads:
            if ratio_spreads[i] == max_ratio_spreads[1]:
                markers, caps, bars = axis.errorbar(
                    x=np.arange(len(mirror_mode_ratio)),
                    y=mirror_mode_ratio,
                    yerr=mirror_mode_ratio_errors[i],
                    fmt="o-",
                    color=colors[1],
                )
            else:
                markers, caps, bars = axis.errorbar(
                    x=np.arange(len(mirror_mode_ratio)),
                    y=mirror_mode_ratio,
                    yerr=mirror_mode_ratio_errors[i],
                    fmt="o-",
                    alpha=0.4,
                    lw=0.5,
                    color=colors[1 + count],
                )
                count += 1
            [bar.set_alpha(0.4) for bar in bars]
        elif ratio_spreads[i] == min_ratio_spread:
            markers, caps, bars = axis.errorbar(
                x=np.arange(len(mirror_mode_ratio)),
                y=mirror_mode_ratio,
                yerr=mirror_mode_ratio_errors[i],
                fmt="o--",
                zorder=np.inf,
                lw=1.4,
            )
            [bar.set_alpha(0.4) for bar in bars]

    axis.set_yscale("log")

    axis.set_xticks(np.arange(len(mirror_mode_ratios[0])))
    axis.set_xticklabels([r"$(2,2)$", r"$(2,1)$", r"$(3,3)$", r"$(3,2)$", r"$(3,1)$"])

    h, l = axis.get_legend_handles_labels()
    h.append(MulticolorPatch(["black"]))
    l.append("non-precessing")
    h.append(MulticolorPatch(colors[1:N_systems]))
    l.append("precessing")
    axis.legend(
        h,
        l,
        loc="lower left",
        handler_map={MulticolorPatch: MulticolorPatchHandler()},
        frameon=True,
        framealpha=1,
        fontsize=10,
    )

    axis.set_xlabel(r"$(\ell,|m|)$", fontsize=10)
    axis.set_ylabel(r"$A_{(+,\ell,m,0)}/A_{(+,\ell,-m,0)}$", fontsize=10)

    plt.savefig(f"CCEFigures/mode_asymmetries.pdf", bbox_inches="tight")


def create_QNM_fit_error_figure(thetas, errors, chi_fs):
    fig, axis = plt.subplots(
        2, 1, figsize=(onecol_w_in, onecol_w_in * 0.88), height_ratios=[0.05, 1.0]
    )
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    result = axis[1].scatter(
        chi_fs, np.sqrt(2.0 * errors), c=thetas, s=8, cmap="viridis"
    )

    axis[1].set_yscale("log")

    c = fig.colorbar(result, cax=axis[0], orientation="horizontal", pad=0)

    c.ax.set_xticks(
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
    c.ax.set_xticklabels(
        [r"$0$", None, r"$\pi/4$", None, r"$\pi/2$", None, r"$3\pi/4$", None, r"$\pi$"]
    )

    c.ax.xaxis.set_ticks_position("top")
    c.ax.set_xlabel(r"misalignment angle $\theta$", fontsize=10, labelpad=-30)

    axis[1].set_xlabel(r"$\chi_{f}$", fontsize=10)
    axis[1].set_ylabel(r"relative error of QNM fit", fontsize=10)

    plt.savefig(f"CCEFigures/QNM_fit_errors.pdf", bbox_inches="tight")


def create_parity_breaking_figure(thetas, asymms, kick_rapidities):
    fig, axis = plt.subplots(
        4,
        1,
        figsize=(onecol_w_in, onecol_w_in * 0.88),
        height_ratios=[0.108, 1.0, 0.1, 1.0],
    )
    plt.subplots_adjust(hspace=0.05, wspace=0.02)

    result = axis[1].scatter(
        [None] * len(thetas),
        [None] * len(thetas),
        c=kick_rapidities,
        s=8,
        cmap="cividis",
        norm=mpl.colors.LogNorm(vmin=1e-4, vmax=max(kick_rapidities)),
    )

    axis[2].set_visible(False)

    axis[1].spines.bottom.set_visible(False)
    axis[3].spines.top.set_visible(False)
    axis[1].xaxis.tick_top()
    axis[1].tick_params(labeltop=False)
    axis[3].xaxis.tick_bottom()

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=10,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    axis[1].plot([0, 1], [0, 0], transform=axis[1].transAxes, **kwargs)
    axis[3].plot([0, 1], [1, 1], transform=axis[3].transAxes, **kwargs)

    cm = plt.get_cmap("cividis")
    cNorm = mplcolors.LogNorm(vmin=1e-4, vmax=max(kick_rapidities))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    for k in [1, 3]:
        for i in range(len(thetas)):
            markers, caps, bars = axis[k].errorbar(
                thetas[i],
                asymms[:, 0][i],
                yerr=asymms[:, 1][i],
                fmt="o",
                markersize=np.sqrt(8),
                color=scalarMap.to_rgba(kick_rapidities[i]),
            )
            [bar.set_alpha(0.4) for bar in bars]

        axis[k].set_yscale("log")
        axis[k].set_xlim(0 - 0.2, np.pi + 0.2)

        axis[k].set_xticks(
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
        axis[k].set_xticklabels(
            [
                r"$0$",
                None,
                r"$\pi/4$",
                None,
                r"$\pi/2$",
                None,
                r"$3\pi/4$",
                None,
                r"$\pi$",
            ]
        )

    axis[1].set_ylim(bottom=2e-2)
    axis[3].set_ylim(top=6e-5)

    axis[3].set_xlabel(r"misalignment angle $\theta$", fontsize=10)
    axis[3].set_ylabel(r"parity breaking of $\pm m$ QNMs", fontsize=10, y=1.1)

    c = fig.colorbar(result, cax=axis[0], orientation="horizontal", pad=0)

    c.ax.xaxis.set_ticks_position("top")
    c.ax.set_xlabel(r"kick rapidity", fontsize=10, labelpad=-30)

    plt.savefig(f"CCEFigures/parity_breaking_figure.pdf", bbox_inches="tight")


def create_higher_harmonics_vs_prediction_figure(
    thetas, ratios_L3M2, ratios_L3M1, ratios_L3M0, kick_angles
):
    fig, axis = plt.subplot_mosaic(
        [["A", "A", "A"], ["B", "C", "D"]],
        figsize=(twocol_w_in, twocol_w_in * 0.34),
        height_ratios=[0.05, 1],
    )
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    modes = [(3, 2), (3, 1), (3, 0)]
    datas = [ratios_L3M2, ratios_L3M1, ratios_L3M0]
    for i, plot in enumerate(["B", "C", "D"]):
        result = axis[plot].scatter(
            [None] * len(kick_angles),
            [None] * len(kick_angles),
            c=kick_angles,
            s=8,
            cmap="coolwarm",
        )

        cm = plt.get_cmap("coolwarm")
        cNorm = mplcolors.Normalize(vmin=min(kick_angles), vmax=max(kick_angles))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        for j in range(len(thetas)):
            markers, caps, bars = axis[plot].errorbar(
                thetas[j],
                datas[i][:, 0][j],
                yerr=datas[i][:, 1][j],
                fmt="o",
                markersize=np.sqrt(8),
                color=scalarMap.to_rgba(kick_angles[j]),
            )
            [bar.set_alpha(0.4) for bar in bars]

        axis[plot].set_yscale("log")
        axis[plot].set_xlim(0 - 0.2, np.pi + 0.2)

        axis[plot].set_xticks(
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
        axis[plot].set_xticklabels(
            [
                r"$0$",
                None,
                r"$\pi/4$",
                None,
                r"$\pi/2$",
                None,
                r"$3\pi/4$",
                None,
                r"$\pi$",
            ]
        )

        angles = np.linspace(0, np.pi, 100)
        rotation_factors = np.array(
            [
                abs(compute_rotation_factor((3, 3), (3, modes[i][1]), angle))
                / abs(compute_rotation_factor((3, 3), (3, 3), angle))
                for angle in angles
            ]
        )

        axis[plot].plot(
            angles,
            rotation_factors,
            label=r"$\cfrac{\mathfrak{D}_{"
            + str(modes[i][1])
            + r",3}^{3}(\theta)}{\mathfrak{D}_{3,3}^{3}(\theta)}$",
            zorder=np.inf,
        )

        xlim = axis[plot].get_xlim()
        axis[plot].plot(
            np.arange(-np.pi, 2 * np.pi, 0.01),
            np.ones_like(np.arange(-np.pi, 2 * np.pi, 0.01)),
            ls="--",
            color=colors[0],
            lw=1.4,
            alpha=0.6,
            zorder=np.inf - 1,
        )
        axis[plot].set_xlim(xlim)

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        axis[plot].text(
            0.728,
            0.44,
            r"$m=" + str(modes[i][1]) + "$",
            transform=axis[plot].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        axis[plot].legend(loc="lower right", frameon=True, framealpha=1, fontsize=10)

        axis[plot].set_ylim(1e-5, 2e2)
        if i > 0:
            axis[plot].set_yticklabels([])

        axis[plot].set_xlabel(r"misalignment angle $\theta$", fontsize=10)

    axis["B"].set_ylabel(r"$A_{(+,3,m,0)}/A_{(+,3,3,0)}$", fontsize=10)

    c = fig.colorbar(result, cax=axis["A"], orientation="horizontal", pad=0)

    c.ax.xaxis.set_ticks_position("top")
    c.ax.set_xticks(
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
    c.ax.set_xticklabels(
        [r"$0$", None, r"$\pi/4$", None, r"$\pi/2$", None, r"$3\pi/4$", None, r"$\pi$"]
    )
    c.ax.set_xlabel(r"kick velocity angle $\phi$", fontsize=10, labelpad=-30)

    plt.savefig(f"CCEFigures/higher_harmonics_vs_prediction.pdf", bbox_inches="tight")


def create_OShaughnessy_figure(thetas, asymms, qs, mismatches):
    fig, axis = plt.subplots(
        2,
        1,
        figsize=(onecol_w_in, onecol_w_in * 0.88),
        height_ratios=[0.05, 1.0],
    )
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    result = axis[1].scatter(
        [None] * len(thetas),
        [None] * len(thetas),
        c=qs,
        s=8,
        cmap="magma",
        norm=mpl.colors.Normalize(vmin=min(qs), vmax=8.5),
    )

    cm = plt.get_cmap("magma")
    cNorm = mpl.colors.Normalize(vmin=min(qs), vmax=8.5)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    for i in range(len(thetas)):
        if asymms[i] < 1e-2:
            continue
        axis[1].scatter(
            thetas[i],
            mismatches[i],
            s=8,
            color=scalarMap.to_rgba(qs[i]),
        )

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
    axis[1].set_xticklabels(
        [
            r"$0$",
            None,
            r"$\pi/4$",
            None,
            r"$\pi/2$",
            None,
            r"$3\pi/4$",
            None,
            r"$\pi$",
        ]
    )

    axis[1].set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])

    axis[1].set_xlabel(r"misalignment angle $\theta$", fontsize=10)
    axis[1].set_ylabel(
        r"$\mathcal{M}\left(h,h^{\mathrm{from\,coprec.\,via\,Euler\,angle\,approx.}}\right)$",
        fontsize=10,
    )

    c = fig.colorbar(result, cax=axis[0], orientation="horizontal", pad=0)

    c.ax.xaxis.set_ticks_position("top")
    c.ax.set_xlim(1, 8)
    c.ax.xaxis.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
    c.ax.set_xlabel(r"mass ratio $q$", fontsize=10, labelpad=-30)

    plt.savefig("CCEFigures/OShaughnessy_violation.pdf", bbox_inches="tight")


def compute_mode_amplitude(data, mode, pro_retro=False, mirror=False):
    """
    Compute the QNM amplitude and standard deviation from the complex amplitude obtained from fitting.

    Parameters
    ----------
    data : dict
        Fitting data (output by analysis.py).
    mode : tuple
        (\ell, m, n, s) of QNM.
    pro_retro : bool, optional
        Compute the quadrature sum of the prograde and retrograde mode amplitudes. [Default: False].
    mirror: bool, optional
        Compute the quadrature sum of the mirror mode amplitudes. [Default: False].
    """
    L, M, N, S = mode
    if pro_retro and not mirror:
        # prograde
        A_mode_pro = abs(
            data[str((L, M, N, S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, M, N, S)).replace(" ", "")]["A"][1]
        )
        A_mode_pro_std_re = data[str((L, M, N, S)).replace(" ", "")]["A_std"][0]
        A_mode_pro_std_im = data[str((L, M, N, S)).replace(" ", "")]["A_std"][1]

        A_mode_pro_std = np.sqrt(A_mode_pro_std_re**2 + A_mode_pro_std_im**2)

        # retrograde
        A_mode_retro = abs(
            data[str((L, M, N, -S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, M, N, -S)).replace(" ", "")]["A"][1]
        )
        A_mode_retro_std_re = data[str((L, M, N, -S)).replace(" ", "")]["A_std"][0]
        A_mode_retro_std_im = data[str((L, M, N, -S)).replace(" ", "")]["A_std"][1]

        A_mode_retro_std = np.sqrt(A_mode_retro_std_re**2 + A_mode_retro_std_im**2)

        Q_sum = np.sqrt(A_mode_pro**2 + A_mode_retro**2)

        return (
            Q_sum,
            np.sqrt(
                (
                    A_mode_pro**2 * A_mode_pro_std**2
                    + A_mode_retro**2 * A_mode_retro_std**2
                )
            )
            / Q_sum,
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

        return (
            Q_sum,
            np.sqrt(
                (
                    A_mode_pM**2 * A_mode_pM_std**2
                    + A_mode_nM**2 * A_mode_nM_std**2
                )
            )
            / Q_sum,
        )
    elif pro_retro and mirror:
        # prograde positive M
        A_mode_pro_pM = abs(
            data[str((L, M, N, S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, M, N, S)).replace(" ", "")]["A"][1]
        )
        A_mode_pro_pM_std_re = data[str((L, M, N, S)).replace(" ", "")]["A_std"][0]
        A_mode_pro_pM_std_im = data[str((L, M, N, S)).replace(" ", "")]["A_std"][1]

        A_mode_pro_pM_std = np.sqrt(
            A_mode_pro_pM_std_re**2 + A_mode_pro_pM_std_im**2
        )

        # retrograde positive M
        A_mode_retro_pM = abs(
            data[str((L, M, N, -S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, M, N, -S)).replace(" ", "")]["A"][1]
        )
        A_mode_retro_pM_std_re = data[str((L, M, N, -S)).replace(" ", "")]["A_std"][0]
        A_mode_retro_pM_std_im = data[str((L, M, N, -S)).replace(" ", "")]["A_std"][1]

        A_mode_retro_pM_std = np.sqrt(
            A_mode_retro_pM_std_re**2 + A_mode_retro_pM_std_im**2
        )

        # prograde negative M
        A_mode_pro_nM = abs(
            data[str((L, -M, N, -S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, -M, N, -S)).replace(" ", "")]["A"][1]
        )
        A_mode_pro_nM_std_re = data[str((L, -M, N, -S)).replace(" ", "")]["A_std"][0]
        A_mode_pro_nM_std_im = data[str((L, -M, N, -S)).replace(" ", "")]["A_std"][1]

        A_mode_pro_nM_std = np.sqrt(
            A_mode_pro_nM_std_re**2 + A_mode_pro_nM_std_im**2
        )

        # retrograde negative M
        A_mode_retro_nM = abs(
            data[str((L, -M, N, S)).replace(" ", "")]["A"][0]
            + 1j * data[str((L, -M, N, S)).replace(" ", "")]["A"][1]
        )
        A_mode_retro_nM_std_re = data[str((L, -M, N, S)).replace(" ", "")]["A_std"][0]
        A_mode_retro_nM_std_im = data[str((L, -M, N, S)).replace(" ", "")]["A_std"][1]

        A_mode_retro_nM_std = np.sqrt(
            A_mode_retro_nM_std_re**2 + A_mode_retro_nM_std_im**2
        )

        # Combine
        Q_sum = np.sqrt(
            A_mode_pro_pM**2
            + A_mode_retro_pM**2
            + A_mode_pro_nM**2
            + A_mode_retro_nM**2
        )

        return (
            Q_sum,
            np.sqrt(
                (
                    A_mode_pro_pM**2 * A_mode_pro_pM_std**2
                    + A_mode_retro_pM**2 * A_mode_retro_pM_std**2
                    + A_mode_pro_nM**2 * A_mode_pro_nM_std**2
                    + A_mode_retro_nM**2 * A_mode_retro_nM_std**2
                )
            )
            / Q_sum,
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
    """
    Compute the ratio between two QNM amplitudes.

    Parameters
    ----------
    data : dict
        Fitting data (output by analysis.py).
    mode1 : tuple
        (\ell, m, n, s) of numerator QNM.
    mode2 : tuple
        (\ell, m, n, s) of denominator QNM.
    mode1_pro_retro : bool, optional
        Whether or not to compute the quadrature sum of the prograde/retrograde mode amplitudes for the numerator QNM.
        [Default: False].
    mode2_pro_retro : bool, optional
        Whether or not to compute the quadrature sum of the prograde/retrograde mode amplitudes for the denominator QNM.
        [Default: False].
    mode1_mirror : bool, optional
        Whether or not to compute the quadrature sum of the mirror mode amplitudes for the numerator QNM.
        [Default: False].
    mode2_mirror : bool, optional
        Whether or not to compute the quadrature sum of the mirror mode amplitudes for the denominator QNM.
        [Default: False].

    Returns
    -------
    A_ratio : float
        Ratio of QNM amplitudes.
    A_ratio_std : float
        Standard deviation of ratio of QNM amplitudes.

    """
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
    """
    Compute the violation of the orbital plane symmetry.

    Parameters
    ----------
    data : dict
        Fitting data (output by analysis.py).

    Returns
    -------
    final_asymmetry_S2 : float
        Asymmetry measurement over the two-sphere.
    final_asymmetry_S2_std : float
        Standard deviation of asymmetry measurement over the two-sphere.

    """
    asymms = []

    power = 0
    power_std = 0
    asymmetry_S2 = 0
    asymmetry_S2_std = 0
    for L in range(2, 3 + 1):
        for M in range(-L, L + 1):
            sign_M = np.sign(M)
            if sign_M == 0:
                sign_M = 1
            # positive M
            A_mode_p_pro = (
                data[str((L, M, 0, sign_M)).replace(" ", "")]["A"][0]
                + 1j * data[str((L, M, 0, sign_M)).replace(" ", "")]["A"][1]
            )
            A_mode_p_pro_std_re = data[str((L, M, 0, sign_M)).replace(" ", "")][
                "A_std"
            ][0]
            A_mode_p_pro_std_im = data[str((L, M, 0, sign_M)).replace(" ", "")][
                "A_std"
            ][1]

            A_mode_p_retro = (
                data[str((L, M, 0, -sign_M)).replace(" ", "")]["A"][0]
                + 1j * data[str((L, M, 0, -sign_M)).replace(" ", "")]["A"][1]
            )
            A_mode_p_retro_std_re = data[str((L, M, 0, -sign_M)).replace(" ", "")][
                "A_std"
            ][0]
            A_mode_p_retro_std_im = data[str((L, M, 0, -sign_M)).replace(" ", "")][
                "A_std"
            ][1]

            A_mode_p = A_mode_p_pro + A_mode_p_retro
            A_mode_p_std_re = np.sqrt(
                (A_mode_p_pro_std_re**2 + A_mode_p_retro_std_re**2)
            )
            A_mode_p_std_im = np.sqrt(
                (A_mode_p_pro_std_im**2 + A_mode_p_retro_std_im**2)
            )

            A_mode_p_std = np.sqrt(
                A_mode_p_pro_std_re**2 + A_mode_p_retro_std_re**2
            )

            # negative M
            A_mode_n_pro = (
                data[str((L, -M, 0, -sign_M)).replace(" ", "")]["A"][0]
                + 1j * data[str((L, -M, 0, -sign_M)).replace(" ", "")]["A"][1]
            )
            A_mode_n_pro_std_re = data[str((L, -M, 0, -sign_M)).replace(" ", "")][
                "A_std"
            ][0]
            A_mode_n_pro_std_im = data[str((L, -M, 0, -sign_M)).replace(" ", "")][
                "A_std"
            ][1]

            A_mode_n_retro = (
                data[str((L, -M, 0, sign_M)).replace(" ", "")]["A"][0]
                + 1j * data[str((L, -M, 0, sign_M)).replace(" ", "")]["A"][1]
            )
            A_mode_n_retro_std_re = data[str((L, -M, 0, sign_M)).replace(" ", "")][
                "A_std"
            ][0]
            A_mode_n_retro_std_im = data[str((L, -M, 0, sign_M)).replace(" ", "")][
                "A_std"
            ][1]

            A_mode_n = A_mode_n_pro + A_mode_n_retro
            A_mode_n_std_re = np.sqrt(
                (A_mode_n_pro_std_re**2 + A_mode_n_retro_std_re**2)
            )
            A_mode_n_std_im = np.sqrt(
                (A_mode_n_pro_std_im**2 + A_mode_n_retro_std_im**2)
            )

            A_mode_n_std = np.sqrt(A_mode_n_std_re**2 + A_mode_n_std_im**2)

            asymm = abs(A_mode_p - (-1) ** L * np.conjugate(A_mode_n))
            asymm_std = np.sqrt(A_mode_p_std**2 + A_mode_n_std**2)

            power += abs(A_mode_p) ** 2
            power_std += abs(A_mode_p) ** 2 * A_mode_p_std**2

            asymmetry_S2 += asymm**2
            asymmetry_S2_std += asymm**2 * asymm_std**2

    power = np.sqrt(power)
    power_std = 1.0 / power * np.sqrt(power_std)

    asymmetry_S2 = np.sqrt(asymmetry_S2)
    asymmetr_S2_std = 1.0 / asymmetry_S2 * np.sqrt(asymmetry_S2_std)

    final_asymmetry_S2 = np.sqrt(asymmetry_S2**2 / (4.0 * power**2))
    final_asymmetry_S2_std = final_asymmetry_S2 * np.sqrt(
        (asymmetry_S2_std / asymmetry_S2) ** 2 + (power_std / power) ** 2
    )

    return final_asymmetry_S2, final_asymmetry_S2_std


def main():
    # Load data from QNM fits
    with open("QNM_results.json") as input_file:
        data = json.load(input_file)

    # Construct relevant arrays for ratios, parameters, etc.
    qs = []
    chi_fs = []

    thetas = []
    kick_angles = []
    kick_rapidities = []

    errors = []
    mismatches = []
    t0s = []
    CVs = []

    OShaughnessy_mismatches = []

    ratios_L2M2 = []

    ratios_L2M1 = []
    ratios_L2M1_pro_retro = []
    ratios_L2M1_mirror = []
    ratios_L2M1_pro_retro_mirror = []

    ratios_L2M0 = []
    ratios_L2M0_retro = []
    ratios_L2M0_pro_retro = []
    ratios_L2M0_mirror = []
    ratios_L2M0_pro_retro_mirror = []

    pro_retro_ratios_L2M2 = []
    pro_retro_ratios_L2M1 = []

    mirror_mode_ratios = []
    mirror_mode_ratio_errors = []

    asymms = []

    ratios_L3M2 = []
    ratios_L3M1 = []
    ratios_L3M0 = []

    for i, simulation in enumerate(data):
        q = data[simulation]["q"]
        qs.append(q)

        chi_fs.append(data[simulation]["chi_f"])

        thetas.append(data[simulation]["theta"])

        kick_angles.append(data[simulation]["kick theta"])
        kick_rapidities.append(data[simulation]["kick rapidity"])

        errors.append(data[simulation]["error"])
        mismatches.append(data[simulation]["mismatch"])
        t0s.append(data[simulation]["best t0"])
        CVs.append(data[simulation]["best CV"])

        OShaughnessy_mismatches.append(data[simulation]["R_error"])

        ratios_L2M2.append(
            compute_ratio(data[simulation], (2, -2, 0, -1), (2, 2, 0, 1))
        )

        ratios_L2M1.append(compute_ratio(data[simulation], (2, 1, 0, 1), (2, 2, 0, 1)))
        ratios_L2M1_pro_retro.append(
            compute_ratio(
                data[simulation],
                (2, 1, 0, 1),
                (2, 2, 0, 1),
                mode1_pro_retro=True,
                mode2_pro_retro=True,
            )
        )
        ratios_L2M1_mirror.append(
            compute_ratio(
                data[simulation],
                (2, 1, 0, 1),
                (2, 2, 0, 1),
                mode1_mirror=True,
                mode2_mirror=True,
            )
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
            )
        )

        ratios_L2M0.append(compute_ratio(data[simulation], (2, 0, 0, 1), (2, 2, 0, 1)))
        ratios_L2M0_retro.append(
            compute_ratio(data[simulation], (2, 0, 0, -1), (2, 2, 0, 1))
        )
        ratios_L2M0_pro_retro.append(
            compute_ratio(
                data[simulation],
                (2, 0, 0, 1),
                (2, 2, 0, 1),
                mode1_pro_retro=True,
                mode2_pro_retro=True,
            )
        )
        ratios_L2M0_mirror.append(
            compute_ratio(
                data[simulation],
                (2, 0, 0, 1),
                (2, 2, 0, 1),
                mode1_mirror=False,
                mode2_mirror=True,
            )
        )
        ratios_L2M0_pro_retro_mirror.append(
            compute_ratio(
                data[simulation],
                (2, 0, 0, 1),
                (2, 2, 0, 1),
                mode1_pro_retro=True,
                mode2_pro_retro=True,
                mode1_mirror=True,
                mode2_mirror=True,
            )
        )

        pro_retro_ratios_L2M2.append(
            compute_ratio(data[simulation], (2, -2, 0, 1), (2, 2, 0, 1))
        )
        pro_retro_ratios_L2M1.append(
            compute_ratio(data[simulation], (2, 1, 0, -1), (2, 1, 0, 1))
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

        ratios_L3M2.append(
            compute_ratio(
                data[simulation],
                (3, 2, 0, 1),
                (3, 3, 0, 1),
            )
        )
        ratios_L3M1.append(
            compute_ratio(
                data[simulation],
                (3, 1, 0, 1),
                (3, 3, 0, 1),
            )
        )
        ratios_L3M0.append(
            compute_ratio(
                data[simulation],
                (3, 0, 0, 1),
                (3, 3, 0, 1),
            )
        )

    qs = np.array(qs)

    thetas = np.array(thetas)
    kick_angles = np.array(kick_angles)
    kick_rapidities = np.array(kick_rapidities)

    errors = np.array(errors)
    mismatches = np.array(mismatches)
    t0s = np.array(t0s)
    CVs = np.array(CVs)

    ratios_L2M2 = np.array(ratios_L2M2)

    ratios_L2M1 = np.array(ratios_L2M1)
    ratios_L2M1_pro_retro = np.array(ratios_L2M1_pro_retro)
    ratios_L2M1_mirror = np.array(ratios_L2M1_mirror)
    ratios_L2M1_pro_retro_mirror = np.array(ratios_L2M1_pro_retro_mirror)

    ratios_L2M0 = np.array(ratios_L2M0)
    ratios_L2M0_retro = np.array(ratios_L2M0_retro)
    ratios_L2M0_pro_retro = np.array(ratios_L2M0_pro_retro)
    ratios_L2M0_mirror = np.array(ratios_L2M0_mirror)
    ratios_L2M0_pro_retro_mirror = np.array(ratios_L2M0_pro_retro_mirror)

    pro_retro_ratios_L2M2 = np.array(pro_retro_ratios_L2M2)
    pro_retro_ratios_L2M1 = np.array(pro_retro_ratios_L2M1)

    mirror_mode_ratios = np.array(mirror_mode_ratios)
    mirror_mode_ratio_errors = np.array(mirror_mode_ratio_errors)

    asymms = np.array(asymms)

    ratios_L3M2 = np.array(ratios_L3M2)
    ratios_L3M1 = np.array(ratios_L3M1)
    ratios_L3M0 = np.array(ratios_L3M0)

    create_L2M1_and_L2M0_figure(
        qs, thetas, ratios_L2M1_pro_retro_mirror, ratios_L2M0_pro_retro_mirror
    )

    create_kick_velocity_figure(thetas, ratios_L2M0, pro_retro_ratios_L2M2, kick_angles)

    create_mode_asymmetries_figure(
        list(data.keys()), mirror_mode_ratios, mirror_mode_ratio_errors, N_systems=6
    )

    create_QNM_fit_error_figure(thetas, errors, chi_fs)

    create_parity_breaking_figure(thetas, asymms, kick_rapidities)

    create_higher_harmonics_vs_prediction_figure(
        thetas, ratios_L3M2, ratios_L3M1, ratios_L3M0, kick_angles
    )

    create_OShaughnessy_figure(thetas, asymms[:, 0], qs, OShaughnessy_mismatches)


if __name__ == "__main__":
    main()
