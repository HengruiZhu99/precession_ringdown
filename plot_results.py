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


def compute_rotation_factor(input_mode, output_mode, theta):
    v = quaternion.quaternion(*np.array([np.sin(theta), 0, np.cos(theta)])).normalized()
    R = (1 - v * quaternion.z).normalized()

    return sf.Wigner_D_element(R, input_mode[0], output_mode[1], input_mode[1])


# Figure 1
def create_Figure1(qs, thetas, ratios, ratios_pro_retro, inset_fig=True):
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

    result = axis[1][0].scatter(thetas, qs, s=8, c=np.log10(ratios), cmap="viridis")

    axis[1][0].set_xlabel(r"misalignment angle $\theta$", fontsize=12)
    axis[1][0].set_ylabel(r"mass ratio $q$", fontsize=12)

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
        plt.setp(newax.spines.values(), color = 'lightgrey')

    # panel B

    result = axis[1][1].scatter(thetas, ratios_pro_retro, c=qs, s=8, cmap="magma")

    angles = np.linspace(0, np.pi - 0.01, 100)
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
        label=r"$\cfrac{\phantom{}_{-2}Y_{(2,+1)}^{R,\pm}(\theta)}{\phantom{}_{-2}Y_{(2,+2)}^{R,\pm}(\theta)}$",
    )

    xlim = axis[1][1].get_xlim()
    axis[1][1].plot(
        np.arange(-np.pi, 2 * np.pi, 0.01),
        np.ones_like(np.arange(-np.pi, 2 * np.pi, 0.01)),
        ls="--",
        color=colors[0],
        lw=1.4,
    )
    axis[1][1].set_xlim(xlim)

    x = 3.05
    y = 6.0e-1
    ell_offset = ScaledTranslation(x, y, axis[1][1].transScale)
    ell_tform = ell_offset + axis[1][1].transLimits + axis[1][1].transAxes
    axis[1][1].add_patch(
        Ellipse(
            xy=(0, 0),
            width=0.46,
            height=0.52,
            color=colors[0],
            fill=False,
            lw=1,
            zorder=10,
            transform=ell_tform,
        )
    )

    axis[1][1].set_yscale("log")
    axis[1][1].set_xlim(0 - 0.2, np.pi + 0.2)

    axis[1][1].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    axis[1][1].set_xlabel(r"misalignment angle $\theta$", fontsize=12)

    axis[1][1].set_ylabel(r"$A_{(\pm,2,\pm1,0)}/A_{(\pm,2,\pm2,0)}$", fontsize=12)

    axis[1][1].legend(loc="lower right", frameon=True, framealpha=1, fontsize=12)

    c2 = fig.colorbar(result, cax=axis[0][1], orientation="horizontal")

    c2.ax.xaxis.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])
    c2.ax.xaxis.set_ticks_position("top")
    c2.ax.set_xlabel(r"mass ratio $q$", fontsize=12, labelpad=-36)

    plt.savefig("CCEFigures/Figure1.pdf", bbox_inches="tight")


# Figure 2
def create_Figure2(
    thetas, ratios_L2M1, pro_retro_ratios_L2M2, kick_angles, name_suffix=""
):
    fig = plt.figure(
        figsize=(onecol_w_in, onecol_w_in * 1.4),
    )
    axis = fig.subplot_mosaic(
        mosaic=[["A panel", "C panel"], ["B panel", "C panel"]], width_ratios=[1, 0.05]
    )
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    result = axis["A panel"].scatter(
        thetas, ratios_L2M1, c=kick_angles, s=8, cmap="coolwarm"
    )

    axis["A panel"].set_yscale("log")
    axis["A panel"].set_xlim(0 - 0.2, np.pi + 0.2)
    axis["A panel"].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    axis["A panel"].set_xticklabels([])

    angles = np.linspace(0, np.pi - 0.01, 100)
    rotation_factors = np.array(
        [
            abs(compute_rotation_factor((2, 2), (2, 1), angle))
            / abs(compute_rotation_factor((2, 2), (2, 2), angle))
            for angle in angles
        ]
    )

    # Change label to just be some WignerD notation that we define in methods?
    axis["A panel"].plot(
        angles,
        rotation_factors,
        label=r"$\cfrac{\phantom{}_{-2}Y_{(2,+1)}^{R}(\theta)}{\phantom{}_{-2}Y_{(2,+2)}^{R}(\theta)}$",
    )

    axis["A panel"].set_ylabel(r"$A_{(+,2,1,0)}/A_{(+,2,2,0)}$", fontsize=12)

    axis["A panel"].legend(loc="lower right", frameon=True, framealpha=1, fontsize=12)
    axis["A panel"].grid(alpha=0.2)

    axis["A panel"].set_ylim(bottom=6e-3, top=2e1)

    axis["B panel"].scatter(
        thetas, pro_retro_ratios_L2M2, c=kick_angles, s=8, cmap="coolwarm"
    )

    axis["B panel"].set_yscale("log")
    axis["B panel"].set_xlim(0 - 0.2, np.pi + 0.2)
    axis["B panel"].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    angles = np.linspace(0, np.pi - 0.01, 100)
    rotation_factors = np.array(
        [
            abs(compute_rotation_factor((2, 2), (2, -2), angle))
            / abs(compute_rotation_factor((2, 2), (2, 2), angle))
            for angle in angles
        ]
    )

    axis["B panel"].plot(
        angles,
        rotation_factors,
        label=r"$\cfrac{\phantom{}_{-2}Y_{(2,-2)}^{R}(\theta)}{\phantom{}_{-2}Y_{(2,+2)}^{R}(\theta)}$",
    )

    axis["B panel"].set_xlabel(r"misalignment angle $\theta$", fontsize=12)
    axis["B panel"].set_ylabel(r"$A_{(-,2,2,0)}/A_{(+,2,2,0)}$", fontsize=12)

    axis["B panel"].legend(loc="lower right", frameon=True, framealpha=1, fontsize=12)
    axis["B panel"].grid(alpha=0.2)

    axis["B panel"].set_ylim(bottom=6e-6, top=2e3)

    c = fig.colorbar(result, cax=axis["C panel"], orientation="vertical", pad=0)

    c.ax.set_ylabel(r"kick angle $\phi$", rotation=270, labelpad=15, fontsize=12)

    plt.savefig(f"CCEFigures/Figure2.pdf", bbox_inches="tight")


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


def main():
    # Load data from QNM fits
    with open("QNM_results.json") as input_file:
        data = json.load(input_file)

    # Construct relevant arrays for ratios, parameters, etc.
    qs = []
    chi_ps = []
    kick_angles = []
    thetas = []
    errors = []
    mismatches = []
    t0s = []
    CVs = []
    ratios_L2M1 = []
    ratios_L2M0 = []
    ratios_L2M1_pro_retro = []
    ratios_L2M0_pro_retro = []
    ratios_L2M1_mirror = []
    ratios_L2M0_mirror = []
    ratios_L2M1_pro_retro_mirror = []
    ratios_L2M0_pro_retro_mirror = []
    pro_retro_ratios_L2M2 = []
    pro_retro_ratios_L2M1 = []
    pro_retro_ratios_L2M0 = []

    for simulation in data:
        q = data[simulation]["q"]
        qs.append(q)

        chi1 = data[simulation]["chi1"]
        chi2 = data[simulation]["chi2"]

        # Patricia Schmidt definition
        sin_theta1 = np.sin(np.arccos(np.dot(chi1, [0, 0, 1]) / np.linalg.norm(chi1)))
        sin_theta2 = np.sin(np.arccos(np.dot(chi2, [0, 0, 1]) / np.linalg.norm(chi2)))
        chi_p = max(np.linalg.norm(chi1[:2]), np.linalg.norm(chi2[:2]))
        chi_ps.append(chi_p)

        v_f = data[simulation]["delta_v"]
        kick_angles.append(np.arccos(np.dot(v_f / np.linalg.norm(v_f), [0, 0, 1])))

        thetas.append(data[simulation]["theta_flux"])
        errors.append(data[simulation]["error"])
        mismatches.append(data[simulation]["mismatch"])
        t0s.append(data[simulation]["best t0"])
        CVs.append(data[simulation]["best CV"])

        ratios_L2M1.append(
            compute_ratio(data[simulation], (2, 1, 0, 1), (2, 2, 0, 1))[0]
        )
        ratios_L2M0.append(
            compute_ratio(data[simulation], (2, 0, 0, 1), (2, 2, 0, 1))[0]
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
        ratios_L2M0_pro_retro.append(
            compute_ratio(
                data[simulation],
                (2, 0, 0, 1),
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
        ratios_L2M0_mirror.append(
            compute_ratio(
                data[simulation],
                (2, 0, 0, 1),
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
        ratios_L2M0_pro_retro_mirror.append(
            compute_ratio(
                data[simulation],
                (2, 0, 0, 1),
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
        pro_retro_ratios_L2M0.append(
            compute_ratio(data[simulation], (2, 0, 0, -1), (2, 0, 0, 1))[0]
        )

    qs = np.array(qs)
    chi_ps = np.array(chi_ps)
    kick_angles = np.array(kick_angles)
    thetas = np.array(thetas)
    errors = np.array(errors)
    mismatches = np.array(mismatches)
    t0s = np.array(t0s)
    CVs = np.array(CVs)
    ratios_L2M1 = np.array(ratios_L2M1)
    ratios_L2M0 = np.array(ratios_L2M0)
    ratios_L2M1_pro_retro = np.array(ratios_L2M1_pro_retro)
    ratios_L2M0_pro_retro = np.array(ratios_L2M0_pro_retro)
    ratios_L2M1_mirror = np.array(ratios_L2M1_mirror)
    ratios_L2M0_mirror = np.array(ratios_L2M0_mirror)
    ratios_L2M1_pro_retro_mirror = np.array(ratios_L2M1_pro_retro_mirror)
    ratios_L2M0_pro_retro_mirror = np.array(ratios_L2M0_pro_retro_mirror)
    pro_retro_ratios_L2M2 = np.array(pro_retro_ratios_L2M2)
    pro_retro_ratios_L2M1 = np.array(pro_retro_ratios_L2M1)
    pro_retro_ratios_L2M0 = np.array(pro_retro_ratios_L2M0)

    create_Figure1(qs, thetas, ratios_L2M1, ratios_L2M1_pro_retro)

    create_Figure2(thetas, ratios_L2M1, pro_retro_ratios_L2M2, kick_angles)


if __name__ == "__main__":
    main()
