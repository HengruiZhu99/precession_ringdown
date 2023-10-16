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


# Figure 1 panel A
def create_parameters_plot(qs, thetas, ratios, inset_fig=True):
    fig, axis = plt.subplots(
        1,
        2,
        figsize=(onecol_w_in, onecol_w_in * 1.0),
        sharex=False,
        sharey=False,
        width_ratios=[1, 0.05],
    )
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    result = axis[0].scatter(thetas, qs, s=4, c=np.log10(ratios), cmap="rainbow")

    axis[0].set_xlabel(
        r"$\Delta\theta(\dot{J}^{t=\mathrm{peak}},\chi_{f}^{t=+\infty})$"
    )
    axis[0].set_ylabel(r"$q$")

    c = fig.colorbar(result, cax=axis[1], orientation="vertical", pad=2.5)

    c.ax.get_yaxis().labelpad = 15
    c.ax.set_ylabel(
        r"$\log_{10}\left(A_{(+,2,1,0)}/A_{(+,2,2,0)}\right)$", rotation=270, fontsize=6
    )

    if inset_fig:
        im = plt.imread('CCEFigures/SpinMisalignmentCartoon.jpeg')
        newax = fig.add_axes([0.48,0.06,0.315,0.315], anchor='NE', zorder=1)
        newax.imshow(im)
        newax.get_xaxis().set_ticks([])
        newax.get_yaxis().set_ticks([])

    plt.savefig("CCEFigures/parameters.pdf", bbox_inches="tight")

# Figure 1 panel B
def create_mode_mixings_plot(thetas, ratios, z_axis, name_suffix=""):
    fig, axis = plt.subplots(
        1,
        2,
        figsize=(onecol_w_in, onecol_w_in * 1.0),
        sharex=False,
        sharey=False,
        width_ratios=[1, 0.05],
    )
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    result = axis[0].scatter(thetas, ratios, c=z_axis, s=4, cmap="rainbow")

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

    # Change label to just be some WignerD notation that we define in methods?
    axis[0].plot(
        angles,
        rotation_factors,
        label=r"$\cfrac{\langle\phantom{}_{-2}Y_{(2,\pm1)}(\mathbf{R}),\phantom{}_{-2}Y_{(2,+2)}\rangle}{\langle\phantom{}_{-2}Y_{(2,\pm2)}(\mathbf{R}),\phantom{}_{-2}Y_{(2,+2)}\rangle}$",
    )

    x = 3.05
    y = 6.0e-1
    ell_offset = ScaledTranslation(x, y, axis[0].transScale)
    ell_tform = ell_offset + axis[0].transLimits + axis[0].transAxes
    axis[0].add_patch(
        Ellipse(
            xy=(0, 0),
            width=0.4,
            height=0.68,
            color=colors[0],
            fill=False,
            lw=1,
            zorder=10,
            transform=ell_tform,
        )
    )

    axis[0].set_yscale("log")
    axis[0].set_ylim(top=6e1)
    axis[0].set_xlim(0 - 0.2, np.pi + 0.2)

    axis[0].set_xlabel(
        r"$\Delta\theta(\dot{J}^{t=\mathrm{peak}},\chi_{f}^{t=+\infty})$"
    )
    
    if not "mirror" in name_suffix:
        axis[0].set_ylabel(r"$A_{(\pm,2,1,0)}/A_{(\pm,2,2,0)}$")
    else:
        axis[0].set_ylabel(r"$A_{(\pm,2,\pm1,0)}/A_{(\pm,2,\pm2,0)}$")

    axis[0].legend(loc="upper left", frameon=True, framealpha=1)
    axis[0].grid()

    c = fig.colorbar(result, cax=axis[1], orientation="vertical", pad=0)

    c.ax.get_yaxis().labelpad = 15
    if 'kick_angles' in name_suffix:
        c.ax.set_ylabel(r"$\Delta\theta(v^{t=-\infty},v^{t=+\infty})$", rotation=270)
    else:
        c.ax.set_ylabel(r"$q$", rotation=270)

    plt.savefig(f"CCEFigures/mode_mixings{name_suffix}.pdf", bbox_inches="tight")


# Figure 2
def create_asymmetric_mode_mixings_plot(
    thetas, ratios1, ratios2, kick_angles, name_suffix=""
):
    fig, axis = plt.subplots(
        1,
        4,
        figsize=(twocol_w_in, twocol_w_in * 0.45),
        sharex=False,
        sharey=False,
        width_ratios=[1, 0.28, 1, 0.05],
    )
    plt.subplots_adjust(wspace=0.02)
    axis[1].set_visible(False)

    result = axis[0].scatter(thetas, ratios1, c=kick_angles, s=4, cmap="rainbow")

    axis[0].set_yscale("log")
    axis[0].set_xlim(0 - 0.2, np.pi + 0.2)
    ylim = axis[0].get_ylim()

    angles = np.linspace(0, np.pi - 0.01, 100)
    rotation_factors = np.array(
        [
            abs(compute_rotation_factor((2, 2), (2, 1), angle))
            / abs(compute_rotation_factor((2, 2), (2, 2), angle))
            for angle in angles
        ]
    )

    # Change label to just be some WignerD notation that we define in methods?
    axis[0].plot(
        angles,
        rotation_factors,
        label=r"$\cfrac{\langle\phantom{}_{-2}Y_{(2,+1)}(\mathbf{R}),\phantom{}_{-2}Y_{(2,+2)}\rangle}{\langle\phantom{}_{-2}Y_{(2,+2)}(\mathbf{R}),\phantom{}_{-2}Y_{(2,+2)}\rangle}$",
    )

    axis[0].set_ylim(bottom=1e-5, top=4e2)

    axis[0].set_xlabel(
        r"$\Delta\theta(\dot{J}^{t=\mathrm{peak}},\chi_{f}^{t=+\infty})$"
    )
    axis[0].set_ylabel(r"$A_{(+,2,1,0)}/A_{(+,2,2,0)}$")

    axis[0].legend(loc="upper left", frameon=True, framealpha=1)
    axis[0].grid()

    axis[2].scatter(thetas, ratios2, c=kick_angles, s=4, cmap="rainbow")

    axis[2].set_yscale("log")
    axis[2].set_xlim(0 - 0.2, np.pi + 0.2)
    ylim = axis[2].get_ylim()

    angles = np.linspace(0, np.pi - 0.01, 100)
    rotation_factors = np.array(
        [
            abs(compute_rotation_factor((2, 2), (2, -2), angle))
            / abs(compute_rotation_factor((2, 2), (2, 2), angle))
            for angle in angles
        ]
    )

    # Change label to just be some WignerD notation that we define in methods?
    axis[2].plot(
        angles,
        rotation_factors,
        label=r"$\cfrac{\langle\phantom{}_{-2}Y_{(2,-2)}(\mathbf{R}),\phantom{}_{-2}Y_{(2,+2)}\rangle}{\langle\phantom{}_{-2}Y_{(2,+2)}(\mathbf{R}),\phantom{}_{-2}Y_{(2,+2)}\rangle}$",
    )

    axis[2].set_ylim(bottom=1e-5, top=4e2)

    axis[2].set_xlabel(
        r"$\Delta\theta(\dot{J}^{t=\mathrm{peak}},\chi_{f}^{t=+\infty})$"
    )
    axis[2].set_ylabel(r"$A_{(-,2,2,0)}/A_{(+,2,2,0)}$")

    axis[2].legend(loc="upper left", frameon=True, framealpha=1)
    axis[2].grid()

    c = fig.colorbar(result, cax=axis[3], orientation="vertical", pad=0)

    c.ax.get_yaxis().labelpad = 15
    c.ax.set_ylabel(r"$\Delta\theta(v^{t=-\infty},v^{t=+\infty})$", rotation=270)

    plt.savefig(f"CCEFigures/asymmetric_mixings{name_suffix}.pdf", bbox_inches="tight")

def create_fit_plot(thetas, vertical_axis, qs, name_suffix='errors'):
    fig, axis = plt.subplots(
        1,
        2,
        figsize=(onecol_w_in, onecol_w_in * 0.65),
        sharex=False,
        sharey=False,
        width_ratios=[1, 0.05],
    )
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    result = axis[0].scatter(thetas, vertical_axis, s=4, c=qs, cmap="rainbow")
    
    axis[0].set_yscale('log')
    
    axis[0].set_xlabel(
        r"$\Delta\theta(\dot{J}^{t=\mathrm{peak}},\chi_{f}^{t=+\infty})$"
    )
    if 'error' in name_suffix:
        axis[0].set_ylabel(r"relative error")
    elif 'mismatch' in name_suffix:
        axis[0].set_ylabel(r"$\mathrm{Mismatch}\,\mathcal{M}$")
    elif 't0' in name_suffix:
        axis[0].set_ylabel(r"$t_{0}$")
    elif 'CV' in name_suffix:
        axis[0].set_ylabel(r"Coefficient of Variation")

    c = fig.colorbar(result, cax=axis[1], orientation="vertical", pad=2.5)

    c.ax.get_yaxis().labelpad = 15
    c.ax.set_ylabel(
        r"$q$", rotation=270, fontsize=6
    )

    plt.savefig(f"CCEFigures/{name_suffix}.pdf", bbox_inches="tight")
    

def create_kick_plot(thetas, kick_angles, qs):
    fig, axis = plt.subplots(
        1,
        2,
        figsize=(onecol_w_in, onecol_w_in * 0.65),
        sharex=False,
        sharey=False,
        width_ratios=[1, 0.05],
    )
    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    result = axis[0].scatter(thetas, kick_angles, s=4, c=qs, cmap="rainbow")

    axis[0].set_xlabel(
        r"$\Delta\theta(\dot{J}^{t=\mathrm{peak}},\chi_{f}^{t=+\infty})$"
    )
    axis[0].set_ylabel(r"$\Delta\theta(v)$")

    c = fig.colorbar(result, cax=axis[1], orientation="vertical", pad=2.5)

    c.ax.get_yaxis().labelpad = 15
    c.ax.set_ylabel(
        r"$q$", rotation=270, fontsize=6
    )

    plt.savefig("CCEFigures/kick.pdf", bbox_inches="tight")

def compute_mode_amplitude(data, mode, pro_retro=False, mirror=False):
    L, M, N, S = mode
    if pro_retro and not mirror:
        # prograde
        A_mode_pro = (
            data[str((L, M, N, S)).replace(" ", "")]["A"][
                0
            ]
            + 1j
            * data[str((L, M, N, S)).replace(" ", "")][
                "A"
            ][1]
        )
        A_mode_pro_std_re = data[
            str((L, M, N, S)).replace(" ", "")
        ]["A_std"][0]
        A_mode_pro_std_im = data[
            str((L, M, N, S)).replace(" ", "")
        ]["A_std"][1]

        # retrograde
        A_mode_retro = (
            data[str((L, M, N, -S)).replace(" ", "")][
                "A"
            ][0]
            + 1j
            * data[str((L, M, N, -S)).replace(" ", "")][
                "A"
            ][1]
        )
        A_mode_retro_std_re = data[
            str((L, M, N, -S)).replace(" ", "")
        ]["A_std"][0]
        A_mode_retro_std_im = data[
            str((L, M, N, -S)).replace(" ", "")
        ]["A_std"][1]

        return abs(A_mode_pro + A_mode_retro), np.sqrt(
            (A_mode_pro_std_re**2 + A_mode_retro_std_re**2)
            + (A_mode_pro_std_im**2 + A_mode_retro_std_im**2)
        )
    elif not pro_retro and mirror:
        # positive M
        A_mode_pM = abs(
            data[str((L, M, N, S)).replace(" ", "")]["A"][
                0
            ]
            + 1j
            * data[str((L, M, N, S)).replace(" ", "")][
                "A"
            ][1]
        )
        A_mode_pM_std = np.sqrt(
            data[str((L, M, N, S)).replace(" ", "")][
                "A_std"
            ][0]
            ** 2
            + data[str((L, M, N, S)).replace(" ", "")][
                "A_std"
            ][1]
            ** 2
        )

        A_mode_nM = abs(
            data[str((L, -M, N, -S)).replace(" ", "")][
                "A"
            ][0]
            + 1j
            * data[str((L, -M, N, -S)).replace(" ", "")][
                "A"
            ][1]
        )
        A_mode_nM_std = np.sqrt(
            data[str((L, -M, N, -S)).replace(" ", "")][
                "A_std"
            ][0]
            ** 2
            + data[str((L, -M, N, -S)).replace(" ", "")][
                "A_std"
            ][1]
            ** 2
        )

        Q_sum = np.sqrt(A_mode_pM**2 + A_mode_nM**2)

        return Q_sum, np.sqrt(
            (A_mode_pM * A_mode_pM_std**2 + A_mode_nM * A_mode_nM_std**2) / Q_sum
        )
    elif pro_retro and mirror:
        # prograde positive M
        A_mode_pro_pM = (
            data[str((L, M, N, S)).replace(" ", "")]["A"][
                0
            ]
            + 1j
            * data[str((L, M, N, S)).replace(" ", "")][
                "A"
            ][1]
        )
        A_mode_pro_pM_std_re = data[
            str((L, M, N, S)).replace(" ", "")
        ]["A_std"][0]
        A_mode_pro_pM_std_im = data[
            str((L, M, N, S)).replace(" ", "")
        ]["A_std"][1]

        # retrograde positive M
        A_mode_retro_pM = (
            data[str((L, M, N, -S)).replace(" ", "")][
                "A"
            ][0]
            + 1j
            * data[str((L, M, N, -S)).replace(" ", "")][
                "A"
            ][1]
        )
        A_mode_retro_pM_std_re = data[
            str((L, M, N, -S)).replace(" ", "")
        ]["A_std"][0]
        A_mode_retro_pM_std_im = data[
            str((L, M, N, -S)).replace(" ", "")
        ]["A_std"][1]

        # prograde negative M
        A_mode_pro_nM = (
            data[str((L, -M, N, -S)).replace(" ", "")][
                "A"
            ][0]
            + 1j
            * data[str((L, -M, N, -S)).replace(" ", "")][
                "A"
            ][1]
        )
        A_mode_pro_nM_std_re = data[
            str((L, -M, N, -S)).replace(" ", "")
        ]["A_std"][0]
        A_mode_pro_nM_std_im = data[
            str((L, -M, N, -S)).replace(" ", "")
        ]["A_std"][1]

        # retrograde negative M
        A_mode_retro_nM = (
            data[str((L, -M, N, S)).replace(" ", "")][
                "A"
            ][0]
            + 1j
            * data[str((L, -M, N, S)).replace(" ", "")][
                "A"
            ][1]
        )
        A_mode_retro_nM_std_re = data[
            str((L, -M, N, S)).replace(" ", "")
        ]["A_std"][0]
        A_mode_retro_nM_std_im = data[
            str((L, -M, N, S)).replace(" ", "")
        ]["A_std"][1]

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
            data[str((L, M, N, S)).replace(" ", "")]["A"][
                0
            ]
            + 1j
            * data[str((L, M, N, S)).replace(" ", "")][
                "A"
            ][1]
        )
        A_mode_std = np.sqrt(
            data[str((L, M, N, S)).replace(" ", "")][
                "A_std"
            ][0]
            ** 2
            + data[str((L, M, N, S)).replace(" ", "")][
                "A_std"
            ][1]
            ** 2
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
        errors.append(data[simulation]['error'])
        mismatches.append(data[simulation]['mismatch'])
        t0s.append(data[simulation]['best t0'])
        CVs.append(data[simulation]['best CV'])
        
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

    create_parameters_plot(qs, thetas, ratios_L2M1)
    create_mode_mixings_plot(
        thetas, ratios_L2M1_pro_retro, kick_angles, "_L2M1_pro_retro_kick_angles"
    )
    create_mode_mixings_plot(
        thetas, ratios_L2M1_pro_retro_mirror, qs, "_L2M1_pro_retro_mirror_mass_ratio"
    )
    create_mode_mixings_plot(
        thetas, ratios_L2M1_pro_retro_mirror, kick_angles, "_L2M1_pro_retro_mirror_kick_angles"
    )
    create_asymmetric_mode_mixings_plot(thetas, ratios_L2M1, pro_retro_ratios_L2M2, kick_angles)
    create_fit_plot(thetas, errors, qs, 'errors')

    create_kick_plot(thetas, kick_angles, qs)

if __name__ == "__main__":
    main()
