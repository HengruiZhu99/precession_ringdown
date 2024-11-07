import os
import sxs
import scri
import numpy as np
import spherical_functions as sf

import json

from ringdown import *
import quaternion
from quaternion.calculus import derivative
from quaternion.calculus import indefinite_integral as integrate
from scri.mode_calculations import LLDominantEigenvector

from scri.asymptotic_bondi_data.map_to_superrest_frame import MT_to_WM, WM_to_MT


def read_waveform(
    simulation, waveform_dir="bondi_cce_superrest_iplus", suffix="_superrest"
):
    """Load a waveform or an AsymptoticBondiData object for a certain simulation.

    Parameters
    ----------
    simulation : string
    waveform_dir : string, optional
    suffix: string, optional

    Returns
    -------
    abd : AsymptoticBondiData
        Data object containing the strain and Weyl scalars.
    remnant_spin : float
    remnant_mass : float

    """
    Lev = "."
    radius = [
        x.split("_R")[-1][:4]
        for x in os.listdir(f"{simulation}/{Lev}/{waveform_dir}/")
        if "rhOverM_BondiCce"
    ][0]

    waveform_file = [
        x
        for x in os.listdir(f"{simulation}/{Lev}/{waveform_dir}/")
        if "rhOverM_BondiCce" in x and ".h5" in x
    ][0]

    if not os.path.exists(
        f"{simulation}/{Lev}/{waveform_dir}/{waveform_file[:-3]}.json"
    ):
        file_format = "SXS"
    else:
        file_format = "RPXM"

    abd = scri.SpEC.file_io.create_abd_from_h5(
        h=f"{simulation}/{Lev}/{waveform_dir}/rhOverM_BondiCce_R{radius}{suffix}.h5",
        Psi4=f"{simulation}/{Lev}/{waveform_dir}/rMPsi4_BondiCce_R{radius}{suffix}.h5",
        Psi3=f"{simulation}/{Lev}/{waveform_dir}/r2Psi3_BondiCce_R{radius}{suffix}.h5",
        Psi2=f"{simulation}/{Lev}/{waveform_dir}/r3Psi2OverM_BondiCce_R{radius}{suffix}.h5",
        Psi1=f"{simulation}/{Lev}/{waveform_dir}/r4Psi1OverM2_BondiCce_R{radius}{suffix}.h5",
        Psi0=f"{simulation}/{Lev}/{waveform_dir}/r5Psi0OverM3_BondiCce_R{radius}{suffix}.h5",
        file_format=file_format,
    )

    abd.t -= abd.t[np.argmax(MT_to_WM(2.0 * abd.sigma.bar.dot).norm())]
    abd = abd.interpolate(np.arange(-1000, min(abd.t[-1], 250), 0.1))

    remnant_spin = abd.bondi_dimensionless_spin()[-1]
    remnant_mass = abd.bondi_rest_mass()[-1]

    return abd, remnant_spin, remnant_mass


def compute_omega(modes, chi_f, M_f):
    """Compute the QNM frequency for a single QNM tuple or a combination of QNM tuples.

    Parameters
    ----------
    modes : tuple or list
        Tuple of (l, m, n, p) QNM indexes or list of tuple of QNM indexes.
    chi_f : float
        Remnant spin.
    M_f : float
        Remnant_mass.

    Returns
    -------
    omega: float

    """
    omega = 0
    if type(modes) == list:
        for mode in modes:
            omega += qnm_from_tuple(mode, chi_f, M_f)[0]
    else:
        omega = qnm_from_tuple(modes, chi_f, M_f)[0]

    return omega


def compute_change_in_flux(abd, t1=0, integrated=False):
    """Compute angle between angular momentum flux and remnant spin axis.

    Parameters
    ----------
    abd : AsymptoticBondiData
    t1 : float, optional
        Time at which to compute the angular momentum flux. [Default: 0].
    integrated : bool, optional
        If True, use the angular momentum, rather than the angular momentum flux. [Default: False].

    Returns
    -------
    misalignment_angle : float

    """
    h = MT_to_WM(2.0 * abd.sigma.bar)
    charge = abd.bondi_dimensionless_spin()[-1]

    J = h.angular_momentum_flux()

    if integrated:
        J = integrate(J, h.t)
        J -= J[-1] - np.array(charge) * np.linalg.norm(J[-1])

    if t1 is not None:
        J_t1 = J[np.argmin(abs(h.t - t1))]
    else:
        J_t1 = J

    J_t2 = charge

    if t1 is not None:
        cos_theta = np.dot(J_t1, J_t2) / (np.linalg.norm(J_t1) * np.linalg.norm(J_t2))
    else:
        cos_theta = np.dot(J_t1, J_t2) / (
            np.linalg.norm(J_t1, axis=1) * np.linalg.norm(J_t2)
        )

    misalignment_angle = np.arccos(cos_theta)

    return misalignment_angle


def h_to_Euler_angles(h, return_rotor=False):
    RoughDirection = np.array([0.0, 0.0, 1.0])
    RoughDirectionIndex = h.n_times // 8

    dpa = LLDominantEigenvector(
        h, RoughDirection=RoughDirection, RoughDirectionIndex=RoughDirectionIndex
    )
    R = np.array(
        [
            quaternion.quaternion.sqrt(
                -quaternion.quaternion(0, *q).normalized() * quaternion.z
            )
            for q in dpa
        ]
    )
    R = quaternion.minimal_rotation(R, h.t, iterations=3)

    if return_rotor:
        return R

    euler_angles = np.unwrap(quaternion.as_euler_angles(R), axis=0)

    return euler_angles


def compute_Euler_angle_error(h, chi_f, M_f):
    dJdt = h.angular_momentum_flux()[np.argmax(MT_to_WM(WM_to_MT(h).dot).norm())]
    theta = np.arccos(
        np.dot(dJdt, [0, 0, 1]) / (np.linalg.norm(dJdt) * np.linalg.norm([0, 0, 1]))
    )

    h = h[:, 2:4]

    euler_angles = h_to_Euler_angles(h)

    # times for determining late-time average
    t1 = 40
    t2 = 60
    idx1 = np.argmin(abs(h.t - t1))
    idx2 = np.argmin(abs(h.t - t2)) + 1

    if theta < np.pi / 2:
        omega22 = qnm_from_tuple((2, 2, 0, 1), chi_f, M_f)[0]
        omega21 = qnm_from_tuple((2, 1, 0, 1), chi_f, M_f)[0]
    else:
        omega22 = qnm_from_tuple((2, 2, 0, -1), chi_f, M_f)[0]
        omega21 = qnm_from_tuple((2, 1, 0, -1), chi_f, M_f)[0]

    alpha_PHM = (omega22 - omega21).real * h.t
    alpha_PHM += (
        np.mean(euler_angles[idx1:idx2, 0])
        - alpha_PHM[np.argmin(abs(h.t - (t1 + t2) / 2))]
    )

    beta0 = 2 * np.tan(np.mean(euler_angles[idx1:idx2, 1]) / 2)
    beta_PHM = -2 * np.arctan(
        beta0
        * np.exp(-omega21.imag * (h.t - (t1 + t2) / 2))
        / (2 * np.exp(-omega22.imag * (h.t - (t1 + t2) / 2)))
    ) + 2 * np.mean(euler_angles[idx1:idx2, 1])

    gamma_PHM = -integrate((omega22 - omega21).real * np.cos(beta_PHM), h.t)
    gamma_PHM += (
        np.mean(euler_angles[idx1:idx2, 2])
        - gamma_PHM[np.argmin(abs(h.t - (t1 + t2) / 2))]
    )

    q_NR = quaternion.from_euler_angles(
        np.array([euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]]).T
    )

    q_PHM = quaternion.from_euler_angles(np.array([alpha_PHM, beta_PHM, gamma_PHM]).T)

    h_coprec_NR = h.copy()
    h_coprec_NR = h_coprec_NR.rotate_decomposition_basis(q_NR)

    h_inertial_via_PHM = h_coprec_NR.copy()
    h_inertial_via_PHM = h_inertial_via_PHM.rotate_decomposition_basis([q.inverse() for q in q_PHM])

    def compute_mismatch(h1, h2):
        overlap = integrate(MT_to_WM(WM_to_MT(h1) * WM_to_MT(h2).bar).norm(), h1.t)[
            -1
        ].real
        norm1 = integrate(MT_to_WM(WM_to_MT(h1) * WM_to_MT(h1).bar).norm(), h1.t)[
            -1
        ].real
        norm2 = integrate(MT_to_WM(WM_to_MT(h2) * WM_to_MT(h2).bar).norm(), h1.t)[
            -1
        ].real

        return 1.0 - overlap / np.sqrt(norm1 * norm2)

    R_error = compute_mismatch(h, h_inertial_via_PHM)

    return R_error, theta


def fit_QNMs(h, chi_f, M_f, t0s, tf=100, ell_max=4, window_size=20):
    """Fit waveform with QNMs over a range of start times and return the amplitudes that are most stable over a 20M window.

    Parameters
    ----------
    h : WaveformModes
    chi_f: float
       Remnant spin.
    M_f:
       Remnant mass.
    t0s: list
       Start times to loop over.
    t_f: float, optional
       Final time to use in QNM fits. [Default: 100].
    ell_max: int, optional
       Maximum \ell to include in QNM fits. [Default: 4].
    window_size: float, optional
       Window size over which to check stability. [Default: 20].

    Returns
    -------
    QNM_As : dict
        Dictionary of QNM amplitudes and standard deviations and various fit statistics.

    """
    QNM_dict = {}

    count = 0
    for L in range(2, ell_max + 1):
        for M in range(-L, L + 1):
            for N in range(0, 1):
                for S in [-1, +1]:
                    QNM_dict[count] = {
                        "type": "QNM",
                        "mode": (L, M, N, S),
                        "omega": qnm_from_tuple((L, M, N, S), chi_f, M_f)[0],
                    }
                    count += 1

    As = []
    As_re = []
    As_im = []
    As_for_CV = []
    errors = []
    mismatches = []
    for t0 in t0s:
        times = (t0, tf)
        idx1 = np.argmin(abs(h.t - times[0]))
        idx2 = np.argmin(abs(h.t - times[1])) + 1
        h_ring = h.copy()[idx1:idx2:, 2 : 3 + 1]

        (h_QNM, h_ring, error, mismatch, QNM_fits) = fit_ringdown_waveform_LLSQ_S2(
            h_ring,
            [(L, M) for L in [2, 3] for M in range(-L, L + 1)],
            times,
            chi_f,
            M_f,
            QNM_dict,
            t_ref=0,
        )

        As.append([abs(term["A"]) for term in QNM_fits.values()])
        As_re.append([term["A"].real for term in QNM_fits.values()])
        As_im.append([term["A"].imag for term in QNM_fits.values()])

        errors.append(error)
        mismatches.append(mismatch)

    window = np.array([t0s[0], t0s[0] + window_size])

    best_CV_mean = np.inf
    while window[1] < t0s[-1]:
        idx1 = np.argmin(abs(t0s - window[0]))
        idx2 = np.argmin(abs(t0s - window[1])) + 1
        As_cut = As[idx1:idx2]

        As_mean = np.mean(As_cut, axis=0)
        As_stds = np.std(As_cut, axis=0)

        As_CV = As_stds / As_mean

        As_CV_mean = np.mean(As_CV)

        if As_CV_mean < best_CV_mean:
            best_CV_mean = As_CV_mean

            A_re_im = [
                np.mean(As_re[idx1:idx2], axis=0),
                np.mean(As_im[idx1:idx2], axis=0),
            ]
            A_re_im_std = [
                np.std(As_re[idx1:idx2], axis=0),
                np.std(As_im[idx1:idx2], axis=0),
            ]
            best_t0 = window[0]
            best_error = np.mean(errors[idx1:idx2])
            best_mismatch = np.mean(mismatches[idx1:idx2])

        window += t0s[1] - t0s[0]

    QNM_As = {}
    for i, term in enumerate(QNM_fits.values()):
        QNM_As[str(term["mode"]).replace(" ", "")] = {}
        QNM_As[str(term["mode"]).replace(" ", "")]["A"] = list(np.array(A_re_im).T[i])
        QNM_As[str(term["mode"]).replace(" ", "")]["A_std"] = list(
            np.array(A_re_im_std).T[i]
        )
        QNM_As["best CV"] = best_CV_mean
        QNM_As["best t0"] = best_t0
        QNM_As["error"] = best_error
        QNM_As["mismatch"] = best_mismatch

    return QNM_As


simulations = ["192"]

data = {}

for simulation in simulations:
    if simulation in data:
        print("Continue-ing! ", simulation)
        continue
    else:
        print(simulation)

    metadata = sxs.Metadata.from_file(f"{simulation}/metadata.json")
    m1 = metadata["reference-mass1"]
    m2 = metadata["reference-mass2"]
    M_total = m1 + m2

    abd, chi_f, M_f = read_waveform(simulation)

    abd.t *= M_total

    CoM_charge = abd.bondi_CoM_charge() / abd.bondi_four_momentum()[:, 0, None]

    idx1 = np.argmin(abs(abd.t - -1000))
    idx2 = np.argmin(abs(abd.t - -500)) + 1
    fit_0 = np.polyfit(abd.t[idx1:idx2], CoM_charge[idx1:idx2], 1)

    idx1 = np.argmin(abs(abd.t - 200))
    idx2 = np.argmin(abs(abd.t - 250)) + 1
    fit_1 = np.polyfit(abd.t[idx1:idx2], CoM_charge[idx1:idx2], 1)

    v_f = fit_1[0] - fit_0[0]

    kick_theta = np.arccos(
        np.dot(v_f, chi_f) / (np.linalg.norm(v_f) * np.linalg.norm(chi_f))
    )

    kick_rapidity = np.arctanh(np.linalg.norm(v_f))

    h = MT_to_WM(2.0 * abd.sigma.bar)

    metadata = sxs.Metadata.from_file(f"{simulation}/metadata.json")
    q = metadata["reference-mass1"] / metadata["reference-mass2"]
    chi1 = metadata["reference-dimensionless-spin1"]
    chi2 = metadata["reference-dimensionless-spin2"]

    theta = compute_change_in_flux(abd, t1=0)

    t0s = np.arange(20, 80 + 0.5, 0.5)

    QNM_As = fit_QNMs(
        h, np.linalg.norm(chi_f), M_f, t0s, tf=100, ell_max=3, window_size=20
    )

    data_per_sim = {
        "q": q,
        "chi1": chi1,
        "chi2": chi2,
        "M_f": M_f,
        "chi_f": np.linalg.norm(chi_f),
        "theta": theta,
        "kick theta": kick_theta,
        "kick rapidity": kick_rapidity,
    }

    data_per_sim = {**data_per_sim, **QNM_As}

    h = h[np.argmin(abs(h.t - 0)) : np.argmin(abs(h.t - 100)) + 1]
    R_error, _ = compute_Euler_angle_error(h, np.linalg.norm(chi_f), M_f)

    data_per_sim["R_error"] = R_error

    data[simulation] = data_per_sim

    with open("QNM_results_example.json", "w") as output_file:
        json.dump(
            dict(sorted(data.items())),
            output_file,
            indent=2,
            separators=(",", ": "),
            ensure_ascii=True,
        )
