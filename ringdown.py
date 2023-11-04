import qnm
import scri
import numpy as np
import spherical_functions as sf
from scri.sample_waveforms import modes_constructor

from varpro import varpro

from scipy.integrate import trapezoid
from scipy import linalg
from scipy.sparse import spdiags

_ksc = qnm.modes_cache


def waveform_mismatch(h_A, h_B, modes=None, t1=0, t2=100):
    """Compute the mismatch between two waveforms.
    Assumes that the waveforms have the same time arrays.

    Parameters
    ----------
    h_A : scri.WaveformModes
        One of the waveforms to use in the mismatch computation.
    h_B : scri.WaveformModes
        The other waveform to use in the mismatch computation.
    modes : list, optional
        list of modes, e.g., (ell, m), which will be included in the mismatch.
        Default is to use every mode.
    t1 : float, optional
        Lower boundary of times to include in the mismatch. [Default: 0.]
    t2 : float, optional
        Upper boundary of times to include in the mismatch. [Default: 100.]

    Returns
    -------
    mismatch : float
        Mismatch between the two waveforms.

    """
    h_A_copy = h_A.copy()
    h_B_copy = h_B.copy()
    if modes != None:
        for L, M in [
            (L_value, M_value)
            for L_value in range(2, h_A_copy.ell_max + 1)
            for M_value in range(-L_value, L_value + 1)
        ]:
            if not (L, M) in modes:
                h_A_copy.data[:, sf.LM_index(L, M, h_A_copy.ell_min)] *= 0
                h_B_copy.data[:, sf.LM_index(L, M, h_B_copy.ell_min)] *= 0

    h_A_copy = h_A_copy[np.argmin(abs(h_A.t - t1)) + 1 : np.argmin(abs(h_A.t - t2)) + 1]
    if h_A_copy.t.shape[0] != h_B_copy.t.shape[0]:
        h_B_copy = h_B_copy.interpolate(h_A_copy.t)
    else:
        h_B_copy = h_B_copy[
            np.argmin(abs(h_B.t - t1)) + 1 : np.argmin(abs(h_B.t - t2)) + 1
        ]
    return 1 - trapezoid(
        np.sum(np.real(h_A_copy.data * np.conjugate(h_B_copy.data)), axis=1), h_A_copy.t
    ) / np.sqrt(
        trapezoid(h_A_copy.norm(), h_A_copy.t) * trapezoid(h_B_copy.norm(), h_B_copy.t)
    )


def qnm_from_tuple(tup, chi, mass, s=-2):
    """Get frequency and spherical_spheroidal mixing from qnm module

    Parameters
    ----------
    tup : tuple
        Index (ell,m,n,sign) of QNM
    chi : float
        The dimensionless spin of the black hole, 0. <= chi < 1.
    mass : float
        The mass of the black hole, M > 0.
    s : int, optional [Default: -2]

    Returns
    -------
    omega: complex
        Frequency of QNM. This frequency is the same units as arguments,
        as opposed to being in units of remnant mass.
    C : complex ndarray
        Spherical-spheroidal decomposition coefficient array
    ells : ndarray
        List of ell values for the spherical-spheroidal mixing array

    """
    ell, m, n, sign = tup
    if sign == +1:
        mode_seq = _ksc(s, ell, m, n)
    elif sign == -1:
        mode_seq = _ksc(s, ell, -m, n)
    else:
        raise ValueError(
            "Last element of mode label must be "
            "+1 or -1, instead got {}".format(sign)
        )

    # The output from mode_seq is M*\omega
    try:
        Momega, _, C = mode_seq(chi, store=True)
    except:
        Momega, _, C = mode_seq(chi, interp_only=True)

    ells = qnm.angular.ells(s, m, mode_seq.l_max)

    if sign == -1:
        Momega = -np.conj(Momega)
        C = (-1) ** (ell + ells) * np.conj(C)

    # Convert from M*\omega to \omega
    omega = Momega / mass
    return omega, C, ells


def qnm_modes(chi, mass, mode_dict, dest=None, t_0=0.0, t_ref=0.0, **kwargs):
    """Convert a dictionary of QNMs to a scri.WaveformModes object.
    Additional keyword arguments are passed to `modes_constructor`.

    Parameters
    ----------
    chi : float
        The dimensionless spin of the black hole, 0. <= chi < 1.
    mass : float
        The mass of the black hole, M > 0.
    mode_dict : dict
        Dict with terms which are either QNM or other terms.
        Each term should be a dict with:
            - a 'type' (QNM or other),
            - a 'mode' (if 'type' == 'QNM')
              in the format (ell, m, n, sign)
            - a 'A'
            - a 'omega'
                - this is really only necessary if the QNM 'type'
                  is 'other', in which case this 'omega' is used as the frequency
    dest : ndarray, optional [Default: None]
        If passed, the storage to use for the scri.WaveformModes.data.
        Must be the correct shape.
    t_0 : float, optional [Default: 0.]
        Waveform model is 0 for t < t_0.
    t_ref : float, optional [Default: 0.]
        Time at which amplitudes are specified.

    Returns
    -------
    Q : scri.WaveformModes

    """
    s = -2

    ell_max = 12

    def data_functor(t, LM):
        d_shape = (t.shape[0], LM.shape[0])

        if dest is None:
            data = np.zeros(d_shape, dtype=complex)
        else:
            if (dest.shape != d_shape) or (dest.dtype is not np.dtype(complex)):
                raise TypeError("dest has wrong dtype or shape")
            data = dest
            data.fill(0.0)

        chi_is_array = type(chi) == list or type(chi) == np.ndarray
        mass_is_array = type(mass) == list or type(mass) == np.ndarray
        for term in mode_dict.values():
            if term["type"] == "QNM":
                if chi_is_array or mass_is_array:
                    for i in range(len(t)):
                        if chi_is_array:
                            chi_value = chi[i]
                        else:
                            chi_value = chi

                        if mass_is_array:
                            mass_value = mass[i]
                        else:
                            mass_value = mass

                        ell, m, n, sign = term["mode"]
                        omega, C, ells = qnm_from_tuple(
                            (ell, m, n, sign), chi_value, mass_value, s
                        )

                        A = term["A"]

                        if t[i] < t_0:
                            expiwt = 0.0
                        else:
                            expiwt = np.exp(complex(0.0, -1.0) * omega * (t[i] - t_ref))
                        for _l, _m in LM:
                            if _m == m:
                                c_l = C[ells == _l]
                                if len(c_l) > 0:
                                    c_l = c_l[0]

                                data[i, sf.LM_index(_l, _m, min(LM[:, 0]))] += (
                                    A * expiwt * c_l
                                )
                else:
                    ell, m, n, sign = term["mode"]
                    omega, C, ells = qnm_from_tuple((ell, m, n, sign), chi, mass, s)

                    A = term["A"]

                    expiwt = np.exp(complex(0.0, -1.0) * omega * (t - t_ref))
                    expiwt[t < t_0] = 0.0
                    for _l, _m in LM:
                        if _m == m:
                            c_l = C[ells == _l]
                            if len(c_l) > 0:
                                c_l = c_l[0]

                            data[:, sf.LM_index(_l, _m, min(LM[:, 0]))] += (
                                A * expiwt * c_l
                            )
            elif term["type"] == "other":
                omega = term["omega"]

                A = term["A"]

                expiwt = np.exp(complex(0.0, -1.0) * omega * (t - t_ref))
                expiwt[t < t_0] = 0.0
                data[
                    :,
                    sf.LM_index(
                        term["target mode"][0], term["target mode"][1], min(LM[:, 0])
                    ),
                ] += (
                    A * expiwt
                )
            else:
                raise ValueError("QNM term type not recognized...")

        return data

    return modes_constructor(
        "qnm_modes({0}, {1}, {2}, t_0={3}, t_ref={4}, **{5})".format(
            chi, mass, mode_dict, t_0, t_ref, kwargs
        ),
        data_functor,
        **kwargs
    )


def qnm_modes_as(
    chi, mass, mode_dict, W_other, dest=None, t_0=0.0, t_ref=0.0, **kwargs
):
    """Convert a dictionary of QNMs to a scri.WaveformModes object.
    Will match the structure of W_other.
    Additional keyword arguments are passed to `modes_constructor`.

    Parameters
    ----------
    chi : float
        The dimensionless spin of the black hole, 0. <= chi < 1.
    mass : float
        The mass of the black hole, M > 0.
    mode_dict : dict
        Dict with terms which are either QNM or other terms.
        Each term should be a dict with:
            - a 'type' (QNM or other),
            - a 'mode' (if 'type' == 'QNM')
              in the format (ell, m, n, sign)
            - a 'A'
            - a 'omega'
                - this is really only necessary if the QNM 'type'
                  is 'other', in which case this 'omega' is used as the frequency
    W_other : scri.WaveformModes object
        Get the time and LM from this scri.WaveformModes object
    dest : ndarray, optional [Default: None]
        If passed, the storage to use for the scri.WaveformModes.data.
        Must be the correct shape.
    t_0 : float, optional [Default: 0.]
        Waveform model is 0 for t < t_0.
    t_ref : float, optional [Default: 0.]
        Time at which amplitudes are specified.

    Returns
    -------
    Q : scri.WaveformModes

    """
    t = W_other.t
    ell_min = W_other.ell_min
    ell_max = W_other.ell_max

    return qnm_modes(
        chi,
        mass,
        mode_dict,
        dest=dest,
        t_0=t_0,
        t_ref=t_ref,
        t=t,
        ell_min=ell_min,
        ell_max=ell_max,
        **kwargs
    )


def fit_ringdown_waveform_functions(t, fixed_QNMs, free_QNMs, t_ref=0):
    """QNM fitting function for varpro.
    Computes Phi (the QNM waveform) and
    dPhi (the derivative of the QNM waveform w.r.t. nonlinear parameters).

    (see fit_ringdown_waveform for more details).

    """
    N_fixed = len(fixed_QNMs) // 2
    N_free = len(free_QNMs) // 2
    N = N_fixed + N_free

    omegas = [fixed_QNMs[2 * i] + 1j * fixed_QNMs[2 * i + 1] for i in range(N_fixed)]
    omegas += [free_QNMs[2 * i] + 1j * free_QNMs[2 * i + 1] for i in range(N_free)]

    # Construct Phi, with the four terms (per QNM) decomposed as
    # QNM = term1 + term2 + term3 + term4, where term1 and term2 are the real components
    # and term 3 and term 4 are the imaginary components. Specifically, these are
    # (a + i * b) * exp(-i \omega t)] =
    # a Re[exp(-i \omega t)] - b * Im[exp(-i \omega t)] +
    # i * (a * Im[exp(-i \omega t)] + b * Im[exp(-i \omega t)]).
    # We will put the real terms in the 1st part of Phi, and the imaginary terms in the 2nd part
    Phi = np.zeros((2 * t.size, 2 * N))
    for i in range(N):
        # re
        # term 1
        Phi[: t.size, 2 * i] = np.real(np.exp(-1j * omegas[i] * (t - t_ref)))
        # term 2
        Phi[: t.size, 2 * i + 1] = -np.imag(np.exp(-1j * omegas[i] * (t - t_ref)))
        # im
        # term 3
        Phi[t.size :, 2 * i] = np.imag(np.exp(-1j * omegas[i] * (t - t_ref)))
        # term 4
        Phi[t.size :, 2 * i + 1] = np.real(np.exp(-1j * omegas[i] * (t - t_ref)))

    # We have 4*N terms per Phi entry (4 terms (see above))
    # and 2*N_free parameters, since each frequency has a real and imaginary part.
    # So there Phi must be of length (4*N)*(2*N_free).
    # We'll order the nonlinear parameter dependence in the trivial way, i.e., 0, 1, 2, ...
    # but with the fixed QNMs first.
    Ind = np.array(
        [
            [i // (2 * N_free) for i in range((2 * N) * (2 * N_free))],
            (2 * N) * list(np.arange(2 * N_free)),
        ]
    )

    # Construct dPhi, where each of the 4 terms (per QNM), if the QNM is free, has two components.
    dPhi = np.zeros((2 * t.size, (2 * N) * (2 * N_free)))
    # Loop over freqs
    for freq in range(N):
        # Loop over terms in real and imaginary parts,
        # i.e., if term == 0 then we're considering term1 and term3
        # while if term == 1 then we're considering term2 and term4
        for term in range(2):
            # Loop over the number of freq_derivs we have to take
            # which is just the number of free QNMs
            for freq_deriv in range(N_free):
                # shift to current QNM, shift to current term, shift to current frequency
                idx = (2 * N_free) * (2 * freq) + (2 * N_free) * term + 2 * freq_deriv

                # First, set the dPhi terms to zero when they correspond to a QNM w/ fixed frequency
                if freq - N_fixed != freq_deriv:
                    # term1/term2
                    # deriv w.r.t real part of freq
                    dPhi[: t.size, idx] = 0
                    # deriv w.r.t imag part of freq
                    dPhi[: t.size, idx + 1] = 0
                    # term3/term4
                    # deriv w.r.t real part of freq
                    dPhi[t.size :, idx] = 0
                    # deriv w.r.t imag part of freq
                    dPhi[t.size :, idx + 1] = 0
                else:
                    if term == 0:
                        # term 1
                        # deriv w.r.t real part of freq
                        dPhi[: t.size, idx] = np.real(
                            -1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )
                        # deriv w.r.t imag part of freq
                        dPhi[: t.size, idx + 1] = np.real(
                            -1j * 1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )
                        # term 3
                        # deriv w.r.t real part of freq
                        dPhi[t.size :, idx] = np.imag(
                            -1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )
                        # deriv w.r.t imag part of freq
                        dPhi[t.size :, idx + 1] = np.imag(
                            -1j * 1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )
                    else:
                        # term 2
                        # deriv w.r.t real part of freq
                        dPhi[: t.size, idx] = -np.imag(
                            -1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )
                        # deriv w.r.t imag part of freq
                        dPhi[: t.size, idx + 1] = -np.imag(
                            -1j * 1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )
                        # term 4
                        # deriv w.r.t real part of freq
                        dPhi[t.size :, idx] = np.real(
                            -1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )
                        # deriv w.r.t imag part of freq
                        dPhi[t.size :, idx + 1] = np.real(
                            -1j * 1j * t * np.exp(-1j * omegas[freq] * (t - t_ref))
                        )

    return Phi, dPhi, Ind


def fit_ringdown_waveform_LLSQ_S2(
    h_ring, modes, times, chi_f, M_f, fixed_QNMs, t_ref=0
):
    """Linear least squares routine for fitting a NR waveform with QNMs.

    (see fit_ringdown_waveform for more details).

    """
    t_0 = times[0]

    m_list = []
    [
        m_list.append(term["mode"][1])
        for term in fixed_QNMs.values()
        if term["mode"][1] not in m_list
    ]
    # break problem into one m at a time.
    # the m's are decoupled, and the truncation in ell for each m is different.
    for m in m_list:
        mode_labels_m = [
            (i, term)
            for i, term in enumerate(fixed_QNMs.values())
            if term["mode"][1] == m
        ]

        # restrict the modes included in the least squares fit to the modes of interest
        ell_min_m = h_ring.ell_min
        ell_max_m = h_ring.ell_max
        if modes is None:
            data_index_m = [
                sf.LM_index(l, m, h_ring.ell_min) for l in range(2, ell_max_m + 1)
            ]
        else:
            data_index_m = [
                sf.LM_index(l, m, h_ring.ell_min)
                for l in range(2, ell_max_m + 1)
                if (l, m) in modes
            ]
            ell_min_m = min(np.array([_l for (_l, _m) in modes if m == _m]))
            ell_max_m = max(np.array([_l for (_l, _m) in modes if m == _m]))

        A = np.zeros((len(h_ring.t), ell_max_m - ell_min_m + 1), dtype=complex)
        B = np.zeros(
            (len(h_ring.t), ell_max_m - ell_min_m + 1, len(mode_labels_m)),
            dtype=complex,
        )

        h_ring_trunc = h_ring[:, : ell_max_m + 1]
        A = h_ring_trunc.data[:, data_index_m]
        for mode_index, (i, term) in enumerate(mode_labels_m):
            term_w_A = term.copy()
            term["A"] = 1
            h_QNM = qnm_modes_as(
                chi_f, M_f, {0: term}, h_ring_trunc, t_0=t_0, t_ref=t_ref
            )
            B[:, :, mode_index] = h_QNM.data[:, data_index_m]

        A = np.reshape(A, len(h_ring.t) * (ell_max_m - ell_min_m + 1))
        B = np.reshape(
            B, (len(h_ring.t) * (ell_max_m - ell_min_m + 1), len(mode_labels_m))
        )
        C = np.linalg.lstsq(B, A, rcond=None)

        count = 0
        for i, term in mode_labels_m:
            fixed_QNMs[i]["A"] = C[0][count]
            count += 1

    h_QNM = qnm_modes_as(chi_f, M_f, fixed_QNMs, h_ring, t_0=t_0, t_ref=t_ref)

    h_diff = h_ring.copy()
    h_diff.data *= 0

    if modes is None:
        modes = [(L, M) for L in range(2, h_ring.ell_max + 1) for M in range(-L, L + 1)]

    for mode in modes:
        L, M = mode
        h_diff.data[:, sf.LM_index(L, M, h_diff.ell_min)] = (
            h_ring.data[:, sf.LM_index(L, M, h_ring.ell_min)]
            - h_QNM.data[:, sf.LM_index(L, M, h_QNM.ell_min)]
        )

    error = (
        0.5 * trapezoid(h_diff.norm(), h_diff.t) / trapezoid(h_ring.norm(), h_ring.t)
    )
    mismatch = waveform_mismatch(h_ring, h_QNM, modes=modes, t1=times[0], t2=times[1])

    return h_QNM, h_ring, error, mismatch, fixed_QNMs, None, "LLSQ"


def fit_ringdown_waveform_LLSQ(h_ring, modes, times, chi_f, M_f, fixed_QNMs, t_ref=0):
    """Linear least squares routine for fitting a NR waveform with QNMs.

    (see fit_ringdown_waveform for more details).

    """
    t_0 = times[0]

    m_list = []
    [
        m_list.append(m)
        for (_, m) in [term["target mode"] for term in fixed_QNMs.values()]
        if m not in m_list
    ]
    # break problem into one m at a time.
    # the m's are decoupled, and the truncation in ell for each m is different.
    for m in m_list:
        mode_labels_m = [
            (i, term)
            for i, term in enumerate(fixed_QNMs.values())
            if term["target mode"][1] == m
        ]

        # restrict the modes included in the least squares fit to the modes of interest
        ell_min_m = h_ring.ell_min
        ell_max_m = h_ring.ell_max
        if modes is None:
            data_index_m = [
                sf.LM_index(l, m, h_ring.ell_min) for l in range(2, ell_max_m + 1)
            ]
        else:
            data_index_m = [
                sf.LM_index(l, m, h_ring.ell_min)
                for l in range(2, ell_max_m + 1)
                if (l, m) in modes
            ]
            ell_min_m = min(np.array([_l for (_l, _m) in modes if m == _m]))
            ell_max_m = max(np.array([_l for (_l, _m) in modes if m == _m]))

        A = np.zeros((len(h_ring.t), ell_max_m - ell_min_m + 1), dtype=complex)
        B = np.zeros(
            (len(h_ring.t), ell_max_m - ell_min_m + 1, len(mode_labels_m)),
            dtype=complex,
        )

        h_ring_trunc = h_ring[:, : ell_max_m + 1]
        A = h_ring_trunc.data[:, data_index_m]
        for mode_index, (i, term) in enumerate(mode_labels_m):
            term_w_A = term.copy()
            term["A"] = 1
            h_QNM = qnm_modes_as(
                chi_f, M_f, {0: term}, h_ring_trunc, t_0=t_0, t_ref=t_ref
            )
            B[:, :, mode_index] = h_QNM.data[:, data_index_m]

        A = np.reshape(A, len(h_ring.t) * (ell_max_m - ell_min_m + 1))
        B = np.reshape(
            B, (len(h_ring.t) * (ell_max_m - ell_min_m + 1), len(mode_labels_m))
        )
        C = np.linalg.lstsq(B, A, rcond=None)

        count = 0
        for i, term in mode_labels_m:
            fixed_QNMs[i]["A"] = C[0][count]
            count += 1

    h_QNM = qnm_modes_as(chi_f, M_f, fixed_QNMs, h_ring, t_0=t_0, t_ref=t_ref)

    h_diff = h_ring.copy()
    h_diff.data *= 0
    for mode in modes:
        L, M = mode
        h_diff.data[:, sf.LM_index(L, M, h_diff.ell_min)] = (
            h_ring.data[:, sf.LM_index(L, M, h_ring.ell_min)]
            - h_QNM.data[:, sf.LM_index(L, M, h_QNM.ell_min)]
        )

    error = (
        0.5 * trapezoid(h_diff.norm(), h_diff.t) / trapezoid(h_ring.norm(), h_ring.t)
    )
    mismatch = waveform_mismatch(h_ring, h_QNM, modes=modes, t1=times[0], t2=times[1])

    wresid_norm = np.linalg.norm(
        h_diff.data[:, sf.LM_index(modes[0][0], modes[0][1], h_diff.ell_min)]
    )

    # Compute std devs (is this right?)
    N_QNMs = len(fixed_QNMs)
    N_free_QNMs = 0
    wresid_re_im = h_diff.data[:, sf.LM_index(modes[0][0], modes[0][1], h_diff.ell_min)]

    fixed_QNM_components = []
    for fixed_QNM in fixed_QNMs.values():
        fixed_QNM_components.append(fixed_QNM["omega"].real)
        fixed_QNM_components.append(fixed_QNM["omega"].imag)

    Phi, dPhi, Ind = fit_ringdown_waveform_functions(h_diff.t, fixed_QNM_components, [])
    xx, pp = dPhi.shape
    J = np.zeros((2 * len(h_diff.t), 2 * N_free_QNMs))
    for kk in np.arange(pp):
        j = Ind[0, kk]
        i = Ind[1, kk]
        if j > 2 * N_QNMs:
            J[:, i] = J[:, i] + dPhi[:, kk]
        else:
            J[:, i] = J[:, i] + c[j] * dPhi[:, kk]

    Phi_re_im = np.zeros((Phi.shape[0] // 2, Phi.shape[1] // 2), dtype=complex)
    for i in range(Phi.shape[1] // 2):
        Phi_re_im[:, i] += (
            Phi[: h_diff.t.size, 2 * i] + 1j * Phi[h_diff.t.size :, 2 * i]
        )
        Phi_re_im[:, i] += (
            Phi[: h_diff.t.size, 2 * i + 1] + 1j * Phi[h_diff.t.size :, 2 * i + 1]
        )

    J_re_im = np.zeros((J.shape[0] // 2, J.shape[1] // 2), dtype=complex)
    for i in range(J.shape[1] // 2):
        J_re_im[:, i] += J[: h_diff.t.size, 2 * i] + 1j * J[h_diff.t.size :, 2 * i]
        J_re_im[:, i] += (
            J[: h_diff.t.size, 2 * i + 1] + 1j * J[h_diff.t.size :, 2 * i + 1]
        )

    W_re_im = spdiags(np.ones(h_diff.t.size), 0, h_diff.t.size, h_diff.t.size)

    Mat = W_re_im.dot(
        np.concatenate((Phi_re_im[:, np.arange(N_QNMs)], J_re_im), axis=1)
    )

    sigma_K2 = np.conjugate(np.transpose(wresid_re_im)).dot(wresid_re_im) / (
        h_diff.t.size - N_QNMs - N_free_QNMs + 1
    )
    sigma_J2 = np.transpose(wresid_re_im).dot(wresid_re_im) / (
        h_diff.t.size - N_QNMs - N_free_QNMs + 1
    )

    Qj, Rj, Pj = linalg.qr(Mat, mode="economic", pivoting=True)
    T2 = linalg.solve_triangular(Rj, (np.identity(Rj.shape[0])))

    K_cov = np.zeros(Rj.shape, dtype=complex)
    J_cov = np.zeros(Rj.shape, dtype=complex)
    K_cov_temp = np.zeros(Rj.shape, dtype=complex)
    J_cov_temp = np.zeros(Rj.shape, dtype=complex)

    K_cov_temp[:, Pj] = np.conjugate(np.transpose(T2)).dot(T2) * sigma_K2
    J_cov_temp[:, Pj] = np.transpose(T2).dot(T2) * sigma_J2

    K_cov[Pj, :] = K_cov_temp
    J_cov[Pj, :] = J_cov_temp

    std_dev_of_real_parts = np.sqrt(np.diag(0.5 * (K_cov + J_cov).real))
    std_dev_of_imag_parts = np.sqrt(np.diag(0.5 * (K_cov - J_cov).real))

    std_dev_params = [
        std_dev_of_real_parts[i] + 1j * std_dev_of_imag_parts[i]
        for i in range(len(std_dev_of_imag_parts))
    ]

    return h_QNM, h_ring, error, mismatch, fixed_QNMs, std_dev_params, "LLSQ"


def fit_ringdown_waveform(
    h,
    mode,
    times,
    chi_f,
    M_f,
    fixed_QNMs,
    N_free_QNMs,
    initial_guess=None,
    bounds=None,
    t_ref=0,
    ftol=1e-8,
    gtol=1e-8,
):
    """Fit a waveform with QNMs.

    Parameters
    ----------
    h: scri.WaveformModes
        Input waveform to fit to.
    mode: tuple
        Mode of waveform, e.g., (ell, m), to fit to.
    times: tuple
        Times to fit over, e.g., (0, 100).
    chi: float
        The dimensionless spin of the black hole, 0. <= chi < 1.
    mass: float
        The mass of the black hole, M > 0.
    fixed_QNMs: list
        List of fixed QNMs to fit to, e.g., [(2,2,0,1), (2,2,1,1), ...]
    N_free_QNMs: int
        Number of free frequencies to fit for using varpro.
    initial_guess: list, optional
        Initial guess for free frequencies.
        Default is [0.5, -0.2] * N_free_QNMs.
    bounds: list, optional
        Bounds for free frequencies.
        Default is [(-np.inf, np.inf), (-np.inf, 0)] * N_free_QNMs.
    t_ref: float, optional [Default: 0.]
        Time at which amplitudes are specified.
    ftol: float, optional [Default: 1e-8]
        ftol used in nonlinear least squares optimization.
    gtol: float, optional [Default: 1e-8]
        gtol used in nonlinear least squares optimization.

    Returns
    -------
    h_QNM: scri.WaveformModes
        Waveform of QNMs fit to the input waveform.
    h_ring: scri.WaveformModes
        Waveform (computed from the input waveform) that was fit to.
    error: float
        error between the waveforms, i.e., 0.5 * (relative error over times)^2.
    mismatch: float
        mismatch between the waveforms.
    QNMs: dict
        Dictionary of the fit QNMs.
    std_dev_params: np.array
        standard deviations for fit results.
    message: string
        Output message of nonlinear least squares optimization.

    """
    # Create ringdown waveform and varpro inputs
    L, M = mode
    idx1 = np.argmin(abs(h.t - times[0]))
    idx2 = np.argmin(abs(h.t - times[1])) + 1

    h_ring = h.copy()[idx1:idx2]

    if N_free_QNMs == 0:
        return fit_ringdown_waveform_LLSQ(h_ring, [mode], times, chi_f, M_f, fixed_QNMs)

    N_fixed_QNMs = len(fixed_QNMs)
    N_QNMs = N_fixed_QNMs + N_free_QNMs

    w = np.ones(2 * h_ring.t.size)
    if initial_guess is None:
        initial_guess = np.array([0.5, -0.2] * N_free_QNMs)

    fixed_QNM_components = []
    for fixed_QNM in fixed_QNMs.values():
        fixed_QNM_components.append(fixed_QNM["omega"].real)
        fixed_QNM_components.append(fixed_QNM["omega"].imag)

    if bounds is None:
        bounds = ([-np.inf, -np.inf], [np.inf, 0])

    # Use varpro
    (
        res,
        c,
        wresid,
        wresid_norm,
        y_est,
        CorMx,
        std_dev_params,
        message,
        success,
    ) = varpro.varpro(
        h_ring.t,
        np.concatenate(
            (
                h_ring.data[:, sf.LM_index(L, M, h_ring.ell_min)].real,
                h_ring.data[:, sf.LM_index(L, M, h_ring.ell_min)].imag,
            )
        ),
        w,
        initial_guess,
        2 * N_QNMs,
        lambda alpha: fit_ringdown_waveform_functions(
            h_ring.t, fixed_QNM_components, alpha
        ),
        bounds=(bounds[0] * N_free_QNMs, bounds[1] * N_free_QNMs),
        ftol=ftol,
        gtol=gtol,
        verbose=False,
    )

    # Compute error and mismatch
    h_QNM = h_ring.copy()
    h_QNM.data *= 0
    h_QNM.data[:, sf.LM_index(L, M, h_QNM.ell_min)] = (
        y_est[: h_ring.t.size] + 1j * y_est[h_ring.t.size :]
    )

    h_diff = h_ring.copy()
    h_diff.data *= 0
    h_diff.data[:, sf.LM_index(L, M, h_diff.ell_min)] = h_ring.data[
        :, sf.LM_index(L, M, h_ring.ell_min)
    ]
    h_diff.data -= h_QNM.data

    error = (
        0.5 * trapezoid(h_diff.norm(), h_diff.t) / trapezoid(h_ring.norm(), h_ring.t)
    )
    mismatch = waveform_mismatch(
        h_ring, h_QNM, modes=[(L, M)], t1=times[0], t2=times[1]
    )

    # Create QNMs Dict
    As = np.array([c[2 * i] + 1j * c[2 * i + 1] for i in range(N_QNMs)])
    omegas = np.array(
        [
            fixed_QNM_components[2 * i] + 1j * fixed_QNM_components[2 * i + 1]
            for i in range(N_fixed_QNMs)
        ]
        + [res[2 * i] + 1j * res[2 * i + 1] for i in range(N_free_QNMs)]
    )
    QNMs = {
        i: {
            "type": "other",
            "A": As[i],
            "omega": omegas[i],
            "target mode": (L, M),
        }
        for i in range(N_QNMs)
    }

    # Compute std devs (is this right?)
    y_est_re_im = y_est[: h_diff.t.size] + 1j * y_est[h_diff.t.size :]
    wresid_re_im = h_ring.data[:, sf.LM_index(L, M, h_ring.ell_min)] - y_est_re_im

    Phi, dPhi, Ind = fit_ringdown_waveform_functions(
        h_diff.t, fixed_QNM_components, res
    )
    xx, pp = dPhi.shape
    J = np.zeros((2 * len(h_diff.t), 2 * N_free_QNMs))
    for kk in np.arange(pp):
        j = Ind[0, kk]
        i = Ind[1, kk]
        if j > 2 * N_QNMs:
            J[:, i] = J[:, i] + dPhi[:, kk]
        else:
            J[:, i] = J[:, i] + c[j] * dPhi[:, kk]

    Phi_re_im = np.zeros((Phi.shape[0] // 2, Phi.shape[1] // 2), dtype=complex)
    for i in range(Phi.shape[1] // 2):
        Phi_re_im[:, i] += (
            Phi[: h_diff.t.size, 2 * i] + 1j * Phi[h_diff.t.size :, 2 * i]
        )
        Phi_re_im[:, i] += (
            Phi[: h_diff.t.size, 2 * i + 1] + 1j * Phi[h_diff.t.size :, 2 * i + 1]
        )

    J_re_im = np.zeros((J.shape[0] // 2, J.shape[1] // 2), dtype=complex)
    for i in range(J.shape[1] // 2):
        J_re_im[:, i] += J[: h_diff.t.size, 2 * i] + 1j * J[h_diff.t.size :, 2 * i]
        J_re_im[:, i] += (
            J[: h_diff.t.size, 2 * i + 1] + 1j * J[h_diff.t.size :, 2 * i + 1]
        )

    W_re_im = spdiags(np.ones(h_diff.t.size), 0, h_diff.t.size, h_diff.t.size)

    Mat = W_re_im.dot(
        np.concatenate((Phi_re_im[:, np.arange(N_QNMs)], J_re_im), axis=1)
    )

    sigma_K2 = np.conjugate(np.transpose(wresid_re_im)).dot(wresid_re_im) / (
        h_diff.t.size - N_QNMs - N_free_QNMs + 1
    )
    sigma_J2 = np.transpose(wresid_re_im).dot(wresid_re_im) / (
        h_diff.t.size - N_QNMs - N_free_QNMs + 1
    )

    Qj, Rj, Pj = linalg.qr(Mat, mode="economic", pivoting=True)
    T2 = linalg.solve_triangular(Rj, (np.identity(Rj.shape[0])))

    K_cov = np.zeros(Rj.shape, dtype=complex)
    J_cov = np.zeros(Rj.shape, dtype=complex)
    K_cov_temp = np.zeros(Rj.shape, dtype=complex)
    J_cov_temp = np.zeros(Rj.shape, dtype=complex)

    K_cov_temp[:, Pj] = np.conjugate(np.transpose(T2)).dot(T2) * sigma_K2
    J_cov_temp[:, Pj] = np.transpose(T2).dot(T2) * sigma_J2

    K_cov[Pj, :] = K_cov_temp
    J_cov[Pj, :] = J_cov_temp

    std_dev_of_real_parts = np.sqrt(np.diag(0.5 * (K_cov + J_cov).real))
    std_dev_of_imag_parts = np.sqrt(np.diag(0.5 * (K_cov - J_cov).real))

    std_dev_params = [
        std_dev_of_real_parts[i] + 1j * std_dev_of_imag_parts[i]
        for i in range(len(std_dev_of_imag_parts))
    ]

    return h_QNM, h_ring, error, mismatch, QNMs, std_dev_params, message
