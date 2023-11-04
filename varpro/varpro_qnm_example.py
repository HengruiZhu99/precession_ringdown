import sys
import random
import numpy as np

from varpro import *

# Fit QNM ringdown-like data with varpro. The QNM ringdown-like data is produced randomly and contains
# N_fixed QNM frequencies and N_free QNM frequncies, which are set by sys.argv[1] and sys.argv[2].
# Initial guesses for these parameters are randomly taken to be 40% within the true parameter values.
# If varpro fails to match the linear or nonlinear parameters to a relative tolerance of 1e-5 and an
# absolute tolerance of 1e-8, then we run scipy's least_squares implementation to see if it can do better (as a test).


def build_ringdown_model(As_and_omegas, fixed_omegas, t, data, t_ref=0):
    h_ringdown = np.zeros(t.size, dtype=complex)
    for i in range((len(As_and_omegas) + len(fixed_omegas)) // 4):
        if i < len(fixed_omegas) // 2:
            A = As_and_omegas[2 * i] + 1j * As_and_omegas[2 * i + 1]
            omega = fixed_omegas[2 * i] + 1j * fixed_omegas[2 * i + 1]
        else:
            idx = 2 * len(fixed_omegas) // 2 + 4 * (i - len(fixed_omegas) // 2)
            A = As_and_omegas[idx] + 1j * As_and_omegas[idx + 1]
            omega = As_and_omegas[idx + 2] + 1j * As_and_omegas[idx + 3]

        h_ringdown += A * np.exp(-1j * omega * (t - t_ref))

    return np.linalg.norm(data - h_ringdown)


def fit_ringdown_waveform(t, fixed_QNMs, free_QNMs, t_ref=0):
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
                # print(freq, term, freq_deriv, idx, (freq - N_fixed != freq_deriv))

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


N_fixed = 1
if len(sys.argv) > 1:
    N_fixed = int(sys.argv[1])

N_free = 1
if len(sys.argv) > 2:
    N_free = int(sys.argv[2])

N = N_fixed + N_free

t = np.arange(0, 100, 0.1)

Amplitudes = [(4 * random.random() - 2) for i in range(2 * N)]

fixed_QNMs = []
for i in range(N_fixed):
    # re
    fixed_QNMs.append(2 * random.random() - 1)
    # im
    fixed_QNMs.append(random.random() - 1)

free_QNMs = []
initial_guess = []
for i in range(N_free):
    # re
    re_QNM = 2 * random.random() - 1
    free_QNMs.append(re_QNM)
    initial_guess.append((1 + (0.8 * random.random() - 0.4)) * re_QNM)
    # im
    im_QNM = random.random() - 1
    free_QNMs.append(im_QNM)
    initial_guess.append((1 + (0.8 * random.random() - 0.4)) * im_QNM)
initial_guess = np.array(initial_guess)

QNMs = fixed_QNMs + free_QNMs

print("-------------------")
print("Creating data with...")
print(f"Amplitudes: {Amplitudes}")
print(f"Fixed QNMs: {fixed_QNMs}")
print(f"Free QNMs: {free_QNMs}")
print(f"Initial Guess: {initial_guess}")
print("-------------------\n")

data = np.zeros_like(t, dtype=complex)
for i in range(N):
    data += (Amplitudes[2 * i] + 1j * Amplitudes[2 * i + 1]) * np.exp(
        -1j * (QNMs[2 * i] + 1j * QNMs[2 * i + 1]) * t
    )

w = np.ones(2 * t.size)

res, c, wresid, wresid_norm, y_est, CorMx, std_dev_paramsm, message, success = varpro(
    t,
    np.concatenate((data.real, data.imag)),
    w,
    initial_guess,
    2 * N,
    lambda alpha: fit_ringdown_waveform(t, fixed_QNMs, alpha),
    bounds=([-np.inf, -np.inf] * N_free, [np.inf, 0] * N_free),
)

linear_param_success = np.allclose(np.sort(Amplitudes), np.sort(c))
nonlinear_param_success = np.allclose(np.sort(free_QNMs), np.sort(res))

print("-------------------")
print(
    "Linear parameters match: ",
    linear_param_success,
    "; error: ",
    np.linalg.norm(np.array(np.sort(Amplitudes)) - np.array(np.sort(c))),
)
print(
    "Nonlinear parameters match: ",
    nonlinear_param_success,
    "; error: ",
    np.linalg.norm(np.array(np.sort(free_QNMs)) - np.array(np.sort(res))),
)
print("-------------------")

if not linear_param_success or not nonlinear_param_success:
    print(
        "\nExecuting scipy.optimize.least_squares to see if failure is 'expected'...\n"
    )

    lower_bounds = []
    upper_bounds = []
    scipy_initial_guess = []
    for i in range(N):
        # Amplitude initial guesses
        scipy_initial_guess.append(1)
        lower_bounds.append(-np.inf)
        upper_bounds.append(np.inf)

        scipy_initial_guess.append(1)
        lower_bounds.append(-np.inf)
        upper_bounds.append(np.inf)

        # Omega initial guesses
        if i < N_fixed:
            continue
        else:
            idx = 2 * (i - N_fixed)
            scipy_initial_guess.append(initial_guess[idx])
            lower_bounds.append(-np.inf)
            upper_bounds.append(np.inf)
            scipy_initial_guess.append(initial_guess[idx + 1])
            lower_bounds.append(-np.inf)
            upper_bounds.append(0)

    res = least_squares(
        build_ringdown_model,
        scipy_initial_guess,
        bounds=(lower_bounds, upper_bounds),
        args=(fixed_QNMs, t, data),
    )

    scipy_linear_results = []
    scipy_nonlinear_results = []
    for i in range(N):
        if i < N_fixed:
            scipy_linear_results.append(res.x[2 * i])
            scipy_linear_results.append(res.x[2 * i + 1])
        else:
            idx = 2 * N_fixed + 4 * (i - N_fixed)
            scipy_linear_results.append(res.x[idx])
            scipy_linear_results.append(res.x[idx + 1])
            scipy_nonlinear_results.append(res.x[idx + 2])
            scipy_nonlinear_results.append(res.x[idx + 3])

    scipy_linear_param_success = np.allclose(
        np.sort(Amplitudes), np.sort(scipy_linear_results)
    )
    scipy_nonlinear_param_success = np.allclose(
        np.sort(free_QNMs), np.sort(scipy_nonlinear_results)
    )

    print("-------------------")
    print(
        "(scipy) Linear parameters match: ",
        scipy_linear_param_success,
        "; error: ",
        np.linalg.norm(
            np.array(np.sort(Amplitudes)) - np.array(np.sort(scipy_linear_results))
        ),
    )
    print(
        "(scipy) Nonlinear parameters match: ",
        scipy_nonlinear_param_success,
        "; error: ",
        np.linalg.norm(
            np.array(np.sort(free_QNMs)) - np.array(np.sort(scipy_nonlinear_results))
        ),
    )
    print("-------------------")

    if not scipy_linear_param_success or not scipy_nonlinear_param_success:
        print("\n*** Varpro failure expected. Don't worry. ***\n")
    else:
        print("\n*** Varpro failure not expected. ***\n")
