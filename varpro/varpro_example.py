# varpro_example.m
# Sample problem illustrating the use of varpro.m
#
# Observations y(t) are given at 10 values of t.
#
# The model for this function y(t) is
#
#   eta(t) = c1 exp(-alpha2 t)*cos(alpha3 t)
#          + c2 exp(-alpha1 t)*cos(alpha2 t)
#
# The linear parameters are c1 and c2, and the
#  nonlinear parameters are alpha1, alpha2, and alpha3.
#
# The two nonlinear functions in the model are
#
#   Phi1(alpha,t) = exp(-alpha2 t)*cos(alpha3 t),
#   Phi2(alpha,t) = exp(-alpha1 t)*cos(alpha2 t).
#
# Dianne P. O'Leary and Bert W. Rust, September 2010

import numpy as np
from varpro import *


def examplePhiFunction(alpha, t):
    # Phi,dPhi,Ind = examplePhiFunction(alpha,t)
    # This is a sample user-defined function to be used by varpro.m.

    # The model for this sample problem is

    #   eta(t) = c0 exp(-alpha1 t)*cos(alpha2 t)
    #                + c1 exp(-alpha0 t)*cos(alpha1 t)
    #              = c0 Phi0 + c1 Phi1

    # Given t and alpha, we evaluate Phi, dPhi, and Ind.

    # Dianne P. O'Leary and Bert W. Rust, September 2010.

    Phi = np.empty([t.shape[0], 2])

    Phi[:, 0] = np.multiply(np.exp(-alpha[1] * t), np.cos(alpha[2] * t))
    Phi[:, 1] = np.multiply(np.exp(-alpha[0] * t), np.cos(alpha[1] * t))
    # The nonzero partial derivatives of Phi with respect to alpha are
    #              d Phi_0 / d alpha_1 ,
    #              d Phi_0 / d alpha_2 ,
    #              d Phi_1 / d alpha_0 ,
    #              d Phi_1 / d alpha_1 ,
    # and this determines Ind.
    # The ordering of the columns of Ind is arbitrary but must match dPhi.

    Ind = np.array([[0, 0, 1, 1], [1, 2, 0, 1]])
    # Evaluate the four nonzero partial derivatives of Phi at each of
    # the data points and store them in dPhi.

    dPhi = np.empty([t.shape[0], 4])
    dPhi[:, 0] = np.multiply(-np.ndarray.flatten(t), Phi[:, 0])
    dPhi[:, 1] = np.multiply(
        np.multiply(-t, np.exp(-alpha[1] * t)), np.sin(alpha[2] * t)
    )
    dPhi[:, 2] = np.multiply(-np.ndarray.flatten(t), Phi[:, 1])
    dPhi[:, 3] = np.multiply(
        np.multiply(-t, np.exp(-alpha[0] * t)), np.sin(alpha[1] * t)
    )
    return Phi, dPhi, Ind


print("****************************************")
print("Sample problem illustrating the use of varpro.m")

# Data observations y(t) were taken at these times:
t = np.array([0, 0.1, 0.22, 0.31, 0.46, 0.5, 0.63, 0.78, 0.85, 0.97])
y = np.array(
    [
        6.9842,
        5.1851,
        2.8907,
        1.4199,
        -0.2473,
        -0.5243,
        -1.0156,
        -1.026,
        -0.9165,
        -0.6805,
    ]
)

# The weights for the least squares fit are stored in w.
w = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 0.5, 0.5])

# Set the initial guess for alpha and call varpro to estimate
# alpha and c.
alphainit = np.array([0.5, 2, 3])

# Example call for bound alpha[0] >= 0.4
# bounds=([0.4,-np.inf,-np.inf],np.inf)
# alpha,c,wresid,resid_norm,y_est= varpro(t,y,w,alphainit,2,lambda alpha = None: examplePhiFunction(alpha,t),bounds)
alpha, c, wresid, resid_norm, y_est, CorMx, std_dev_params = varpro(
    t, y, w, alphainit, 2, lambda alpha: examplePhiFunction(alpha, t)
)

# The data y(t) were generated using the parameters
#         alphatrue = [1.0; 2.5; 4.0], ctrue = [6; 1].
# Noise was added to the data, so these parameters do not provide
# the best fit.  The computed solution is:
#    Linear Parameters:
#      5.8416357e+00    1.1436854e+00
#    Nonlinear Parameters:
#      1.0132255e+00    2.4968675e+00    4.0625148e+00
# Note that there are many sets of parameters that fit the data well.

print("End of sample problem.")
print("****************************************")
