import numpy as np
from scipy.sparse import spdiags
from scipy.optimize import least_squares
from scipy import linalg


def varpro(
    t,
    y,
    w,
    alpha,
    n,
    ada,
    bounds=None,
    ftol=1e-8,
    gtol=1e-8,
    xtol=1e-8,
    max_nfev=None,
    verbose=False,
    **kwargs
):
    # Solve a separable nonlinear least squares problem.
    # This is a slightly simplified Python translation of the Matlab code by
    # Dianne P. O'Leary and Bert W. Rust. The original code is documented in
    # Computational Optimization and Applications, 54, 579 (2013). The Matlab
    # code itself with extensive comments and references is available in the
    # supplementary material to the online version of this paper at
    # doi:10.1007/s10589-012-9492-9
    #
    # Given a set of m observations y(0),...,y(m-1) at "times" t(0),...,t(m-1),
    # this program computes a weighted least squares fit using the model
    #
    #    eta(alpha,c,t) =
    #            c_0 * phi_0 (alpha,t) + ...  + c_(n-1) * phi_(n-1) (alpha,t)
    # (possibly with an extra term  + phi_n (alpha,t) ).
    #
    # This program determines optimal values of the q nonlinear parameters
    # alpha and the n linear parameters c, given observations y at m
    # different values of the "time" t and given evaluation of phi and
    # derivatives of phi.
    #
    # On Input:
    #
    #   t    1-d array containing the m "times" (independent variables).
    #   y    1-d array containing the m observations.
    #   w    1-d array containing the m weights used in the least squares
    #        fit.  We minimize the norm of the weighted residual
    #        vector r, where, for i=0:m-1,
    #
    #                r(i) = w(i) * (y(i) - eta(alpha, c, t(i))).
    #
    #                Therefore, w(i) should be set to 1 divided by
    #                the standard deviation in the measurement y(i).
    #                If this number is unknown, set w(i) = 1.
    #   alpha 1-d array of length q with initial estimates of the parameters alpha.
    #   n            number of linear parameters c
    #   ada          a function described below.
    #   bounds       If supplied, must be a tuple of both lower and upper
    #   (Optional)   bounds for all q  of the parameters alpha. See documentation
    #                of scipy.optimize.least_squares for examples.
    #   **kwargs     Any keyword argument taken by scipy.optimize.least_squares
    #   (Optional)   other than bounds can be passed here. See
    #                scipy.optimize.least_squares documentation for possible
    #                arguments.
    #
    # On Output:
    #
    #  alpha   length q    Estimates of the nonlinear parameters.
    #  c       length n    Estimates of the linear parameters.
    #  wresid  length m    Weighted residual vector, with i-th component
    #                      w(i) * (y(i) - eta(alpha, c, t(i))).
    #  wresid_norm         Norm of wresid.
    #  y_est   length m    The model estimates = eta(alpha, c, t(i)))
    #                **************************************************
    #                *                C a u t i o n:                  *
    #                *   The theory that makes statistical            *
    #                *   diagnostics useful is derived for            *
    #                *   linear regression, with no upper- or         *
    #                *   lower-bounds on variables.                   *
    #                *   The relevance of these quantities to our     *
    #                *   nonlinear model is determined by how well    *
    #                *   the linearized model (Taylor series model)   *
    #                *         eta(alpha_true, c_true)                *
    #                *            +  Phi * (c  - c_true)              *
    #                *            + dPhi * (alpha - alpha_true)       *
    #                *   fits the data in the neighborhood of the     *
    #                *   true values for alpha and c, where Phi       *
    #                *   and dPhi contain the partial derivatives     *
    #                *   of the model with respect to the c and       *
    #                *   alpha parameters, respectively, and are      *
    #                *   defined in ada.                              *
    #                **************************************************
    #
    #  CorMx:  (n+q) x (n+q)
    #                This is the estimated correlation matrix for the
    #                parameters.  The linear parameters c are ordered
    #                first, followed by the nonlinear parameters alpha.
    #  std_dev_param: length n+q
    #                This vector contains the estimate of the standard
    #                deviation for each parameter.
    #                The k-th element is the square root of the k-th main
    #                diagonal element of the covariance matrix CovMatrix
    #
    # ---------------------------------------------------------------
    # Specification of the function ada, which computes information
    # related to Phi:
    #
    #   Phi,dPhi,Ind = ada(alpha)
    #
    #     This function computes Phi and dPhi.
    #
    #     On Input:
    #
    #     alpha    length q  contains the current value of the alpha parameters.
    #
    #     Note:  If more input arguments are needed, call with a lambda.
    #            For example, if the input arguments to ada are alpha and t,
    #            then before calling varpro, initialize t and call varpro with
    #            "lambda alpha = None: ada(alpha,t)"
    #
    #     On Output:
    #
    #     Phi      m x n1   where Phi(i,j) = phi_j(alpha,t_i).
    #                       (n1 = n if there is no extra term;
    #                        n1 = n+1 if an extra term is used for a nonlinear
    #                        term with no linear coefficient)
    #     dPhi     m x p    where the columns contain partial derivative
    #                       information for Phi and p is the number of
    #                       columns in Ind
    #                       Use numerical differentiation if analytical derivatives
    #                       not available
    #     Ind      2 x p    Column k of dPhi contains the partial
    #                       derivative of Phi_j with respect to alpha_i,
    #                       evaluated at the current value of alpha,
    #                       where j = Ind(0,k) and i = Ind(1,k).
    #                       Columns of dPhi that are always zero, independent
    #                       of alpha, need not be stored.
    #     Example:     If  phi_0 is a function of alpha_1 and alpha_2,
    #                  and phi_1 is a function of alpha_0 and alpha_1, then
    #                  we can set
    #                          Ind = [ 0 0 1 1
    #                                  1 2 0 1 ]
    #                  In this case, the p=4 columns of dPhi contain
    #                          d phi_0 / d alpha_1,
    #                          d phi_0 / d alpha_2,
    #                          d phi_1 / d alpha_0,
    #                          d phi_1 / d alpha_1,
    #                  evaluated at each t_i.
    #                  There are no restrictions on how the columns of
    #                  dPhi are ordered, as long as Ind correctly specifies
    #                  the ordering.
    #
    # ---------------------------------------------------------------
    #
    #  Note that another non-linear least squares solver can be substituted
    #  for the scipy one used. Also, the method "dogbox" can be replaced
    #  by one of the others available in the scipy routine and default
    #  tolerances in that routine can also be replaced.
    #
    #  Any linear parameters that require upper or lower bounds should be put in
    #  alpha, not c, and treated non-linearly. (This should rarely be needed.)
    #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    class RankError(Exception):
        """Raised when linear parameters are ill-determined."""

        def __init__(self, rank, num_c, linear_params, nonlin_params):
            message = """
The linear parameters are not well-determined.
The rank of the matrix in the subproblem is %d,
which is less than the number of linear parameters (%d).

Linear parameters:
    %s
Nonlinear parameters:
    %s""" % (
                rank,
                num_c,
                " ".join(map(str, linear_params)),
                " ".join(map(str, nonlin_params)),
            )
            super().__init__(message)

    class MaxIterationError(Exception):
        """Raised when maximum number of iterations is reached
        during nonlinear solve."""

        def __init__(self, linear_params, nonlin_params):
            message = """
Max iterations reached without convergence.

Linear parameters:
    %s
Nonlinear parameters:
    %s""" % (
                " ".join(map(str, linear_params)),
                " ".join(map(str, nonlin_params)),
            )
            super().__init__(message)

    m = len(y)
    m1 = len(w)

    if m1 != m:
        raise Exception("y, and w must be vectors of the same length")

    if len(alpha.shape) > 1:
        raise Exception(
            "alpha must be a 1d vector containing initial guesses for nonlinear parameters"
        )
    q = len(alpha)
    if q == 0:
        raise Exception("No nonlinear parameters: use linear least squares")

    if bounds is not None:
        if type(bounds) is not tuple:
            raise Exception("bounds must be omitted or supplied as a tuple")
        q1 = len(bounds)
        if q1 != 2:
            raise Exception("must specify both lower and upper bounds")
    # rely on scipy least_squares for further checking of bounds
    else:
        bounds = (-np.inf, np.inf)  # default for scipy least_squares

    if verbose:
        print("\n-------------------")
        print("VARPRO is beginning.")

    W = spdiags(w, 0, m, m)  # convert w from 1-d to 2-d array

    Phi, dPhi, Ind = ada(alpha)
    m1, n1 = Phi.shape

    m2, n2 = dPhi.shape
    ell, n3 = Ind.shape
    if np.logical_and((m1 != m2), (m2 > 0)):
        raise Exception(
            "In user function ada: Phi and dPhi must have the same number of rows."
        )

    if np.logical_or((n1 < n), (n1 > n + 1)):
        raise Exception(
            "In user function ada: The number of columns in Phi must be n or n+1."
        )

    if np.logical_and((n2 > 0), (ell != 2)):
        raise Exception("In user function ada: Ind must have two rows.")

    if np.logical_and((n2 > 0), (n2 != n3)):
        raise Exception(
            "In user function ada: dPhi and Ind must have the same number of columns."
        )

    def formJacobian(alpha, Phi, dPhi):
        U, S, V = np.linalg.svd(W * Phi)
        if n >= 1:
            s = S  # S is a vector in Python, not a matrix
        else:  # no linear parameters
            if len(Ind) == 0:
                Jacobian = []
            else:
                Jacobian = np.zeros((len(y), len(alpha)))
                Jacobian[:, Ind[2, :]] = -W * dPhi
            c = []
            y_est = Phi
            wresid = W * (y - y_est)
            myrank = 1
            return Jacobian, c, wresid, y_est, myrank

        # tol = m * sys.float_info.epsilon
        tol = m * np.finfo(float).eps
        myrank = sum(s > tol * s[0])

        s = s[np.arange(myrank)]
        if myrank < n:
            if verbose:
                print("Warning from VARPRO:")
                print("   The linear parameters are currently not well-determined.")
                print("   The rank of the matrix in the subproblem is ", myrank)
                print("   which is less than the no. of linear parameters,", n)

        yuse = y
        if n < n1:
            yuse = y - Phi[:, n1]
        temp = np.ndarray.flatten(
            np.transpose(U[:, np.arange(myrank)]).dot(W.dot(yuse))
        )
        c = (temp / s).dot(V[np.arange(myrank)])
        y_est = Phi[:, np.arange(n)].dot(c)
        wresid = W * (yuse - y_est)
        if n < n1:
            y_est = y_est + Phi[:, n1]

        if len(dPhi) == 0:
            Jacobian = []
            return Jacobian, c, wresid, y_est, myrank
        WdPhi = W * dPhi
        WdPhi_r = wresid.dot(WdPhi)
        T2 = np.zeros((n1, q))
        ctemp = c
        if n1 > n:  # not checked that this is correct!
            ctemp = np.array([ctemp], [1])

        Jac1 = np.zeros((m, q))
        for j in np.arange(q):
            range = np.where(Ind[1, :] == j)[0]
            indrows = Ind[0, range]
            Jac1[:, j] = WdPhi[:, range].dot(ctemp[indrows])
            T2[indrows, j] = WdPhi_r[range]

        Jac1 = U[:, np.arange(myrank, m)].dot(
            np.transpose(U[:, np.arange(myrank, m)]).dot(Jac1)
        )
        T2 = np.diag(1 / s[np.arange(myrank)]).dot(
            V[np.arange(myrank)].dot(T2[np.arange(n), :])
        )
        Jac2 = U[:, np.arange(myrank)].dot(T2)
        Jacobian = -(Jac1 + Jac2)

        return Jacobian, c, wresid, y_est, myrank  # end of formJacobian

    def f_lsq(alpha_trial):
        Phi_trial, dPhi_trial, Ind = ada(alpha_trial)
        Jacobian, c, wr_trial, y_est, myrank = formJacobian(
            alpha_trial, Phi_trial, dPhi_trial
        )
        return wr_trial, Jacobian, Phi_trial, dPhi_trial, y_est, myrank

    # end of f_lsq

    class Func_jacobian:
        # Computes function and Jacobian in a single routine for efficiency,
        # but supplies them as separate functions

        def __init__(self, xold):
            self.x = xold
            self.fun_jac(xold)

        def fun(self, x):
            if not np.array_equal(self.x, x):
                self.x = x
                self.fun_jac(x)
            return self.f

        def jac(self, x):
            if not np.array_equal(self.x, x):
                self.x = x
                self.fun_jac(x)
            return self.j

        def fun_jac(self, x):
            wr_trial, Jacobian, Phi_trial, dPhi_trial, y_est, myrank = f_lsq(x)
            self.f = wr_trial
            self.j = Jacobian

    fj = Func_jacobian(alpha)
    result = least_squares(
        lambda z: fj.fun(z),
        alpha,
        lambda z: fj.jac(z),
        bounds,
        ftol=ftol,
        gtol=gtol,
        xtol=xtol,
        max_nfev=max_nfev,
        **kwargs
    )
    Phi, dPhi, Ind = ada(result.x)
    Jacobian, c, wresid, y_est, myrank = formJacobian(result.x, Phi, dPhi)
    if result.status == 0:  # maximum number of nonlinear fit iterations was exceeded
        raise MaxIterationError(c, result.x)
    if myrank < n:  # linear parameters are ill-determined
        raise RankError(myrank, n, c, result.x)
    if verbose:
        print("residual_norm", result.cost)
        print("gradient norm", result.optimality)
        print("nfev = ", result.nfev)
        print("njev = ", result.njev)
        print("status = ", result.message)

    wresid_norm = np.linalg.norm(wresid)

    xx, pp = dPhi.shape
    J = np.zeros((m, q))
    for kk in np.arange(pp):
        j = Ind[0, kk]
        i = Ind[1, kk]
        if j > n:
            J[:, i] = J[:, i] + dPhi[:, kk]
        else:
            J[:, i] = J[:, i] + c[j] * dPhi[:, kk]
    Mat = W.dot(np.concatenate((Phi[:, np.arange(n)], J), axis=1))
    Qj, Rj, Pj = linalg.qr(Mat, mode="economic", pivoting=True)
    T2 = linalg.solve_triangular(Rj, (np.identity(Rj.shape[0])))
    sigma2 = wresid_norm * wresid_norm / (m - n - q)
    CovMx = sigma2 * T2.dot(np.transpose(T2))
    CovMatrix = np.empty(Rj.shape)
    tempMat = np.empty(Rj.shape)
    tempMat[:, Pj] = CovMx  # Undo the pivoting permutation
    CovMatrix[Pj, :] = tempMat
    d = 1 / np.sqrt(np.diag(CovMatrix))
    D = spdiags(d, 0, n + q, n + q)
    CorMx = D * CovMatrix * D
    std_dev_params = np.sqrt(np.diag(CovMatrix))

    if verbose:
        print("VARPRO Results:")
        print("linear parameters ", c)
        print("nonlinear parameters ", result.x)
        print("std_dev_params = ", std_dev_params)
        print("wresid_norm = ", wresid_norm)
        print("Corr. Matrix =\n", CorMx)
        print("VARPRO is finished.")
        print("-------------------\n")

    return (
        result.x,
        c,
        wresid,
        wresid_norm,
        y_est,
        CorMx,
        std_dev_params,
        result.message,
        result.success,
    )  # end of varpro
