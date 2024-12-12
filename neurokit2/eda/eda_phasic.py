# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.signal

from ..signal import signal_filter, signal_resample, signal_smooth


def eda_phasic(eda_signal, sampling_rate=1000, method="highpass", **kwargs):
    """**Electrodermal Activity (EDA) Decomposition into Phasic and Tonic Components**

    Decompose the Electrodermal Activity (EDA) into two components, namely **Phasic** and
    **Tonic**, using different methods including cvxEDA (Greco, 2016) or Biopac's Acqknowledge
    algorithms.

    * **High-pass filtering**: Method implemented in Biopac's Acqknowledge. The raw EDA signal
      is passed through a high pass filter with a cutoff frequency of 0.05 Hz
      (cutoff frequency can be adjusted by the ``cutoff`` argument).
    * **Median smoothing**: Method implemented in Biopac's Acqknowledge. The raw EDA signal is
      passed through a median value smoothing filter, which removes areas of rapid change. The
      phasic component is then calculated by subtracting the smoothed signal from the original.
      This method is computationally intensive and the processing time depends on the smoothing
      factor, which can be controlled by the as ``smoothing_factor`` argument, set by default to
      ``4`` seconds. Higher values will produce results more rapidly.
    * **cvxEDA**: Convex optimization approach to EDA processing (Greco, 2016). Requires the
      ``cvxopt`` package (`> 1.3.0.1 <https://github.com/neuropsychology/NeuroKit/issues/781>`_) to
      be installed.
    * **SparsEDA**: Sparse non-negative deconvolution (Hernando-Gallego et al., 2017).

    .. warning::

      sparsEDA was newly added thanks to
      `this implementation <https://github.com/yskong224/SparsEDA-python>`_. Help is needed to
      double-check it, improve it and make it more concise and efficient. Also, sometimes it errors
      for unclear reasons. Please help.


    Parameters
    ----------
    eda_signal : Union[list, np.array, pd.Series]
        The raw EDA signal.
    sampling_rate : int
        The sampling frequency of raw EDA signal (in Hz, i.e., samples/second). Defaults to 1000Hz.
    method : str
        The processing pipeline to apply. Can be one of ``"cvxEDA"``, ``"smoothmedian"``,
        ``"highpass"``. Defaults to ``"highpass"``.
    **kwargs : dict
        Additional arguments to pass to the specific method.

    Returns
    -------
    DataFrame
        DataFrame containing the ``"Tonic"`` and the ``"Phasic"`` components as columns.

    See Also
    --------
    eda_simulate, eda_clean, eda_peaks, eda_process, eda_plot


    Examples
    ---------
    **Example 1**: Methods comparison.

    .. ipython:: python

      import neurokit2 as nk

      # Simulate EDA signal
      eda_signal = nk.eda_simulate(duration=30, scr_number=5, drift=0.1)

      # Decompose using different algorithms
      # cvxEDA = nk.eda_phasic(eda_signal, method='cvxeda')
      smoothMedian = nk.eda_phasic(eda_signal, method='smoothmedian')
      highpass = nk.eda_phasic(eda_signal, method='highpass')
      sparse = nk.eda_phasic(eda_signal, method='smoothmedian')

      # Extract tonic and phasic components for plotting
      t1, p1 = smoothMedian["EDA_Tonic"].values, smoothMedian["EDA_Phasic"].values
      t2, p2 = highpass["EDA_Tonic"].values, highpass["EDA_Phasic"].values
      t3, p3 = sparse["EDA_Tonic"].values, sparse["EDA_Phasic"].values

      # Plot tonic
      @savefig p_eda_phasic1.png scale=100%
      nk.signal_plot([t1, t2, t3], labels=["SmoothMedian", "Highpass", "Sparse"])
      @suppress
      plt.close()

      # Plot phasic
      @savefig p_eda_phasic2.png scale=100%
      nk.signal_plot([p1, p2, p3], labels=["SmoothMedian", "Highpass", "Sparse"])
      @suppress
      plt.close()

    **Example 2**: Real data.

    .. ipython:: python

      eda_signal = nk.data("bio_eventrelated_100hz")["EDA"]
      data = nk.eda_phasic(nk.standardize(eda_signal), sampling_rate=100)
      data["EDA_Raw"] = eda_signal

      @savefig p_eda_phasic2.png scale=100%
      nk.signal_plot(data, standardize=True)
      @suppress
      plt.close()

    References
    -----------
    * Greco, A., Valenza, G., & Scilingo, E. P. (2016). Evaluation of CDA and CvxEDA Models. In
      Advances in Electrodermal Activity Processing with Applications for Mental Health (pp. 35-43).
      Springer International Publishing.
    * Greco, A., Valenza, G., Lanata, A., Scilingo, E. P., & Citi, L. (2016). cvxEDA: A convex
      optimization approach to electrodermal activity processing. IEEE Transactions on Biomedical
      Engineering, 63(4), 797-804.
    * Hernando-Gallego, F., Luengo, D., & Artés-Rodríguez, A. (2017). Feature extraction of
      galvanic skin responses by nonnegative sparse deconvolution. IEEE journal of biomedical and
      shealth informatics, 22(5), 1385-1394.

    """
    method = method.lower()  # remove capitalised letters
    if method in ["cvxeda", "convex"]:
        tonic, phasic = _eda_phasic_cvxeda(eda_signal, sampling_rate)
    elif method in ["sparse", "sparseda"]:
        tonic, phasic = _eda_phasic_sparsEDA(eda_signal, sampling_rate)
    elif method in ["median", "smoothmedian"]:
        tonic, phasic = _eda_phasic_mediansmooth(eda_signal, sampling_rate, **kwargs)
    elif method in ["neurokit", "highpass", "biopac", "acqknowledge"]:
        tonic, phasic = _eda_phasic_highpass(eda_signal, sampling_rate, **kwargs)
    else:
        raise ValueError(
            "NeuroKit error: eda_phasic(): 'method' should be one of "
            "'cvxeda', 'median', 'smoothmedian', 'neurokit', 'highpass', "
            "'biopac', 'acqknowledge'."
        )

    return pd.DataFrame({"EDA_Tonic": tonic, "EDA_Phasic": phasic})


# =============================================================================
# Acqknowledge
# =============================================================================
def _eda_phasic_mediansmooth(eda_signal, sampling_rate=1000, smoothing_factor=4):
    """One of the two methods available in biopac's acqknowledge (https://www.biopac.com/knowledge-base/phasic-eda-
    issue/)"""
    size = smoothing_factor * sampling_rate
    tonic = signal_smooth(eda_signal, kernel="median", size=size)
    phasic = eda_signal - tonic

    return np.array(tonic), np.array(phasic)


def _eda_phasic_highpass(eda_signal, sampling_rate=1000, cutoff=0.05):
    """One of the two methods available in biopac's acqknowledge (https://www.biopac.com/knowledge-base/phasic-eda-
    issue/)"""
    phasic = signal_filter(eda_signal, sampling_rate=sampling_rate, lowcut=cutoff, method="butter")
    tonic = signal_filter(eda_signal, sampling_rate=sampling_rate, highcut=cutoff, method="butter")

    return tonic, phasic


# =============================================================================
# cvxEDA (Greco et al., 2016)
# =============================================================================
def _eda_phasic_cvxeda(
    eda_signal,
    sampling_rate=1000,
    tau0=2.0,
    tau1=0.7,
    delta_knot=10.0,
    alpha=8e-4,
    gamma=1e-2,
    solver=None,
    reltol=1e-9,
):
    """A convex optimization approach to electrodermal activity processing (CVXEDA).

    This function implements the cvxEDA algorithm described in "cvxEDA: a
    Convex Optimization Approach to Electrodermal Activity Processing" (Greco et al., 2015).

    Parameters
    ----------
       eda_signal : list or array
           raw EDA signal array.
       sampling_rate : int
           Sampling rate (samples/second).
       tau0 : float
           Slow time constant of the Bateman function.
       tau1 : float
           Fast time constant of the Bateman function.
       delta_knot : float
           Time between knots of the tonic spline function.
       alpha : float
           Penalization for the sparse SMNA driver.
       gamma : float
           Penalization for the tonic spline coefficients.
       solver : bool
           Sparse QP solver to be used, see cvxopt.solvers.qp
       reltol : float
           Solver options, see http://cvxopt.org/userguide/coneprog.html#algorithm-parameters

    Returns
    -------
    Dataframe
        Contains EDA tonic and phasic signals.

    """
    # Try loading cvx
    try:
        import cvxopt
    except ImportError:
        raise ImportError(
            "NeuroKit error: eda_decompose(): the 'cvxopt' module is required for this method to run. ",
            "Please install it first (`pip install cvxopt`).",
        )

    # Internal functions
    def _cvx(m, n):
        return cvxopt.spmatrix([], [], [], (m, n))

    frequency = 1 / sampling_rate

    n = len(eda_signal)
    eda = cvxopt.matrix(eda_signal)

    # bateman ARMA model
    a1 = 1.0 / min(tau1, tau0)  # a1 > a0
    a0 = 1.0 / max(tau1, tau0)
    ar = np.array(
        [
            (a1 * frequency + 2.0) * (a0 * frequency + 2.0),
            2.0 * a1 * a0 * frequency**2 - 8.0,
            (a1 * frequency - 2.0) * (a0 * frequency - 2.0),
        ]
    ) / ((a1 - a0) * frequency**2)
    ma = np.array([1.0, 2.0, 1.0])

    # matrices for ARMA model
    i = np.arange(2, n)
    A = cvxopt.spmatrix(np.tile(ar, (n - 2, 1)), np.c_[i, i, i], np.c_[i, i - 1, i - 2], (n, n))
    M = cvxopt.spmatrix(np.tile(ma, (n - 2, 1)), np.c_[i, i, i], np.c_[i, i - 1, i - 2], (n, n))

    # spline
    delta_knot_s = int(round(delta_knot / frequency))
    spl = np.r_[np.arange(1.0, delta_knot_s), np.arange(delta_knot_s, 0.0, -1.0)]  # order 1
    spl = np.convolve(spl, spl, "full")
    spl /= max(spl)
    # matrix of spline regressors
    i = (
        np.c_[np.arange(-(len(spl) // 2), (len(spl) + 1) // 2)]
        + np.r_[np.arange(0, n, delta_knot_s)]
    )
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl), 1))
    p = np.tile(spl, (nB, 1)).T
    valid = (i >= 0) & (i < n)
    B = cvxopt.spmatrix(p[valid], i[valid], j[valid])

    # trend
    C = cvxopt.matrix(np.c_[np.ones(n), np.arange(1.0, n + 1.0) / n])
    nC = C.size[1]

    # Solve the problem:
    # .5*(M*q + B*l + C*d - eda)^2 + alpha*sum(A, 1)*p + .5*gamma*l'*l
    # s.t. A*q >= 0

    cvxopt.solvers.options.update({"reltol": reltol, "show_progress": False})
    if solver == "conelp":
        # Use conelp
        G = cvxopt.sparse(
            [
                [-A, _cvx(2, n), M, _cvx(nB + 2, n)],
                [_cvx(n + 2, nC), C, _cvx(nB + 2, nC)],
                [_cvx(n, 1), -1, 1, _cvx(n + nB + 2, 1)],
                [_cvx(2 * n + 2, 1), -1, 1, _cvx(nB, 1)],
                [_cvx(n + 2, nB), B, _cvx(2, nB), cvxopt.spmatrix(1.0, range(nB), range(nB))],
            ]
        )
        h = cvxopt.matrix([_cvx(n, 1), 0.5, 0.5, eda, 0.5, 0.5, _cvx(nB, 1)])
        c = cvxopt.matrix(
            [(cvxopt.matrix(alpha, (1, n)) * A).T, _cvx(nC, 1), 1, gamma, _cvx(nB, 1)]
        )
        res = cvxopt.solvers.conelp(c, G, h, dims={"l": n, "q": [n + 2, nB + 2], "s": []})
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = cvxopt.sparse(
            [
                [Mt * M, Ct * M, Bt * M],
                [Mt * C, Ct * C, Bt * C],
                [Mt * B, Ct * B, Bt * B + gamma * cvxopt.spmatrix(1.0, range(nB), range(nB))],
            ]
        )
        f = cvxopt.matrix(
            [(cvxopt.matrix(alpha, (1, n)) * A).T - Mt * eda, -(Ct * eda), -(Bt * eda)]
        )
        res = cvxopt.solvers.qp(
            H,
            f,
            cvxopt.spmatrix(-A.V, A.I, A.J, (n, len(f))),
            cvxopt.matrix(0.0, (n, 1)),
            kktsolver="chol2",
        )
    cvxopt.solvers.options.clear()

    tonic_splines = res["x"][-nB:]
    drift = res["x"][n : n + nC]
    tonic = B * tonic_splines + C * drift
    q = res["x"][:n]
    phasic = M * q

    # Return tonic and phasic components
    return np.array(tonic)[:, 0], np.array(phasic)[:, 0]


# =============================================================================
# sparsEDA (Hernando-Gallego et al., 2017)
# =============================================================================


def _eda_phasic_sparsEDA(
    eda_signal, sampling_rate=8, epsilon=0.0001, Kmax=40, Nmin=5 / 4, rho=0.025
):
    """ "
    Credits go to:
    - https://github.com/fhernandogallego/sparsEDA (Matlab original implementation)
    - https://github.com/yskong224/SparsEDA-python (Python implementation)

    Parameters
    ----------
    epsilon
        step remainder
    maxIters
        maximum number of LARS iterations
    dmin
        maximum distance between sparse reactions
    rho
        minimun threshold of sparse reactions

    Returns
    -------
    driver
        driver responses, tonic component
    SCL
        low component
    MSE
        reminder of the signal fitting
    """

    dmin = Nmin * sampling_rate
    original_length = len(eda_signal)  # Used for resampling at the end

    # Exceptions
    # if len(eda_signal) / sampling_rate < 80:
    #     raise AssertionError("Signal not enough large. longer than 80 seconds")

    if np.sum(np.isnan(eda_signal)) > 0:
        raise AssertionError("Signal contains NaN")

    # Resample to 8 Hz
    eda_signal = signal_resample(eda_signal, sampling_rate=sampling_rate, desired_sampling_rate=8)
    new_sr = 8

    # Preprocessing
    signalAdd = np.zeros(len(eda_signal) + (20 * new_sr) + (60 * new_sr))
    signalAdd[0 : 20 * new_sr] = eda_signal[0]
    signalAdd[20 * new_sr : 20 * new_sr + len(eda_signal)] = eda_signal
    signalAdd[20 * new_sr + len(eda_signal) :] = eda_signal[-1]

    Nss = len(eda_signal)
    Ns = len(signalAdd)
    b0 = 0

    pointerS = 20 * new_sr
    pointerE = pointerS + Nss
    # signalRs = signalAdd[pointerS:pointerE]

    # overlap Save
    durationR = 70
    Lreg = int(20 * new_sr * 3)
    L = 10 * new_sr
    N = durationR * new_sr
    T = 6

    Rzeros = np.zeros([N + L, Lreg * 5])
    srF = new_sr * np.array([0.5, 0.75, 1, 1.25, 1.5])

    for j in range(0, len(srF)):
        t_rf = np.arange(0, 10 + 1e-10, 1 / srF[j])  # 10 sec
        taus = np.array([0.5, 2, 1])
        rf_biexp = np.exp(-t_rf / taus[1]) - np.exp(-t_rf / taus[0])
        rf_est = taus[2] * rf_biexp
        rf_est = rf_est / np.sqrt(np.sum(rf_est**2))

        rf_est_zeropad = np.zeros(len(rf_est) + (N - len(rf_est)))
        rf_est_zeropad[: len(rf_est)] = rf_est
        Rzeros[0:N, j * Lreg : (j + 1) * Lreg] = scipy.linalg.toeplitz(
            rf_est_zeropad, np.zeros(Lreg)
        )

    R0 = Rzeros[0:N, 0 : 5 * Lreg]
    R = np.zeros([N, T + Lreg * 5])
    R[0:N, T:] = R0

    # SCL
    R[0:Lreg, 0] = np.linspace(0, 1, Lreg)
    R[0:Lreg, 1] = -np.linspace(0, 1, Lreg)
    R[int(Lreg / 3) : Lreg, 2] = np.linspace(0, 2 / 3, int((2 * Lreg) / 3))
    R[int(Lreg / 3) : Lreg, 3] = -np.linspace(0, 2 / 3, int((2 * Lreg) / 3))
    R[int(2 * Lreg / 3) : Lreg, 4] = np.linspace(0, 1 / 3, int(Lreg / 3))
    R[int(2 * Lreg / 3) : Lreg, 5] = -np.linspace(0, 1 / 3, int(Lreg / 3))
    Cte = np.sum(R[:, 0] ** 2)
    R[:, 0:6] = R[:, 0:6] / np.sqrt(Cte)

    # Loop
    cutS = 0
    cutE = N
    slcAux = np.zeros(Ns)
    driverAux = np.zeros(Ns)
    resAux = np.zeros(Ns)
    aux = 0

    while cutE < Ns:
        aux = aux + 1
        signalCut = signalAdd[cutS:cutE]

        if b0 == 0:
            b0 = signalCut[0]

        signalCutIn = signalCut - b0
        beta, _, _, _, _, _ = lasso(R, signalCutIn, new_sr, Kmax, epsilon)

        signalEst = (np.matmul(R, beta) + b0).reshape(-1)

        # remAout = (signalCut - signalEst).^2;
        # res2 = sum(remAout(20*sampling_rate+1:(40*sampling_rate)));
        # res3 = sum(remAout(40*sampling_rate+1:(60*sampling_rate)));

        remAout = (signalCut - signalEst) ** 2
        res2 = np.sum(remAout[20 * new_sr : 40 * new_sr])
        res3 = np.sum(remAout[40 * new_sr : 60 * new_sr])

        jump = 1
        if res2 < 1:
            jump = 2
            if res3 < 1:
                jump = 3

        SCL = np.matmul(R[:, 0:6], beta[0:6, :]) + b0

        SCRline = beta[6:, :]

        SCRaux = np.zeros([Lreg, 5])
        SCRaux[:] = SCRline.reshape([5, Lreg]).transpose()
        driver = SCRaux.sum(axis=1)

        b0 = np.matmul(R[jump * 20 * new_sr - 1, 0:6], beta[0:6, :]) + b0

        driverAux[cutS : cutS + (jump * 20 * new_sr)] = driver[0 : jump * new_sr * 20]
        slcAux[cutS : cutS + (jump * 20 * new_sr)] = SCL[
            0 : jump * new_sr * 20
        ].reshape(-1)
        resAux[cutS : cutS + (jump * 20 * new_sr)] = remAout[0 : jump * new_sr * 20]
        cutS = cutS + jump * 20 * new_sr
        cutE = cutS + N

    SCRaux = driverAux[pointerS:pointerE]
    SCL = slcAux[pointerS:pointerE]
    #MSE = resAux[pointerS:pointerE]

    # PP
    ind = np.argwhere(SCRaux > 0).reshape(-1)
    scr_temp = SCRaux[ind]
    ind2 = np.argsort(scr_temp)[::-1]
    scr_ord = scr_temp[ind2]
    scr_fin = [scr_ord[0]]
    ind_fin = [ind[ind2[0]]]

    for i in range(1, len(ind2)):
        if np.all(np.abs(ind[ind2[i]] - ind_fin) >= dmin):
            scr_fin.append(scr_ord[i])
            ind_fin.append(ind[ind2[i]])

    driver = np.zeros(len(SCRaux))
    driver[np.array(ind_fin)] = np.array(scr_fin)

    scr_max = scr_fin[0]
    threshold = rho * scr_max
    driver[driver < threshold] = 0

    # Resample back to original sampling rate
    SCR = eda_signal-SCL
    SCR = signal_resample(SCR, desired_length=original_length)
    SCL = signal_resample(SCL, desired_length=original_length)
    return SCL, SCR


def lasso(R, s, sampling_rate, maxIters, epsilon):
    N = len(s)
    W = R.shape[1]

    OptTol = -10
    solFreq = 0
    resStop2 = 0.0005
    lmbdaStop = 0
    zeroTol = 1e-5

    x = np.zeros(W)
    x_old = np.zeros(W)
    iter = 0

    c = np.matmul(R.transpose(), s.reshape(-1, 1)).reshape(-1)

    lmbda = np.max(c)

    if lmbda < 0:
        raise Exception(
            "y is not expressible as a non-negative linear combination of the columns of X"
        )

    newIndices = np.argwhere(np.abs(c - lmbda) < zeroTol).flatten()

    collinearIndices = []
    beta = []
    duals = []
    res = s

    if (lmbdaStop > 0 and lmbda < lmbdaStop) or ((epsilon > 0) and (np.linalg.norm(res) < epsilon)):
        activationHist = []
        numIters = 0

    R_I = []
    activeSet = []

    for j in range(0, len(newIndices)):
        iter = iter + 1
        R_I, flag = updateChol(R_I, N, W, R, 1, activeSet, newIndices[j], zeroTol)
        activeSet.append(newIndices[j])
    activationHist = activeSet.copy()

    # Loop
    done = 0
    while done == 0:
        if len(activationHist) == 4:
            lmbda = np.max(c)
            newIndices = np.argwhere(np.abs(c - lmbda) < zeroTol).flatten()
            activeSet = []
            for j in range(0, len(newIndices)):
                iter = iter + 1
                R_I, flag = updateChol(R_I, N, W, R, 1, activeSet, newIndices[j], zeroTol)
                activeSet.append(newIndices[j])
            [activationHist.append(ele) for ele in activeSet]
        else:
            lmbda = c[activeSet[0]]

        dx = np.zeros(W)

        if len(np.array([R_I]).flatten()) == 1:
            z = scipy.linalg.solve(
                R_I.reshape([-1, 1]),
                np.sign(c[np.array(activeSet).flatten()].reshape(-1, 1)),
                transposed=True,
                lower=False,
            )
        else:
            z = scipy.linalg.solve(
                R_I,
                np.sign(c[np.array(activeSet).flatten()].reshape(-1, 1)),
                transposed=True,
                lower=False,
            )

        if len(np.array([R_I]).flatten()) == 1:
            dx[np.array(activeSet).flatten()] = scipy.linalg.solve(
                R_I.reshape([-1, 1]), z, transposed=False, lower=False
            )
        else:
            dx[np.array(activeSet).flatten()] = scipy.linalg.solve(
                R_I, z, transposed=False, lower=False
            ).flatten()

        v = np.matmul(
            R[:, np.array(activeSet).flatten()], dx[np.array(activeSet).flatten()].reshape(-1, 1)
        )
        ATv = np.matmul(R.transpose(), v).flatten()

        gammaI = np.inf
        removeIndices = []

        inactiveSet = np.arange(0, W)
        if len(np.array(activeSet).flatten()) > 0:
            inactiveSet[np.array(activeSet).flatten()] = -1

        if len(np.array(collinearIndices).flatten()) > 0:
            inactiveSet[np.array(collinearIndices).flatten()] = -1

        inactiveSet = np.argwhere(inactiveSet >= 0).flatten()

        if len(inactiveSet) == 0:
            gammaIc = 1
            newIndices = []
        else:
            epsilon = 1e-12
            gammaArr = (lmbda - c[inactiveSet]) / (1 - ATv[inactiveSet] + epsilon)

            gammaArr[gammaArr < zeroTol] = np.inf
            gammaIc = np.min(gammaArr)
            # Imin = np.argmin(gammaArr)
            newIndices = inactiveSet[(np.abs(gammaArr - gammaIc) < zeroTol)]

        gammaMin = min(gammaIc, gammaI)

        x = x + gammaMin * dx
        res = res - gammaMin * v.flatten()
        c = c - gammaMin * ATv

        if (
            ((lmbda - gammaMin) < OptTol)
            or ((lmbdaStop > 0) and (lmbda <= lmbdaStop))
            or ((epsilon > 0) and (np.linalg.norm(res) <= epsilon))
        ):
            newIndices = []
            removeIndices = []
            done = 1

            if (lmbda - gammaMin) < OptTol:
                # print(lmbda-gammaMin)
                pass
        if np.linalg.norm(res[0 : sampling_rate * 20]) <= resStop2:
            done = 1
            if np.linalg.norm(res[sampling_rate * 20 : sampling_rate * 40]) <= resStop2:
                done = 1
                if np.linalg.norm(res[sampling_rate * 40 : sampling_rate * 60]) <= resStop2:
                    done = 1

        if gammaIc <= gammaI and len(newIndices) > 0:
            for j in range(0, len(newIndices)):
                iter = iter + 1
                R_I, flag = updateChol(
                    R_I, N, W, R, 1, np.array(activeSet).flatten(), newIndices[j], zeroTol
                )

                if flag:
                    collinearIndices.append(newIndices[j])
                else:
                    activeSet.append(newIndices[j])
                    activationHist.append(newIndices[j])

        if gammaI <= gammaIc:
            for j in range(0, len(removeIndices)):
                iter = iter + 1
                col = np.argwhere(np.array(activeSet).flatten() == removeIndices[j]).flatten()

                R_I = downdateChol(R_I, col)
                activeSet.pop(col)
                collinearIndices = []

            x[np.array(removeIndices).flatten()] = 0
            activationHist.append(-removeIndices)
        if iter >= maxIters:
            done = 1

        if len(np.argwhere(x < 0).flatten()) > 0:
            x = x_old.copy()
            done = 1
        else:
            x_old = x.copy()

        if done or ((solFreq > 0) and not (iter % solFreq)):
            beta.append(x)
            duals.append(v)
    numIters = iter
    return np.array(beta).reshape(-1, 1), numIters, activationHist, duals, lmbda, res


def updateChol(R_I, n, N, R, explicitA, activeSet, newIndex, zeroTol):
    # global opts_tr, zeroTol

    flag = 0

    newVec = R[:, newIndex]

    if len(activeSet) == 0:
        R_I0 = np.sqrt(np.sum(newVec**2))
    else:
        if explicitA:
            if len(np.array([R_I]).flatten()) == 1:
                p = scipy.linalg.solve(
                    np.array(R_I).reshape(-1, 1),
                    np.matmul(R[:, activeSet].transpose(), R[:, newIndex]),
                    transposed=True,
                    lower=False,
                )
            else:
                p = scipy.linalg.solve(
                    R_I,
                    np.matmul(R[:, activeSet].transpose(), R[:, newIndex]),
                    transposed=True,
                    lower=False,
                )

        else:
            # Original matlab code:

            # Global var for linsolve functions..
            # optsUT = True
            # opts_trUT = True
            # opts_trTRANSA = True
            # AnewVec = feval(R,2,n,length(activeSet),newVec,activeSet,N);
            # p = linsolve(R_I,AnewVec,opts_tr);

            # Translation by chatGPT-3, might be wrong
            AnewVec = np.zeros((n, 1))
            for i in range(len(activeSet)):
                AnewVec += R[2, :, activeSet[i]] * newVec[i]
            p = scipy.linalg.solve(R_I, AnewVec, transposed=True, lower=False)

        q = np.sum(newVec**2) - np.sum(p**2)
        if q <= zeroTol:
            flag = 1
            R_I0 = R_I.copy()
        else:
            if len(np.array([R_I]).shape) == 1:
                R_I = np.array([R_I]).reshape(-1, 1)
            # print(R_I)
            R_I0 = np.zeros([np.array(R_I).shape[0] + 1, R_I.shape[1] + 1])
            R_I0[0 : R_I.shape[0], 0 : R_I.shape[1]] = R_I
            R_I0[0 : R_I.shape[0], -1] = p
            R_I0[-1, -1] = np.sqrt(q)

    return R_I0, flag


def downdateChol(R, j):
    # global opts_tr, zeroTol

    def planerot(x):
        # http://statweb.stanford.edu/~susan/courses/b494/index/node30.html
        if x[1] != 0:
            r = np.linalg.norm(x)
            G = np.zeros(len(x) + 2)
            G[: len(x)] = x / r
            G[-2] = -x[1] / r
            G[-1] = x[0] / r
        else:
            G = np.eye(2)
        return G, x

    R1 = np.zeros([R.shape[0], R.shape[1] - 1])
    R1[:, :j] = R[:, :j]
    R1[:, j:] = R[:, j + 1 :]
    # m = R1.shape[0]
    n = R1.shape[1]

    for k in range(j, n):
        p = np.array([k, k + 1])
        G, R[p, k] = planerot(R[p, k])
        if k < n:
            R[p, k + 1 : n] = G * R[p, k + 1 : n]

    return R[:n, :]
