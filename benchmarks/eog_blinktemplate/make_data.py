import neurokit2 as nk
#import neurokit as nk1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import statsmodels.api as sm
import biosppy
import scipy.signal

import nolds
import sklearn.neighbors
import sklearn.decomposition



signals = [pd.read_csv("../../data/eog_100hz.csv")["vEOG"].values]



def fit_gamma(x, size, loc, a, scale):
    """
    >>> x = np.arange(100)
    >>> nk.signal_plot(fit_gamma(x, 0, 1, 2, 2, 3))
    """
    x = nk.rescale(x, to=[0, 20])
    gamma = scipy.stats.gamma.pdf(x, a=a, loc=loc, scale=scale)
    y = size * gamma
    return y

def fit_bateman(x, size=1, loc=0, t1=0.75, t2=2):
    """
    >>> x = np.arange(100)
    >>> nk.signal_plot(fit_bateman(x, 0, 1, 1, 0.75, 2))
    """
    x = nk.rescale(x, to=[-loc, 10])
    bateman = np.exp(-x / t2) - np.exp(-x / t1)
    bateman[np.where(x < 0)] = 0
    y = size * bateman
    return y

def fit_scr(x, size, time_peak, rise, decay1, decay2):
    """
    >>> x = np.arange(100)
    >>> nk.signal_plot(fit_scr(x, 0, 1, 3, 0.7, 3, 5))
    """
    x = nk.rescale(x, to=[0, 20])
    gt = np.exp(-((x - time_peak) ** 2) / (2 * rise ** 2))
    ht = np.exp(-x / decay1) + np.exp(-x / decay2)

    ft = np.convolve(gt, ht)
    ft = ft[0 : len(x)]
    y = size * ft
    return y


def fit_poly(x, coefs):
    """
    >>> x = np.arange(100)
    >>> nk.signal_plot(fit_poly(x, [30, -60, 25, 8, -3, 0]))
    """
    x = np.linspace(0, 1, num=len(x))
    y = np.polyval(coefs, x)
    return y

def params_poly(y, order=4):
    x = np.linspace(0, 1, num=len(y))
    coefs = np.polyfit(x, y, order)
    return coefs



for i in range(len(signals)):
    signal = signals[i]

    cleaned = nk.eog_clean(signal, sampling_rate=100, method='neurokit')
    blinks = nk.eog_findpeaks(cleaned, sampling_rate=100, method="mne")

    events = nk.epochs_create(cleaned, blinks, sampling_rate=100, epochs_start=-0.4, epochs_end=0.6)
    events = nk.epochs_to_array(events)  # Convert to 2D array

    x = np.linspace(0, 100, num=len(events))

    p_gamma = np.full((events.shape[1], 4), np.nan)
    p_bateman = np.full((events.shape[1], 4), np.nan)
    p_scr = np.full((events.shape[1], 5), np.nan)
    p_poly = np.full((events.shape[1], 5), np.nan)

    for i in range(events.shape[1]):
        events[:, i] = nk.rescale(events[:, i], to=[0, 1])  # Reshape to 0-1 scale
        try:
            p_gamma[i, :], _ = scipy.optimize.curve_fit(fit_gamma, x, events[:, i], p0=[1, 2, 2, 3])
            p_bateman[i, :], _ = scipy.optimize.curve_fit(fit_bateman, x, events[:, i], p0=[1, 1, 0.75, 2])
            p_scr[i, :], _ = scipy.optimize.curve_fit(fit_scr, x, events[:, i], p0=[1, 3, 0.7, 3, 5])
            p_poly[i, :] = params_poly(events[:, i], order=4)
        except RuntimeError:
            pass


# Visualize for one particpant
cleaned = nk.eog_clean(signals[0], sampling_rate=100, method='neurokit')
blinks = nk.eog_findpeaks(cleaned, sampling_rate=100, method="mne")
events = nk.epochs_create(cleaned, blinks, sampling_rate=100, epochs_start=-0.4, epochs_end=0.6)
events = nk.epochs_to_array(events)
for i in range(events.shape[1]):
        events[:, i] = nk.rescale(events[:, i], to=[0, 1])  # Reshape to 0-1 scale

x = np.linspace(0, 100, num=len(events))
template_gamma = fit_gamma(x, *np.nanmedian(p_gamma, axis=0))
template_bateman = fit_bateman(x, *np.nanmedian(p_bateman, axis=0))
template_scr = fit_scr(x, *np.nanmedian(p_scr, axis=0))
template_poly = fit_poly(x, np.nanmedian(p_poly, axis=0))


plt.plot(events, linewidth=0.25, color="black")
plt.plot(template_gamma, linewidth=2, linestyle='-', color="red", label='Gamma')
plt.plot(template_bateman, linewidth=2, linestyle='-', color="blue", label='Bateman')
plt.plot(template_scr, linewidth=2, linestyle='-', color="orange", label='SCR')
plt.plot(template_poly, linewidth=2, linestyle='-', color="green", label='Polynomial')
plt.legend(loc="upper right")











# =============================================================================
# Re-run with filtering out dissimilar
# =============================================================================
optimal1 = np.nanmedian(p_scr, axis=0)
for i in range(len(signals)):
    signal = signals[i]

    cleaned = nk.eog_clean(signal, sampling_rate=100, method='neurokit')
    blinks = nk.eog_findpeaks(cleaned, sampling_rate=100, method="mne")

    events = nk.epochs_create(cleaned, blinks, sampling_rate=100, epochs_start=-0.4, epochs_end=0.6)
    events = nk.epochs_to_array(events)  # Convert to 2D array

    rmse = np.full(events.shape[1], np.nan)
    for i in range(events.shape[1]):
        events[:, i] = nk.rescale(events[:, i], to=[0, 1])  # Reshape to 0-1 scale
        rmse[i] = nk.fit_rmse(events[:, i], template_scr)

    events = events[:, rmse < 0.4]

    x = np.linspace(0, 100, num=len(events))
    p_scr = np.full((events.shape[1], 5), np.nan)
    for i in range(events.shape[1]):
        try:
            p_scr[i, :], _ = scipy.optimize.curve_fit(fit_scr, x, events[:, i], p0=optimal1)
        except RuntimeError:
            pass


# Visualize for one particpant
cleaned = nk.eog_clean(signals[0], sampling_rate=100, method='neurokit')
blinks = nk.eog_findpeaks(cleaned, sampling_rate=100, method="mne")
events = nk.epochs_create(cleaned, blinks, sampling_rate=100, epochs_start=-0.4, epochs_end=0.6)
events = nk.epochs_to_array(events)
for i in range(events.shape[1]):
        events[:, i] = nk.rescale(events[:, i], to=[0, 1])  # Reshape to 0-1 scale

x = np.linspace(0, 100, num=len(events))
template_scr2 = fit_scr(x, *np.nanmedian(p_scr, axis=0))

plt.plot(events, linewidth=0.25, color="black")
plt.plot(template_scr, linewidth=2, linestyle='-', color="orange", label='SCR')
plt.plot(template_scr2, linewidth=2, linestyle='-', color="red", label='SCR (optimized)')
plt.legend(loc="upper right")

print(np.nanmedian(p_scr, axis=0))