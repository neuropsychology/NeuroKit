# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import detrend

from ..signal import signal_filter
from ..signal import signal_interpolate


class Rsp(object):
    """
    Object for calculation and display of signal-specific statistics.

    Parameters
    ----------
    signal : 1d array
        The respiration signal.
    sfreq : int, optional
        Sampling frequency (Hz). The default is 1000.

    Attributes
    ----------
    signal, sfreq : see Parameters
    peaks : 1d array
        The inhalation peaks.
    troughs : 1d array
        The exhalation troughs
    period : 1d array
        Instantaneous peak to peak difference in milliseconds.
    rate : 1d array
        Instantaneous peak to peak difference in breath per minute.
    amplitude : 1d array
        Instantaneous breathing amplitude based on vertical distance of peaks
        to the preceding troughs.
    period_mean : float
        Mean of period.
    rate_mean : float
        Mean of rate.
    amplitude_mean : float
        Mean of amplitude.

    """

    def __init__(self, signal, sfreq=1000):

        super(Rsp, self).__init__()

        self.signal = signal
        self.sfreq = sfreq

        self.peaks = None
        self.troughs = None

        self.period = None
        self.rate = None
        self.amplitude = None

        self.period_mean = None
        self.rate_mean = None
        self.amplitude_mean = None

        # Compute peaks and troughs.
        self._find_extrema()

        # Compute instantaneous statistics.
        self._calculate_instant_stats()

        # Compute summary statistics.
        self._calculate_summary_stats()

    #######################################################################
    # Methods exposed to the user (i.e., mentioned in public documentation).
    # Note that the user could be given more control over these methods by
    # adding arguments.
    #######################################################################
    def summary_stats(self):
        """Convenience-function to retrieve summary statistics.

        Returns
        -------
        summary_stats : dict
            Contains all summary statistics accessible with the keys
            "mean_period", "mean_rate", and "mean_amplitude".

        """
        summary_stats = {"mean_period": self.period_mean,
                         "mean_rate": self.rate_mean,
                         "mean_amplitude": self.amplitude_mean}
        return summary_stats

    def instantaneous_stats(self):
        """Convenience-function to retrieve instantaneous statistics.

        Returns
        -------
        inst_stats : dict
            Contains all instantaneous statistics accessible with the keys
            "period", "rate", and "amplitude".

        """
        inst_stats = {"period": self.period,
                      "rate": self.rate,
                      "amplitude": self.amplitude}
        return inst_stats

    def extrema(self):
        """Convenience-function to retrieve the signals extrema.

        Returns
        -------
        extrema : dict
            Contains the signal's extrema accessible with the keys "peaks", and
            "troughs".

        """
        extrema = {"peaks": self.peaks,
                   "troughs": self.troughs}
        return extrema

    def plot_summary(self):
        """Plot object overview.

        Displays the signal, its extrema, as well as the instantaneous
        statistics.

        """
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1, sharex=True)
        ax0.set_title("Signal and Breathing Extrema")
        ax0.plot(self.signal)
        ax0.scatter(self.peaks, self.signal[self.peaks])
        ax0.scatter(self.troughs, self.signal[self.troughs])
        ax1.set_title("Breathing Period (based on Inhalation Peaks")
        ax1.plot(self.period)
        ax2.set_title("Breathing Rate (based on Inhalation Peaks")
        ax2.plot(self.rate)
        ax3.set_title("Breathing Amplitude")
        ax3.plot(self.amplitude)
        plt.show()

    ###############################################
    # Internal methods (not publically documented).
    ###############################################
    def _find_extrema(self):

        sig = self.signal
        # Detrend and lowpass-filter the signal to be able to reliably detect
        # zero crossings in raw signal.
        sig = detrend(sig, type="linear")
        sig_filt = signal_filter(sig, sampling_rate=self.sfreq, highcut=2, method="butterworth")

        # Detect zero crossings (note that these are zero crossings in the raw
        # signal, not in its gradient).
        greater = sig_filt > 0
        smaller = sig_filt < 0
        risex = np.where(np.bitwise_and(smaller[:-1], greater[1:]))[0]
        fallx = np.where(np.bitwise_and(greater[:-1], smaller[1:]))[0]

        if risex[0] < fallx[0]:
            startx = "rise"
        elif fallx[0] < risex[0]:
            startx = "fall"

        allx = np.concatenate((risex, fallx))
        allx.sort(kind="mergesort")

        # Find extrema by searching minima between falling zero crossing and
        # rising zero crossing, and searching maxima between rising zero
        # crossing and falling zero crossing.
        extrema = []
        for i in range(len(allx) - 1):

            # Determine whether to search for minimum or maximum.
            if startx == "rise":
                if (i + 1) % 2 != 0:
                    argextreme = np.argmax
                else:
                    argextreme = np.argmin
            elif startx == "fall":
                if (i + 1) % 2 != 0:
                    argextreme = np.argmin
                else:
                    argextreme = np.argmax

            # Get the two zero crossings between which the extreme will be
            # searched.
            beg = allx[i]
            end = allx[i + 1]

            extreme = argextreme(sig_filt[beg:end])
            extrema.append(beg + extreme)

        extrema = np.asarray(extrema)

        # Only consider those extrema that have a minimum vertical distance
        # to their direct neighbor, i.e., define outliers in absolute amplitude
        # difference between neighboring extrema.
        vertdiff = np.abs(np.diff(sig_filt[extrema]))
        avgvertdiff = np.mean(vertdiff)
        minvert = np.where(vertdiff > avgvertdiff * 0.3)[0]
        extrema = extrema[minvert]

        # Make sure that the alternation of peaks and troughs is unbroken. If
        # alternation of sign in extdiffs is broken, remove the extrema that
        # cause the breaks.
        amps = sig_filt[extrema]
        extdiffs = np.sign(np.diff(amps))
        extdiffs = np.add(extdiffs[0:-1], extdiffs[1:])
        removeext = np.where(extdiffs != 0)[0] + 1
        extrema = np.delete(extrema, removeext)
        amps = np.delete(amps, removeext)

        # To be able to consistently calculate breathing amplitude, make
        # sure that the extrema always start with a trough and end with a peak,
        # since breathing amplitude will be defined as vertical distance
        # between each peak and the preceding trough. Note that this also
        # ensures that the number of peaks and troughs is equal.
        if amps[0] > amps[1]:
            extrema = np.delete(extrema, 0)
        if amps[-1] < amps[-2]:
            extrema = np.delete(extrema, -1)
        peaks = extrema[1::2]
        troughs = extrema[0:-1:2]

        self.peaks = peaks
        self.troughs = troughs

    def _calculate_instant_stats(self):

        # Calculate period in msec, based on horizontal peak to peak
        # difference. Make sure that period has the same number of elements as
        # peaks (important for interpolation later) by prepending the mean of
        # all periods.
        period = np.ediff1d(self.peaks, to_begin=0) / self.sfreq
        period[0] = np.mean(period)
        rate = 60 / period
        # TODO: normalize amplitude?
        amplitude = self.peaks - self.troughs

        # Interpolate all statistics to length of the breathing signal.
        nsamps = len(self.signal)
        self.period = signal_interpolate(self.peaks, x_axis=period, length=nsamps)
        self.rate = signal_interpolate(self.peaks, x_axis=rate, length=nsamps)
        self.amplitude = signal_interpolate(self.peaks, x_axis=amplitude, length=nsamps)

    def _calculate_summary_stats(self):

        # This is the place to add the calculation of additional statistics.
        self.period_mean = np.mean(self.period)
        self.rate_mean = np.mean(self.rate)
        self.amplitude_mean = np.mean(self.amplitude)
