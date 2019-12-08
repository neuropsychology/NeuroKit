# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from neurokit2.signal import signal_filter, statistics


class Rsp():
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

        # Initialize peaks and troughs.
        self._find_extrema()

        # Initialize instantaneous statistics.
        self._calculate_instant_stats()

        # Initialize summary statistics.
        self._calculate_summary_stats()

    #######################################################################
    # Methods exposed to the user (i.e., mentioned in public documentation).
    # Note that the user could be given more control over these methods by
    # adding arguments.
    #######################################################################
    def summary_stats(self):

        summary_stats = {"mean_period": self.period_mean,
                         "mean_rate": self.rate_mean,
                         "mean_amplitude": self.amplitude_mean}
        return summary_stats

    def instantaneous_stats(self):

        inst_stats = {"period": self.period,
                      "rate": self.rate,
                      "amplitude": self.amplitude}
        return inst_stats

    def extrema(self):

        extrema = {"peaks": self.peaks,
                   "troughs": self.troughs}
        return extrema

    def plot_summary(self):

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
    # Internal methods (not publically documented)-
    ###############################################
    def _find_extrema(self):

        sig = self.signal
        # Detrend and lowpass-filter the signal to be able to reliably detect
        # zero crossings in raw signal.
        sig = detrend(sig, type="linear")
        sig_filt = signal_filter.butter_filter(sig, "lowpass",
                                               sfreq=self.sfreq,
                                               highcut=2)

        # Detect zero crossings (note that these are zero crossings in the raw
        # signal, not in it's gradient).
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

        # Find extrema.
        extrema = []
        for i in range(len(allx) - 1):

            # Determine whether to search for min or max.
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

            beg = allx[i]
            end = allx[i + 1]

            extreme = argextreme(sig_filt[beg:end])
            extrema.append(beg + extreme)

        extrema = np.asarray(extrema)

        # Only consider those extrema that have a minimum vertical difference
        # to their direct neighbor, i.e., define outliers in absolute amplitude
        # difference between neighboring extrema.
        vertdiff = np.abs(np.diff(sig_filt[extrema]))
        avgvertdiff = np.mean(vertdiff)
        minvert = np.where(vertdiff > avgvertdiff * 0.3)[0]
        extrema = extrema[minvert]

        # Make sure that the alternation of peaks and troughs is unbroken: if
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

        # Calculate period, based on horizontal peak to peak difference.
        # Make sure that period has the same number of elements as peaks
        # (important for interpolation later) by prepending the mean of all
        # periods.
        period = np.ediff1d(self.peaks, to_begin=0) / self.sfreq
        period[0] = np.mean(period)

        rate = 60 / period

        # TODO: normalize amplitude?
        amplitude = self.peaks - self.troughs

        # Interpolate all statistics to length of the breathing signal.
        nsamps = len(self.signal)
        self.period = statistics.interp_stats(self.peaks, period, nsamps)
        self.rate = statistics.interp_stats(self.peaks, rate, nsamps)
        self.amplitude = statistics.interp_stats(self.peaks, amplitude, nsamps)

    def _calculate_summary_stats(self):

        # This is the place to add the calculation of additional statistics.
        self.period_mean = np.mean(self.period)
        self.rate_mean = np.mean(self.rate)
        self.amplitude_mean = np.mean(self.amplitude)
