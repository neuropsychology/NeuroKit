import numpy as np

from ..misc import check_random_state


def eeg_simulate(duration=1, length=None, sampling_rate=1000, noise=0.1, random_state=None):
    """**EEG Signal Simulation**

    Simulate an artificial EEG signal. This is a crude implementation based on the MNE-Python raw
    simulation example. Help is needed to improve this function.

    Parameters
    ----------
    duration : int
        Desired recording length in seconds.
    length : int
        The desired length of the signal (in samples).
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    noise : float
        Noise level.
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator. See for ``misc.check_random_state`` for further information.

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      eeg = nk.eeg_simulate(duration=3, sampling_rate=500, noise=0.2)

      @savefig p_eeg_simulate1.png scale=100%
      _ = nk.signal_psd(eeg, sampling_rate=500, show=True, max_frequency=100)
      @suppress
      plt.close()

    """
    # Try loading mne
    try:
        import mne
        import mne.datasets
        import mne.simulation

    except ImportError as e:
        raise ImportError(
            "The 'mne' module is required for this function to run. ",
            "Please install it first (`pip install mne`).",
        ) from e

    # Seed the random generator for reproducible results
    rng = check_random_state(random_state)

    # Generate number of samples automatically if length is unspecified
    if length is None:
        length = duration * sampling_rate
    if duration is None:
        duration = length / sampling_rate

    # Get paths to data
    path = mne.datasets.sample.data_path() / "MEG" / "sample"
    raw_file = path / "sample_audvis_raw.fif"
    fwd_file = path / "sample_audvis-meg-eeg-oct-6-fwd.fif"

    # Load real data as the template
    raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
    raw = raw.set_eeg_reference(projection=True, verbose=False)

    n_dipoles = 4  # number of dipoles to create

    def data_fun(times, n_dipoles=4):
        """Generate time-staggered sinusoids at harmonics of 10Hz"""
        n = 0  # harmonic number
        n_samp = len(times)
        window = np.zeros(n_samp)
        start, stop = [int(ii * float(n_samp) / (2 * n_dipoles)) for ii in (2 * n, 2 * n + 1)]
        window[start:stop] = 1.0
        n += 1
        data = 25e-9 * np.sin(2.0 * np.pi * 10.0 * n * times)
        data *= window
        return data

    times = raw.times[: int(raw.info["sfreq"] * 2)]
    fwd = mne.read_forward_solution(fwd_file, verbose=False)
    stc = mne.simulation.simulate_sparse_stc(
        fwd["src"],
        n_dipoles=n_dipoles,
        times=times,
        data_fun=data_fun,
        random_state=rng,
    )

    # Repeat the source activation multiple times.
    raw_sim = mne.simulation.simulate_raw(raw.info, [stc] * int(np.ceil(duration / 2)), forward=fwd, verbose=False)
    cov = mne.make_ad_hoc_cov(raw_sim.info, std=noise / 1000000)
    raw_sim = mne.simulation.add_noise(raw_sim, cov, iir_filter=[0.2, -0.2, 0.04], verbose=False, random_state=rng)

    # Resample
    raw_sim = raw_sim.resample(sampling_rate, verbose=False)

    # Add artifacts
    # mne.simulation.add_ecg(raw_sim, verbose=False)
    # mne.simulation.add_eog(raw_sim, verbose=False)

    eeg = raw_sim.pick_types(eeg=True, verbose=False).get_data()
    return eeg[0, 0 : int(duration * sampling_rate)]
