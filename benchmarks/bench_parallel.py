"""Benchmark: Sequential vs Parallel performance for NeuroKit2 optimizations.

Run: .venv/bin/python benchmarks/bench_parallel.py
"""

import time
import warnings

import numpy as np


warnings.filterwarnings("ignore")

import neurokit2 as nk  # noqa: E402


def bench(label, fn_seq, fn_par, n_runs=3):
    """Run a benchmark comparing sequential vs parallel execution."""
    # Warmup
    fn_seq()
    fn_par()

    # Sequential
    times_seq = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn_seq()
        times_seq.append(time.perf_counter() - t0)

    # Parallel
    times_par = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn_par()
        times_par.append(time.perf_counter() - t0)

    seq_avg = np.mean(times_seq)
    par_avg = np.mean(times_par)
    speedup = seq_avg / par_avg if par_avg > 0 else float("inf")

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Sequential : {seq_avg:.3f}s  (best: {min(times_seq):.3f}s)")
    print(f"  Parallel   : {par_avg:.3f}s  (best: {min(times_par):.3f}s)")
    print(f"  Speedup    : {speedup:.2f}x")
    return seq_avg, par_avg, speedup


def main():
    print("NeuroKit2 Parallel Benchmarks")
    print(f"CPU cores: {__import__('multiprocessing').cpu_count()}")
    results = {}

    # =========================================================================
    # 1. bio_process — 5 signal types, 5min at 1000Hz (heavy workload)
    # =========================================================================
    print("\nPreparing bio_process benchmark (5min signals @ 1000Hz, 5 modalities)...")
    dur, sr = 300, 1000
    ecg = nk.ecg_simulate(duration=dur, sampling_rate=sr)
    rsp = nk.rsp_simulate(duration=dur, sampling_rate=sr)
    eda = nk.eda_simulate(duration=dur, sampling_rate=sr, scr_number=10)
    emg = nk.emg_simulate(duration=dur, sampling_rate=sr, burst_number=10)
    ppg = nk.ppg_simulate(duration=dur, sampling_rate=sr, heart_rate=70)

    results["bio_process"] = bench(
        "bio_process (5 modalities, 5min @ 1000Hz)",
        fn_seq=lambda: nk.bio_process(ecg=ecg, rsp=rsp, eda=eda, emg=emg, ppg=ppg, sampling_rate=sr, parallel=False),
        fn_par=lambda: nk.bio_process(ecg=ecg, rsp=rsp, eda=eda, emg=emg, ppg=ppg, sampling_rate=sr, parallel=True),
    )

    # =========================================================================
    # 2. ecg_findpeaks ProMAC — 5min ECG at 1000Hz
    # =========================================================================
    print("\nPreparing ProMAC benchmark (5min ECG @ 1000Hz)...")
    ecg_long = nk.ecg_simulate(duration=dur, sampling_rate=sr)
    ecg_clean = nk.ecg_clean(ecg_long, sampling_rate=sr)

    results["promac"] = bench(
        "ecg_findpeaks ProMAC (5min @ 1000Hz, 10 methods)",
        fn_seq=lambda: nk.ecg_findpeaks(ecg_clean, sampling_rate=sr, method="promac", n_jobs=1),
        fn_par=lambda: nk.ecg_findpeaks(ecg_clean, sampling_rate=sr, method="promac", n_jobs=-1),
    )

    # =========================================================================
    # 3. entropy_multiscale — longer signal, more scales
    # =========================================================================
    print("\nPreparing entropy_multiscale benchmark (30s @ 500Hz, 30 scales)...")
    signal_ent = nk.signal_simulate(duration=30, frequency=[5, 12, 40], sampling_rate=500)

    results["entropy_mse"] = bench(
        "entropy_multiscale (30s @ 500Hz, 30 scales)",
        fn_seq=lambda: nk.entropy_multiscale(signal_ent, scale=30, show=False, n_jobs=1),
        fn_par=lambda: nk.entropy_multiscale(signal_ent, scale=30, show=False, n_jobs=-1),
    )

    # =========================================================================
    # 4. HRV nonlinear (DFA thread parallelism) — 8min resting ECG
    # =========================================================================
    print("\nPreparing HRV nonlinear benchmark (8min resting ECG)...")
    data_hrv = nk.data("bio_resting_8min_100hz")
    peaks_hrv, _ = nk.ecg_peaks(data_hrv["ECG"], sampling_rate=100)

    def run_hrv_nonlinear():
        return nk.hrv_nonlinear(peaks_hrv, sampling_rate=100)

    times_hrv = []
    run_hrv_nonlinear()  # warmup
    for _ in range(3):
        t0 = time.perf_counter()
        run_hrv_nonlinear()
        times_hrv.append(time.perf_counter() - t0)

    print(f"\n{'=' * 60}")
    print("  HRV nonlinear (8min ECG, DFA with thread parallelism)")
    print(f"{'=' * 60}")
    print(f"  Time       : {np.mean(times_hrv):.3f}s  (best: {min(times_hrv):.3f}s)")
    print("  (Internal thread parallelism for mono/multifractal DFA)")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Benchmark':<45} {'Seq (s)':>8} {'Par (s)':>8} {'Speedup':>8}")
    print(f"  {'-' * 45} {'-' * 8} {'-' * 8} {'-' * 8}")
    for name, (seq, par, speedup) in results.items():
        print(f"  {name:<45} {seq:>8.3f} {par:>8.3f} {speedup:>7.2f}x")
    print(f"  {'HRV nonlinear (DFA threads)':<45} {np.mean(times_hrv):>8.3f}      -        -")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
