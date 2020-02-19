from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np


import neurokit2 as nk
import ecg_findpeaks2_lib as lib


def evaluate(filenames: List[str], evaluating_fields: List[str]):
    """Evaluate the peak finding functionality.

    Args:
      filenames: List of filename to load. Syntax: `{word}_{samplingrate}.csv`
       Should have an associated annotation file with name `{word}_{samplingrate}_annotation.csv`

    Returns:
      Dict: Result dictionary.

    """
    results = []

    for filename in filenames:
        ecg = np.array(pd.read_csv(filename))[:, 1]
        sampling_rate = int(filename.split('.')[0].split('_')[1])

        filename_annotation = filename.split('.')[0] + '_annotation' + '.' + filename.split('.')[1]
        expecte_res = pd.read_csv(filename_annotation, index_col=0, header=None).transpose()

        peaks = lib.ecg_find_peaks(ecg, sampling_rate)

        for field in evaluating_fields:
            assert field in peaks and field in expecte_res
            if len(peaks[field]) <= len(expecte_res[field]):
                s1 = np.ones(len(expecte_res[field])) * np.nan
                s1[:len(peaks[field])] = peaks[field]
                s2 = np.array(expecte_res[field])
            else:
                raise ValueError('Return more result than annotated. Very strange, better check!')
            results.append((filename, field, sampling_rate, s1, s2))

    return results


def show_result(evaluation_result: List[Tuple[str, str, Any, Any]]) -> None:
    """Human-friendly result display."""
    for res in evaluation_result:
        filename, field, sampling_rate, s1, s2 = res
        diff = (s1 - s2) / sampling_rate * 1000
        s_max = np.max(diff)
        s_min = np.min(diff)
        s_mean = np.mean(diff)
        s_std = np.std(diff)
        print(f"{res[0]:10s}  {res[1]:15s} [stat(ms)] {s_mean:05.2f} +- {s_std:05.2f} max {s_max} min {s_min}")


def main():
    eval_result = evaluate([
        'good_4000.csv'
    ], [
        'qrsoffsets', 'tpeaks', 'ppeaks', 'qrsonsets'
    ])
    show_result(eval_result)


if __name__ == '__main__':
    main()
    
