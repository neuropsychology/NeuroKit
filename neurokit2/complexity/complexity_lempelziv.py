# -*- coding: utf-8 -*-
import numpy as np


def complexity_lempelziv(signal, threshold="median", normalize=True):

    # method to convert signal by
    if threshold == "median":
        threshold = np.median(signal)
    elif threshold == "mean":
        threshold = np.mean(signal)
    
    p_seq = signal.copy()
    # convert signal into binary sequence
    for index, value in enumerate(signal):
        if value < threshold:
            p_seq[index] = 0
        else:
            p_seq[index] = 1
    p_seq = p_seq.astype(int)

    # pre-set variables
    complexity = 1
    n = len(p_seq)
    pointer = 0
    current_prefix_len = 1
    current_substring_len = 1
    final_substring_len = 1    

    # iterate over sequence
    while current_prefix_len + current_substring_len <= n:
        if (p_seq[pointer + current_substring_len - 1] == p_seq[current_prefix_len + current_substring_len - 1]):
            current_substring_len += 1
        else:
            final_substring_len = max(current_substring_len, final_substring_len)
            pointer += 1
            if pointer == current_prefix_len:
                complexity += 1
                current_prefix_len = current_prefix_len + final_substring_len
                current_substring_len = 1
                pointer = 0
                final_substring_len = 1
            else:
                current_substring_len = 1
    
    if current_substring_len != 1:
        complexity += 1
    
    if normalize is True:
        complexity = _complexity_lempelziv_normalize(p_seq, complexity)

    return complexity
            

def _complexity_lempelziv_normalize(sequence, complexity):
    
    n = len(sequence)
    upper_bound = n / np.log2(n)
    complexity = complexity / upper_bound

    return complexity / upper_bound
