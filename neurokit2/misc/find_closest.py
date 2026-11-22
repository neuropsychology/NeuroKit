import numpy as np

from .type_converters import as_vector


def find_closest(closest_to, list_to_search_in, direction="both", strictly=False, return_index=False):
    """**Find the closest number in the array from a given number x**

    Parameters
    ----------
    closest_to : float
        The target number(s) to find the closest of.
    list_to_search_in : list
        The list of values to look in.
    direction : str
        ``"both"`` for smaller or greater, ``"greater"`` for only greater numbers and ``"smaller"`` for the closest smaller.
    strictly : bool
        ``False`` (default) to include equal values, or ``True`` for strictly greater or smaller values.
    return_index : bool
        If ``True``, will return the index of the closest value in the list.

    Returns
    ----------
    closest : int
        The closest number in the array.

    Example
    ----------
    .. ipython:: python

      import neurokit2 as nk

      # Single number
      x = nk.find_closest(1.8, [3, 5, 6, 1, 2])
      x

      y = nk.find_closest(1.8, [3, 5, 6, 1, 2], return_index=True)
      y

      # Vectorized version
      x = nk.find_closest([1.8, 3.6], [3, 5, 6, 1, 2])
      x

    """
    if direction not in ("both", "all", "smaller", "below", "greater", "above"):
        raise ValueError("NeuroKit error: find_closest(): 'direction' must be 'both', 'smaller', or 'greater'.")

    targets_raw = as_vector(closest_to)
    vals_raw = as_vector(list_to_search_in)
    targets = np.asarray(targets_raw, dtype=float)
    vals_f = np.asarray(vals_raw, dtype=float)
    n_targets = len(targets)
    if n_targets == 0:
        return np.array([])

    valid_vals_mask = ~np.isnan(vals_f)
    if not np.any(valid_vals_mask):
        return np.nan if n_targets == 1 else np.full(n_targets, np.nan)

    # Sort valid search values stably and keep only the first occurrence of each distinct value
    # so tie-breaking matches the earliest index in `list_to_search_in`.
    orig_indices = np.flatnonzero(valid_vals_mask)
    vals_valid = vals_f[valid_vals_mask]
    order = np.argsort(vals_valid, kind="stable")
    s_vals = vals_valid[order]
    s_orig = orig_indices[order]
    uniq_mask = np.empty(len(s_vals), dtype=bool)
    uniq_mask[0] = True
    uniq_mask[1:] = s_vals[1:] != s_vals[:-1]
    u_vals = s_vals[uniq_mask]
    u_orig = s_orig[uniq_mask]
    n_unique = len(u_vals)

    valid_t = ~np.isnan(targets)
    t_valid = targets[valid_t]

    if direction in ("smaller", "below"):
        pos = np.searchsorted(u_vals, t_valid, side="left" if strictly else "right") - 1
        ok = pos >= 0
        chosen_pos = np.where(ok, pos, 0)
    elif direction in ("greater", "above"):
        pos = np.searchsorted(u_vals, t_valid, side="right" if strictly else "left")
        ok = pos < n_unique
        chosen_pos = np.where(ok, pos, 0)
    else:
        left_pos = np.searchsorted(u_vals, t_valid, side="left" if strictly else "right") - 1
        right_pos = np.searchsorted(u_vals, t_valid, side="right" if strictly else "left")
        left_ok = left_pos >= 0
        right_ok = right_pos < n_unique
        ok = left_ok | right_ok
        lp = np.clip(left_pos, 0, n_unique - 1)
        rp = np.clip(right_pos, 0, n_unique - 1)
        d_left = np.where(left_ok, t_valid - u_vals[lp], np.inf)
        d_right = np.where(right_ok, u_vals[rp] - t_valid, np.inf)
        choose_left = (d_left < d_right) | ((d_left == d_right) & (u_orig[lp] <= u_orig[rp]))
        chosen_pos = np.where(choose_left, lp, rp)

    matched_orig = u_orig[chosen_pos]
    all_ok = np.zeros(n_targets, dtype=bool)
    all_ok[valid_t] = ok

    if np.all(all_ok):
        if return_index is True:
            return int(matched_orig[0]) if n_targets == 1 else matched_orig
        res = vals_raw[matched_orig]
        return res[0] if n_targets == 1 else res

    if return_index is True:
        out = np.full(n_targets, np.nan, dtype=float)
        out[valid_t] = np.where(ok, matched_orig.astype(float), np.nan)
        return np.nan if n_targets == 1 else out

    out = np.full(n_targets, np.nan, dtype=float)
    out[valid_t] = np.where(ok, vals_f[matched_orig], np.nan)
    return np.nan if n_targets == 1 else out
