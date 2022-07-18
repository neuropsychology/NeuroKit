import numpy as np


def write_csv(data, filename, parts=None, **kwargs):
    """**Write data to multiple csv files**

    Split the data into multiple CSV files. You can then re-create them as follows:


    Parameters
    ----------
    data : list
        List of dictionaries.
    filename : str
        Name of the CSV file (without the extension).
    parts : int
        Number of parts to split the data into.

    Returns
    -------
    None

    Example
    --------
    Save big file in parts

    .. ipython:: python

      import pandas as pd
      import neurokit2 as nk

      # Split data into multiple files
      # nk.write_csv(data, 'C:/Users/.../data', parts=6)

    Read the files back

    .. ipython:: python

      # Iterate through 6-parts and concatenate the pieces
      # data_all = pd.concat(
      #     [pd.read_csv(f"data_part{i}.csv") for i in range(1, 7)],
      #     axis=0,
      # )

    """
    if isinstance(parts, int):
        # Add column to identify parts
        data["__Part__"] = np.repeat(range(parts), np.ceil(len(data) / parts))[0 : len(data)]
        for i, part in data.groupby("__Part__"):
            part.drop(["__Part__"], axis=1).to_csv(filename + f"_part{i + 1}.csv", **kwargs)
    else:
        data.to_csv(filename, **kwargs)
