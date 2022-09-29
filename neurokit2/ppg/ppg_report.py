import numpy as np
import pandas as pd

from ..misc.report import html_combine, text_combine
from .ppg_plot import ppg_plot


def ppg_report(file="myreport.html", signals=None, info={"sampling_rate": 1000}):
    """Create report containing description and figures of processing.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      ppg_report(file="myreport.html")

    """

    description, ref = text_combine(info)
    summary_table = ppg_table(signals)
    fig = ppg_plot(signals, sampling_rate=info["sampling_rate"], static=False)
    contents = [description, summary_table, fig, ref]
    html_combine(contents=contents, file=file)


def ppg_table(signals):
    """Create table to summarize statistics of a PPG signal."""

    summary = {}
    # currently only implemented for PPG
    summary["PPG_Rate_Mean"] = np.mean(signals["PPG_Rate"])
    summary["PPG_Rate_SD"] = np.std(signals["PPG_Rate"])
    summary_table = pd.DataFrame(summary, index=[0])  # .transpose()
    print(summary_table.to_markdown(index=None))
    return "<br><b>Summary table</b><br>" + summary_table.to_html(index=None)
