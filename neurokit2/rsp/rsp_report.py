import numpy as np
import pandas as pd

from ..misc.report import html_save, text_combine
from .rsp_plot import rsp_plot


def rsp_report(file="myreport.html", signals=None, info={"sampling_rate": 1000}):
    """**RSP Reports**

    Create report containing description and figures of processing.
    This function is meant to be used via the `rsp_process()` function.

    Parameters
    ----------
    file : str
        Name of the file to save the report to. Can also be ``"text"`` to simply print the text in
        the console.
    signals : pd.DataFrame
        A DataFrame of signals. Usually obtained from :func:`.rsp_process`.
    info : dict
        A dictionary containing the information of peaks and the signals' sampling rate. Usually
        obtained from :func:`.rsp_process`.

    Returns
    -------
    str
        The report as a string.

    See Also
    --------
    ppg_process

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      ppg = nk.rsp_simulate(duration=10, sampling_rate=200)
      signals, info = nk.rsp_process(rsp, sampling_rate=200, report="console_only")

    """

    description, ref = text_combine(info)
    table_html, table_md = rsp_table(signals)

    # Print text in the console
    for key in [k for k in info.keys() if "text_" in k]:
        print(info[key] + "\n")

    print(table_md)

    print("\nReferences")
    for s in info["references"]:
        print("- " + s)

    # Make figures
    fig = '<h2 style="background-color: #FB661C">Visualization</h1>'
    fig += (
        rsp_plot(signals, sampling_rate=info["sampling_rate"], static=False)
        .to_html()
        .split("<body>")[1]
        .split("</body>")[0]
    )

    # Save report
    if ".html" in file:
        print(f"The report has been saved to {file}")
        contents = [description, table_html, fig, ref]
        html_save(contents=contents, file=file)


# =============================================================================
# Internals
# =============================================================================
def rsp_table(signals):
    """Create table to summarize statistics of a RSP signal."""

    # TODO: add more features
    summary = {}

    summary["RSP_Rate_Mean"] = np.mean(signals["RSP_Rate"])
    summary["RSP_Rate_SD"] = np.std(signals["RSP_Rate"])
    summary_table = pd.DataFrame(summary, index=[0])

    # Make HTML and Markdown versions
    html = '<h2 style="background-color: #D60574">Summary table</h1>' + summary_table.to_html(
        index=None
    )

    try:
        md = summary_table.to_markdown(index=None)
    except ImportError:
        md = summary_table  # in case printing markdown export fails
    return html, md
