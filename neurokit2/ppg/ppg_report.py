import numpy as np
import pandas as pd

from ..misc.report import html_save, text_combine
from .ppg_plot import ppg_plot


def ppg_report(file="myreport.html", signals=None, info={"sampling_rate": 1000}):
    """**PPG Reports**

    Create report containing description and figures of processing.
    This function is meant to be used via the `ppg_process()` function.

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      ppg = nk.ppg_simulate(duration=10, sampling_rate=200, heart_rate=70)
      signals, info = nk.ppg_process(ppg, sampling_rate=200, report="console_only")

    """

    description, ref = text_combine(info)
    table_html, table_md = ppg_table(signals)

    # Print text in the console
    for key in ["text_cleaning", "text_peaks"]:
        print(info[key] + "\n")

    print(table_md)

    print("\nReferences")
    for s in info["references"]:
        print("- " + s)

    # Make figures
    fig = '<h2 style="background-color: #FB661C">Visualization</h1>'
    fig += (
        ppg_plot(signals, sampling_rate=info["sampling_rate"], static=False)
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
def ppg_table(signals):
    """Create table to summarize statistics of a PPG signal."""

    # TODO: add more features
    summary = {}

    summary["PPG_Rate_Mean"] = np.mean(signals["PPG_Rate"])
    summary["PPG_Rate_SD"] = np.std(signals["PPG_Rate"])
    summary_table = pd.DataFrame(summary, index=[0])  # .transpose()

    # Make HTML and Markdown versions
    html = (
        '<h2 style="background-color: #D60574">Summary table</h1>'
        + summary_table.to_html(index=None)
    )

    try:
        md = summary_table.to_markdown(index=None)
    except ImportError:
        md = summary_table  # in case printing markdown export fails
    return html, md
