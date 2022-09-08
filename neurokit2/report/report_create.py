# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from ..ppg import ppg_plot

# TODO: comments

def report_create(filename="myreport.html", signals=None, sampling_rate=1000):
    description, ref = describe_processing(
        processing_kw=dict(sampling_rate=sampling_rate)
    )
    summary_table = summary_table_create(signals)
    fig = ppg_plot(signals, sampling_rate=sampling_rate, static=False)
    contents = [description, summary_table, fig, ref]
    html_combine(contents=contents, filename=filename)


def describe_processing(processing_kw={"sampling_rate": 1000}):
    # TODO: automate references?
    description = "<br><b>Description</b><br>The raw PPG signal "
    if processing_kw["sampling_rate"] is not None:
        description += "(sampled at " + str(processing_kw["sampling_rate"]) + " Hz)"
    description += (
        " was cleaned with a bandpass filter ([0.5, 8], butterworth 3rd order). <br>"
        + "The peak detection was carried out using the Elgendi et al. (2013) method."
    )
    ref = "<br><b>References</b><br>"
    ref += (
        "Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection in <br>"
        + "Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions. <br>"
        + "PLoS ONE 8(10): e76585. doi:10.1371/journal.pone.0076585."
    )
    return description, ref


def summary_table_create(signals):
    summary = {}
    summary["PPG_Rate_Mean"] = np.mean(signals["PPG_Rate"])
    summary["PPG_Rate_SD"] = np.std(signals["PPG_Rate"])
    summary_table = pd.DataFrame(summary, index=[0])  # .transpose()
    print(summary_table.to_markdown(index=None))
    return (
        "<br> <b>Summary table</b> <br>" + summary_table.to_html(index=None)
    )


def html_combine(contents=[], filename="myreport.html"):
    # https://stackoverflow.com/questions/59868987/plotly-saving-multiple-plots-into-a-single-html
    with open(filename, "w") as page:
        page.write("<html><head></head><body>" + "\n")
        for content in contents:
            if isinstance(content, str):
                inner_html = content
            else:
                inner_html = content.to_html().split("<body>")[1].split("</body>")[0]
            page.write(inner_html)
            page.write("<br>")
        page.write("</body></html>" + "\n")
