# -*- coding: utf-8 -*-
import inspect
import numpy as np
import pandas as pd

from ..ppg import ppg_plot


def report_create(
    filename="myreport.html", signals=None, report_info={"sampling_rate": 1000}
):
    """Create report containing description and figures of processing"""
    description, ref = process_text_combine(report_info)
    summary_table = summary_table_create(signals)
    fig = ppg_plot(signals, sampling_rate=report_info["sampling_rate"], static=False)
    contents = [description, summary_table, fig, ref]
    html_combine(contents=contents, filename=filename)


def process_text_combine(report_info):
    """Reformat dictionary describing processing methods as strings to be inserted into HTML file"""
    description = "<br><b>Description</b><br>"
    for key in ["text_cleaning", "text_peaks"]:
        if key in report_info.keys():
            description += report_info[key] + "<br>"
    ref = "<br><b>References</b><br>"

    if "references" in report_info.keys():
        for reference in report_info["references"]:
            ref += reference + "<br>"
    return description, ref


def summary_table_create(signals):
    """Create table to summarize statistics of a PPG signal"""
    summary = {}
    # currently only implemented for PPG
    summary["PPG_Rate_Mean"] = np.mean(signals["PPG_Rate"])
    summary["PPG_Rate_SD"] = np.std(signals["PPG_Rate"])
    summary_table = pd.DataFrame(summary, index=[0])  # .transpose()
    print(summary_table.to_markdown(index=None))
    return "<br> <b>Summary table</b> <br>" + summary_table.to_html(index=None)


def html_combine(contents=[], filename="myreport.html"):
    """Combine figures and text in a single HTML document"""
    # https://stackoverflow.com/questions/59868987/plotly-saving-multiple-plots-into-a-single-html
    with open(filename, "w") as page:
        page.write("<html><head></head><body>" + "\n")
        for content in contents:
            if isinstance(content, str):
                inner_html = content
            else:
                # assume the content is an interactive plotly figure and export to HTML
                inner_html = content.to_html().split("<body>")[1].split("</body>")[0]
            page.write(inner_html)
            page.write("<br>")
        page.write("</body></html>" + "\n")


def get_default_args(func):
    """Get the default values of a function's arguments"""
    # https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
