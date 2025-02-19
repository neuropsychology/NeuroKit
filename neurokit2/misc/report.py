# -*- coding: utf-8 -*-
import inspect

import matplotlib
import numpy as np
import pandas as pd


def create_report(file="myreport.html", signals=None, info={"sampling_rate": 1000}, fig=None):
    """**Reports**

    Create report containing description and figures of processing.
    This function is meant to be used via the :func:`.rsp_process` or :func:`.ppg_process`
    functions.

    Parameters
    ----------
    file : str
        Name of the file to save the report to. Can also be ``"text"`` to simply print the text in
        the console.
    signals : pd.DataFrame
        A DataFrame of signals. Usually obtained from :func:`.rsp_process`, :func:`.ppg_process`, or
            :func:`.emg_process`.
    info : dict
        A dictionary containing the information of peaks and the signals' sampling rate. Usually
        obtained from :func:`.rsp_process` or :func:`.ppg_process`.
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        A figure containing the processed signals. Usually obtained from :func:`.rsp_plot`,
        :func:`.ppg_plot`, or :func:`.emg_plot`.

    Returns
    -------
    str
        The report as a string.

    See Also
    --------
    rsp_process, ppg_process, emg_process

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      rsp = nk.rsp_simulate(duration=30, sampling_rate=200, random_state=0)
      signals, info = nk.rsp_process(rsp, sampling_rate=200, report="text")

    """

    description, ref = text_combine(info)
    table_html, table_md = summarize_table(signals)

    # Print text in the console
    for key in [k for k in info.keys() if "text_" in k]:
        print(info[key] + "\n")

    print(table_md)

    print("\nReferences")
    for s in info["references"]:
        print("- " + s)

    # Save report
    if ".html" in file:
        # Make figures
        fig_html = '<h2 style="background-color: #FB661C">Visualization</h1>'
        fig_html += fig_to_html(fig)
        print(f"The report has been saved to {file}")
        contents = [description, table_html, fig_html, ref]
        html_save(contents=contents, file=file)


def summarize_table(signals):
    """Create table to summarize statistics of a signal."""

    # TODO: add more features
    summary = {}

    rate_cols = [col for col in signals.columns if "Rate" in col]
    if len(rate_cols) > 0:
        rate_col = rate_cols[0]
        summary[rate_col + "_Mean"] = np.mean(signals[rate_col])
        summary[rate_col + "_SD"] = np.std(signals[rate_col])
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
    else:
        return "", ""


def text_combine(info):
    """Reformat dictionary describing processing methods as strings to be inserted into HTML file."""
    preprocessing = '<h2 style="background-color: #FB1CF0">Preprocessing</h1>'
    for key in ["text_cleaning", "text_peaks", "text_quality"]:
        if key in info.keys():
            preprocessing += info[key] + "<br>"
    ref = '<h2 style="background-color: #FBB41C">References</h1>'
    if "references" in info.keys():
        ref += "\n <ul> \n"
        for reference in info["references"]:
            ref += "<li>" + reference + "</li>" + "\n"
        ref += "\n </ul> \n"
    return preprocessing, ref


def fig_to_html(fig):
    """Convert a figure to HTML."""
    if isinstance(fig, str):
        return fig
    elif isinstance(fig, matplotlib.pyplot.Figure):
        # https://stackoverflow.com/questions/48717794/matplotlib-embed-figures-in-auto-generated-html
        import base64
        from io import BytesIO

        temp_file = BytesIO()
        fig.savefig(temp_file, format="png")
        encoded = base64.b64encode(temp_file.getvalue()).decode("utf-8")
        return "<img src='data:image/png;base64,{}'>".format(encoded)
    else:
        try:
            import plotly

            if isinstance(fig, plotly.graph_objs.Figure):
                # https://stackoverflow.com/questions/59868987/plotly-saving-multiple-plots-into-a-single-html
                return fig.to_html().split("<body>")[1].split("</body>")[0]
            else:
                return ""
        except ImportError:
            return ""


def html_save(contents=[], file="myreport.html"):
    """Combine figures and text in a single HTML document."""
    # https://stackoverflow.com/questions/59868987/plotly-saving-multiple-plots-into-a-single-html
    with open(file, "w", encoding="utf-8") as page:
        page.write(
            r"""<html>
                           <head>
                           <style>
                           h1 {
                               text-align: center;
                               font-family: Arial, Helvetica, sans-serif;
                               }
                           h2 {
                               text-align: center;
                               font-family: Arial, Helvetica, sans-serif;
                               }
                           p {
                               text-align: left;
                               font-family: Arial, Helvetica, sans-serif;
                               }
                           div {
                               text-align: center;
                               font-family: Arial, Helvetica, sans-serif;
                               }
                           ul {
                               text-align: left;
                               list-style-position: inside;
                               font-family: Arial, Helvetica, sans-serif;
                               }
                           </style>
                           </head>
                           <body>
                           <h1>NeuroKit Processing Report</h1>"""
        )
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
    """Get the default values of a function's arguments."""
    # https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_kwargs(report_info, func):
    """Get keyword arguments from report_info and update report_info if defaults."""
    defaults = get_default_args(func)
    kwargs = {}
    for key in defaults:
        if key not in ["sampling_rate", "method"]:
            # if arguments have not been specified by user,
            # set them to the defaults
            if key not in report_info:
                report_info[key] = defaults[key]
            elif report_info[key] != defaults[key]:
                kwargs[key] = report_info[key]
    return kwargs, report_info
