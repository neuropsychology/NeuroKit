# -*- coding: utf-8 -*-
import inspect


def text_combine(info):
    """Reformat dictionary describing processing methods as strings to be inserted into HTML file."""
    preprocessing = '<h2 style="background-color: #FB1CF0">Preprocessing</h1>'
    for key in ["text_cleaning", "text_peaks"]:
        if key in info.keys():
            preprocessing += info[key] + "<br>"
    ref = '<h2 style="background-color: #FBB41C">References</h1>'
    if "references" in info.keys():
        ref += "\n <ul> \n"
        for reference in info["references"]:
            ref += "<li>" + reference + "</li>" + "\n"
        ref += "\n </ul> \n"
    return preprocessing, ref


def html_save(contents=[], file="myreport.html"):
    """Combine figures and text in a single HTML document."""
    # https://stackoverflow.com/questions/59868987/plotly-saving-multiple-plots-into-a-single-html
    with open(file, "w") as page:
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
