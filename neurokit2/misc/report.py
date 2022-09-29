# -*- coding: utf-8 -*-
import inspect


def text_combine(info):
    """Reformat dictionary describing processing methods as strings to be inserted into HTML file."""
    preprocessing = "<br><b>Preprocessing</b><br>"
    for key in ["text_cleaning", "text_peaks"]:
        if key in info.keys():
            preprocessing += info[key] + "<br>"

    ref = "<br><b>References</b><br>"
    if "references" in info.keys():
        ref += "<ul>"
        for reference in info["references"]:
            ref += "<li>" + reference + "</li>"
        ref += "</ul>"
    return preprocessing, ref


def html_combine(contents=[], file="myreport.html"):
    """Combine figures and text in a single HTML document."""
    # https://stackoverflow.com/questions/59868987/plotly-saving-multiple-plots-into-a-single-html
    with open(file, "w") as page:
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
    """Get the default values of a function's arguments."""
    # https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
