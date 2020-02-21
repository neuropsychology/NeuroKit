# -*- coding: utf-8 -*-
def _eventrelated_addinfo(epoch, output={}):

    # Add label
    if "Label" in epoch.columns:
        if len(set(epoch["Label"])) == 1:
            output["Label"] = epoch["Label"].values[0]

    # Add condition
    if "Condition" in epoch.columns:
        if len(set(epoch["Condition"])) == 1:
            output["Condition"] = epoch["Condition"].values[0]
    return output
