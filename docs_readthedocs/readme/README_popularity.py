import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import popularipy  # https://github.com/DominiqueMakowski/popularipy

import neurokit2 as nk

# accesstoken = "b547333010d0b1253ab44569df3efd94c8a93a63" # old one
accesstoken = "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"
# =============================================================================
# Data
# =============================================================================
downloads = popularipy.pypi_downloads("neurokit2")


neurokit = (
    popularipy.github_stars("neuropsychology/neurokit", accesstoken)
    .rename(columns={"Stars": "NeuroKit2"})
    .set_index("Date")
)
neurokit1 = (
    popularipy.github_stars("neuropsychology/neurokit.py", accesstoken)
    .rename(columns={"Stars": "NeuroKit1"})
    .set_index("Date")
)
biosppy = (
    popularipy.github_stars("PIA-Group/BioSPPy", accesstoken)
    .rename(columns={"Stars": "BioSPPy"})
    .set_index("Date")
)
pysiology = (
    popularipy.github_stars("Gabrock94/Pysiology", accesstoken)
    .rename(columns={"Stars": "Pysiology"})
    .set_index("Date")
)
hrvanalysis = (
    popularipy.github_stars("Aura-healthcare/hrvanalysis", accesstoken)
    .rename(columns={"Stars": "hrvanalysis"})
    .set_index("Date")
)
heartpy = (
    popularipy.github_stars("paulvangentcom/heartrate_analysis_python", accesstoken)
    .rename(columns={"Stars": "HeartPy"})
    .set_index("Date")
)
systole = (
    popularipy.github_stars("embodied-computation-group/systole", accesstoken)
    .rename(columns={"Stars": "systole"})
    .set_index("Date")
)
pyhrv = (
    popularipy.github_stars("PGomes92/pyhrv", accesstoken)
    .rename(columns={"Stars": "pyhrv"})
    .set_index("Date")
)
hrv = (
    popularipy.github_stars("rhenanbartels/hrv", accesstoken)
    .rename(columns={"Stars": "hrv"})
    .set_index("Date")
)
phys2bids = (
    popularipy.github_stars("physiopy/phys2bids", accesstoken)
    .rename(columns={"Stars": "phys2bids"})
    .set_index("Date")
)
biopeaks = (
    popularipy.github_stars("JanCBrammer/biopeaks", accesstoken)
    .rename(columns={"Stars": "biopeaks"})
    .set_index("Date")
)
eegsynth = (
    popularipy.github_stars("eegsynth/eegsynth", accesstoken)
    .rename(columns={"Stars": "eegsynth"})
    .set_index("Date")
)
wfdb = (
    popularipy.github_stars("MIT-LCP/wfdb-python", accesstoken)
    .rename(columns={"Stars": "wfdb-python"})
    .set_index("Date")
)
BioPsyKit = (
    popularipy.github_stars("mad-lab-fau/BioPsyKit", accesstoken)
    .rename(columns={"Stars": "BioPsyKit"})
    .set_index("Date")
)


stars = (
    pd.concat(
        [
            neurokit,
            neurokit1,
            biosppy,
            pysiology,
            hrvanalysis,
            heartpy,
            systole,
            pyhrv,
            hrv,
            phys2bids,
            biopeaks,
            eegsynth,
            wfdb,
            BioPsyKit
        ],
        axis=1,
    )
    .reset_index()
    .rename(columns={"index": "Date"})
)

stars["Date"] = pd.to_datetime(stars["Date"], infer_datetime_format=True)
stars = stars.sort_values("Date").reset_index(drop=True)


contributors = pd.DataFrame(
    {
        "Package": [
            "NeuroKit2",
            "NeuroKit1",
            "BioSPPy",
            "Pysiology",
            "HeartPy",
            "hrvanalysis",
            "systole",
            "pyhrv",
            "hrv",
            "phys2bids",
            "biopeaks",
            "eegsynth",
            "wfdb-python",
            "BioPsyKit"
        ],
        "Contributors": [
            len(
                popularipy.github_contributors("neuropsychology/neurokit", accesstoken)
            ),
            len(
                popularipy.github_contributors(
                    "neuropsychology/neurokit.py", accesstoken
                )
            ),
            len(popularipy.github_contributors("PIA-Group/BioSPPy", accesstoken)),
            len(popularipy.github_contributors("Gabrock94/Pysiology", accesstoken)),
            len(
                popularipy.github_contributors(
                    "Aura-healthcare/hrvanalysis", accesstoken
                )
            ),
            len(
                popularipy.github_contributors(
                    "paulvangentcom/heartrate_analysis_python", accesstoken
                )
            ),
            len(
                popularipy.github_contributors(
                    "embodied-computation-group/systole", accesstoken
                )
            ),
            len(popularipy.github_contributors("PGomes92/pyhrv", accesstoken)),
            len(popularipy.github_contributors("rhenanbartels/hrv", accesstoken)),
            len(popularipy.github_contributors("physiopy/phys2bids", accesstoken)),
            len(popularipy.github_contributors("JanCBrammer/biopeaks", accesstoken)),
            len(popularipy.github_contributors("eegsynth/eegsynth", accesstoken)),
            len(popularipy.github_contributors("MIT-LCP/wfdb-python", accesstoken)),
            len(popularipy.github_contributors("mad-lab-fau/BioPsyKit", accesstoken)),
        ],
    }
).sort_values("Contributors", ascending=False)


colors = {
    "NeuroKit2": "#E91E63",
    "NeuroKit1": "#F8BBD0",
    "BioSPPy": "#2196F3",
    "Pysiology": "#03A9F4",
    "HeartPy": "#f44336",
    "hrvanalysis": "#FF5722",
    "systole": "#FF9800",
    "pyhrv": "#FFC107",
    "hrv": "#FF4081",
    "phys2bids": "#8BC34A",
    "biopeaks": "#4A148C",
    "eegsynth": "#3F51B5",
    "wfdb-python": "#3F51B5",
    "BioPsyKit": "#d32f2f",
}
# =============================================================================
# Downloads
# ===========================================================================

# Plot
fig, axes = plt.subplots(3, 1, figsize=(7, 3))

# Downloads plot
downloads.plot.area(x="Date", y="Downloads", ax=axes[0], legend=False, color="#2196F3")
downloads.plot(x="Date", y="Trend", ax=axes[0], legend=False, color="#E91E63")

axes[0].xaxis.label.set_visible(False)
axes[0].xaxis.set_ticks_position("none")
# axes[0].set_xticklabels("Date")
axes[0].text(
    0.5,
    0.9,
    "Downloads / Day",
    horizontalalignment="center",
    transform=axes[0].transAxes,
)


# =============================================================================
# Stars
# =============================================================================
stars.plot(x="Date", y="NeuroKit2", ax=axes[1], color=colors["NeuroKit2"], linewidth=3)
stars.plot(
    x="Date",
    y="NeuroKit1",
    ax=axes[1],
    color=colors["NeuroKit1"],
    linewidth=1.75,
    linestyle="dotted",
)
stars.plot(x="Date", y="BioSPPy", ax=axes[1], color=colors["BioSPPy"], linewidth=1.5)
stars.plot(
    x="Date", y="Pysiology", ax=axes[1], color=colors["Pysiology"], linewidth=1.5
)
stars.plot(x="Date", y="HeartPy", ax=axes[1], color=colors["HeartPy"], linewidth=1.5)
stars.plot(
    x="Date", y="hrvanalysis", ax=axes[1], color=colors["hrvanalysis"], linewidth=1.5
)
stars.plot(x="Date", y="systole", ax=axes[1], color=colors["systole"], linewidth=1.5)
stars.plot(x="Date", y="pyhrv", ax=axes[1], color=colors["pyhrv"], linewidth=1.5)
stars.plot(x="Date", y="hrv", ax=axes[1], color=colors["hrv"], linewidth=1.5)
stars.plot(
    x="Date", y="phys2bids", ax=axes[1], color=colors["phys2bids"], linewidth=1.5
)
stars.plot(x="Date", y="biopeaks", ax=axes[1], color=colors["biopeaks"], linewidth=1.5)
stars.plot(x="Date", y="eegsynth", ax=axes[1], color=colors["eegsynth"], linewidth=1.5)
stars.plot(
    x="Date", y="wfdb-python", ax=axes[1], color=colors["wfdb-python"], linewidth=1.5
)

axes[1].text(
    0.5, 0.9, "GitHub Stars", horizontalalignment="center", transform=axes[1].transAxes
)
axes[1].set_ylim(ymin=0)


# =============================================================================
# Contributors
# =============================================================================

axes[2].plot(
    contributors["Package"], contributors["Contributors"], color="black", zorder=1
)
for i, pkg in enumerate(contributors["Package"]):
    axes[2].plot(
        pkg,
        contributors.iloc[i]["Contributors"],
        color=colors[pkg],
        zorder=2,
        marker="o",
        markersize=12,
    )
axes[2].tick_params(axis="x", rotation=45)
axes[2].text(
    0.5,
    0.9,
    "Number of Contributors",
    horizontalalignment="center",
    transform=axes[2].transAxes,
)


# =============================================================================
# Save
# =============================================================================
[ax.legend(loc=0) for ax in plt.gcf().axes]
fig = plt.gcf()
fig.set_size_inches(8 * 2, 8 * 2, forward=True)
fig.savefig("README_popularity.png", dpi=450)
# fig.savefig("D:/Dropbox/RECHERCHE/N/NeuroKit/docs/readme/README_popularity.png", dpi=450)

# =============================================================================
# Activity
# =============================================================================
# import pandas as pd
# import urllib.request
# import json
#
# user = "DominiqueMakowski"
#
# page_number = 0
# activity_remaining = True
#
# dates = []
# types = []
# repos = []
#
# while activity_remaining:
#    query_url = "https://api.github.com/users/%s/events?page=%s&access_token=%s"  % (user, page_number, accesstoken)
#
#    req = urllib.request.Request(query_url)
#    req.add_header('Accept', 'application/vnd.github.v3.star+json')
#    response = urllib.request.urlopen(req)
#    data = json.loads(response.read())
#
#    for commit in data:
#        dates.append(commit["created_at"])
#        types.append(commit["type"])
#        repos.append(commit["repo"]["name"])
#
#    if page_number > 100:
#        activity_remaining = False
#    print(commit["created_at"])
#
#    page_number += 1
#
#
# data = pd.DataFrame({"Date": dates,
#                     "Type": types,
#                     "Repo": repos})
