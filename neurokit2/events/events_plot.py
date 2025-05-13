# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def events_plot(events, signal=None, color="red", linestyle="--"):
    """**Visualize Events**

    Plot events in signal.

    Parameters
    ----------
    events : list or ndarray or dict
        Events onset location. Can also be a list of lists, in which case it will mark them with
        different colors. If a dict is passed (e.g., from :func:`events_find`), it will only plot
        the onsets.
    signal : array or DataFrame
        Signal array (can be a dataframe with many signals).
    color : str
        Argument passed to matplotlib plotting.
    linestyle : str
        Argument passed to matplotlib plotting.

    Returns
    -------
    fig
        Figure representing a plot of the signal and the event markers.

    See Also
    --------
    events_find

    Examples
    ----------
    .. ipython:: python

      import neurokit2 as nk

      @savefig p_events_plot1.png scale=100%
      nk.events_plot([1, 3, 5])
      @suppress
      plt.close()

    * **Example 1**: With signal

    .. ipython:: python

      signal = nk.signal_simulate(duration=4)
      events = nk.events_find(signal)

      @savefig p_events_plot2.png scale=100%
      nk.events_plot(events, signal)
      @suppress
      plt.close()

    * **Example 2**: Different events

    .. ipython:: python

      events1 = events["onset"]
      events2 = np.linspace(0, len(signal), 8)

      @savefig p_events_plot3.png scale=100%
      nk.events_plot([events1, events2], signal)
      @suppress
      plt.close()

    * **Example 3**: Conditions

    .. ipython:: python

      events = nk.events_find(signal, event_conditions=["A", "B", "A", "B"])

      @savefig p_events_plot4.png scale=100%
      nk.events_plot(events, signal)
      @suppress
      plt.close()

    * **Example 4**: Different colors for all events

    .. ipython:: python

      signal = nk.signal_simulate(duration=10)
      events = nk.events_find(signal)
      events = [[i] for i in events['onset']]

      @savefig p_events_plot5.png scale=100%
      nk.events_plot(events, signal)
      @suppress
      plt.close()

    """

    if isinstance(events, dict):
        if "condition" in events.keys():
            events_list = []
            for condition in set(events["condition"]):
                events_list.append(
                    [x for x, y in zip(events["onset"], events["condition"]) if y == condition]
                )
            events = events_list
        else:
            events = events["onset"]

    if signal is None:
        signal = np.full(events[-1] + 1, 0)
    if isinstance(signal, pd.DataFrame) is False:
        signal = pd.DataFrame({"Signal": signal})

    # Plot signal(s)
    axs = signal.plot()

    # Check if events is list of lists
    try:
        len(events[0])
        is_listoflists = True
    except TypeError:
        is_listoflists = False

    if is_listoflists is False:
        # Loop through sublists
        for event in events:
            plt.axvline(event, color=color, linestyle=linestyle)

    else:
        # Convert color and style to list
        if isinstance(color, str):
            color_map = plt.get_cmap("rainbow")
            color = color_map(np.linspace(0, 1, num=len(events)))
        if isinstance(linestyle, str):
            linestyle = np.full(len(events), linestyle)

        # Loop through sublists
        for i, event in enumerate(events):
            for j in events[i]:
                plt.axvline(j, color=color[i], linestyle=linestyle[i], label=str(i))

        # Display only one legend per event type
        handles, labels = plt.gca().get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)
        plt.legend(newHandles, newLabels)

    return axs.get_figure()
