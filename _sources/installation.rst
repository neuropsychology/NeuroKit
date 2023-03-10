.. highlight:: shell

Installation
============


Open your terminal and run:

.. code-block:: console

    pip install neurokit2

Then, at the top of each of your Python script, you should be able to import the module:

.. code-block:: python

    import neurokit2 as nk


.. Hint::

    Living on the edge? You can get the latest **dev** version from GitHub by running:

    .. code-block:: console

        pip install https://github.com/neuropsychology/neurokit/zipball/dev



You don't have Python
-----------------------

You are new to all this, and you're not even sure if you have Python installed? Don't worry, we'll walk you through all of it.

Python + VS code
^^^^^^^^^^^^^^^^

You will need two things to program in Python, Python itself and an IDE software to edit and work with the Python scripts.

1. You can download Python from https://www.python.org/downloads/
2. For the IDE, we will go with `VS Code <https://code.visualstudio.com/download>`_
3. Once VS Code is launched, the next step is to add functionalities to support your development workflow. In particular, it is critical to get the `Python Interactive extension <https://code.visualstudio.com/docs/python/jupyter-support-py>`_
4. To start running some code, click *New File* and `Ctrl+S` to save the file into whichever directory in your computer you want, naming the file with a `.py` extension. Press `Shift+Enter` to send each line of code to an interactive window

Winpython
^^^^^^^^^

Another, perhaps easier option is to download a full distribution. The advantage of Winpython is its portability (i.e., works out of a folder) and default setup (convenient for science). However, it only exists **for Windows**.

1. Download a non-zero version of `Winpython <http://winpython.github.io/>`_
2. Install it somewhere (the desktop is a good place). It creates a folder called `WPyXX-xxxx`
3. In the `WPyXX-xxxx` folder, open `WinPython Command Prompt.exe`
4. Now you can proceed to running the PIP command mentioned at the top

Miniconda or Anaconda
^^^^^^^^^^^^^^^^^^^^^^

The difference between the two is straightforward, *miniconda* is recommended if you don't have much storage space and you know what you want to install. Similar to Winpython, *Anaconda* comes with a *base* environment, meaning you have basic packages pre-installed.
Here is some `more information <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda>`_ to help you choose between *miniconda* and *Anaconda*.

1. Download and install `Miniconda or Anaconda <https://www.anaconda.com/products/individual>`_ (make sure the ``Anaconda3`` directory is similar to this: ``C:\Users\<username>\anaconda3\``)
2. Open the `Anaconda Prompt` (search for it on your computer; see `here <https://www.youtube.com/watch?time_continue=59&v=gk2CgkURkgY>`_ for Mac users)
3. Run :code:`conda help` to see your options

.. Note:: There should be a name in parentheses before your user's directory, e.g. ``(base) C:\Users\<yourusername>``. That is the name of your computing environment. By default, you have a ``base environment``. We don't want that, so create an environment.

1. Run :code:`conda env create <yourenvname>`; activate it every time you open conda by running :code:`conda activate <yourenvname>`
2. Now you can proceed to the next step.

.. image:: https://raw.github.com/neuropsychology/NeuroKit/master/docs/img/tutorial_installation_conda.jpg



From conda
---------------

If you have Python installed as part of `Miniconda` or `Anaconda`, please follow these steps:

1. As described in above, open the `Anaconda Prompt` and activate your conda environment
2. You can now install NeuroKit2 from `conda-forge <https://anaconda.org/conda-forge/neurokit2>`_ by typing

.. code-block:: console

    conda config --add channels conda-forge
    conda install neurokit2

`conda` or `pip` are the preferred methods to install NeuroKit2, as they will install the most up-to-date stable release.



