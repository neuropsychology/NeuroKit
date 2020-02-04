Understanding NeuroKit
======================
**Here, we review basic coding practices in order for beginners to get familiar with NeuroKit**

If you are reading this, it's because you don't feel confortable with Python and you impatiently want to get to know it and Neurokit to start looking at your data

"*Tous les chemins mènent à Rome*"

	Let me start by saying that there are multiple ways you'll be able to access the documentation in order to get to know different functions, follow examples and other tutorials. So keep in mind that you will find your own workflow, and that these tricks are shared to help you get to know your options.

First : readthedocs
-------------------
You probably already saw the beautiful README.rst file that shows up on NeuroKit's Github home page (right after the list of directories). It contains a brief overview of the project, some examples and figures. *But, most importantly, there's a section dedicated to the Documentation*. 

    **Documentation** basically means code explanations, references and examples. 

In the Documentation section of the README, you'll find links to the `readthedocs website <https://neurokit2.readthedocs.io/en/latest/?badge=latest>`_ like this one : 

.. image:: https://readthedocs.org/projects/neurokit2/badge/?version=latest
        :target: https://neurokit2.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

And a link to the `API (or Application Program Interface <https://neurokit2.readthedocs.io/en/latest/functions.html>`_) like this one:

.. image:: https://img.shields.io/badge/functions-API-orange.svg?colorB=2196F3
        :target: https://neurokit2.readthedocs.io/en/latest/functions.html
        :alt: API

All the info you will see on that webpage is rendered directly from the code. **That's why it's important to structure your code in a standard manner** (You can learn how to contribute `here <https://neurokit2.readthedocs.io/en/latest/contributing.html>`_). 

The API is organized by types of signals. You'll find that each function has a **description**, and that most of them refer to peer-reviewed papers or other Github repository. Also, for each function, **parameters** are described in order. Some of them will take many different **options** and all of them should be described as well. 

    **If the options are not explained, they should be**. 
    
    *It's not your fault you don't understand. That's why we need you to contribute.*

Example
"""""""
    In the **ECG section**, the `find_peaks function <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_findpeaks>`_ takes **4 parameters**. 
    
    One of them is **method**: each method refers to a peer-reviewed paper that published a peak detection algorithm. 

    You can also see what the function **returns** and what **type of data** has been returned (integers and floating point numbers, strings, etc). 

    Additionnaly, you can find **related functions** in the **See also** part. 

    An **example** of the function should also be found. You can copy paste it in your Python kernel, or in a Jupyter Notebook.

Second : the code on Github 
---------------------------
Now that you're familiar with readthedocs' website, let's go back to the `repo <https://github.com/neuropsychology/NeuroKit>`_. What you have to keep in mind is that *everything you saw in the previous section is* **in the Github repository**. Everything is cross-referenced, everything relates to the core which can be found in the repo. If you got here, you probably already know that a repository is like a *tree containing different branches* or directories that eventually lead you to a **script**, in which you can find a **function**.

Example
""""""""
Ready for inception ? let's find the location of the file you're currently reading. Go under ``docs`` and find it by yourself... it should be straight-forward.

See! It's super handy because you can visit the scripts without downloading it. Github also renders Jupyter Notebook quite well, so you can not only see the script, but also figures and markdown sections where the coder discusses results.


Third : the code on YOUR machine
--------------------------------
Now, you're probably telling yourself :

    *If I want to use these functions, it should be somewhere on my computer!* 

For that, I encourage you to visit the `installation page <https://neurokit2.readthedocs.io/en/latest/installation.html>`_ if you didn't already. Once Python is installed, its default pathway should be :

Python directory
"""""""""""""""""
Windows 
"""""""
* ``C:\Users\<username>\anaconda3\`` 
    
(if the directory doesn't match, just search for the folder name ``anaconda3`` or ``miniconda3``. 

Mac
""""
* ``/Users/<username>/anaconda3`` .

*Linux users should know that already*

Environment and NeuroKit directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Neurokit should be located under the environment where you installed it (*if you didn't do it already, set a computing environment. Otherwise, you can run into problems when running your code*). The directory should look like this:

Windows
"""""""
* ``C:\Users\<username>\anaconda3\envs\<yourenv>\lib\site-package\neurokit2``

Mac
"""
* Mac users can figure out

Example
""""""""
**Take the ECG again :**

From the specified directory, I can note that the different folders are arranged in the same way as in the readthedocs website. 

Let's say I want to go back to the same function `find_peaks`: I'll click on ``ecg`` folder, and from there I can see the source code for the function under ; `ecg_findpeaks.py`