How to contribute
==================
**Neurokit2 welcomes everyone to contribute to code, documentation, testing and suggestions.**

This package aims at being beginner-friendly. And if you have yet been familiar with how contribution can be done to open-source packages, this guide is for you!

**Let's dive right into it!**


Step 1: Fork it
----------------
A *fork* is a copy of a repository. Working with the fork allows you to freely experiment with changes without affecting the original project.

Hit the **Fork** button in the top right corner of the page and in a few seconds, you will have a copy of the repository in your own GitHub account.

.. image:: https://raw.github.com/neuropsychology/NeuroKit/dev/docs/img/fork.png

Now, that is the *remote* copy of the project. The next step is to make a *local* copy in your computer. 

While you can explore Git to manage your Github developments, we recommend downloading `Github Desktop <https://desktop.github.com/>`_ instead. It makes the process way easier and more straightforward.
 
Step 2: Clone it
-----------------
Clonning allows you to make a *local* copy of any repositories on Github. 

Go to **File** menu, click **Clone Repository** and since you have forked Neurokit2, you should be able to find it easily under **Your repositories**. 

.. image:: https://raw.github.com/neuropsychology/NeuroKit/dev/docs/img/clone_nk.png

Choose the local path of where you want to save your *local* copy and as simple as that, you have a working repository in your computer.

Step 3: Find it and fix it
---------------------------
And here is where the fun begins. You can start contributing by fixing a bug (or even a typo in the code) that has been annoying you. Or you can go to the `issue section <https://github.com/neuropsychology/NeuroKit/issues/>`_ to hunt for issues that you can address. 

For example, here, as I tried to run the example in `ecg_fixpeaks()` file, I ran into a bug! A typo error!

.. image:: https://raw.github.com/neuropsychology/NeuroKit/dev/docs/img/fix_typo.gif

Fix it and hit the save button! That's one contribution I made to the package!

To save the changes you made (e.g. the typo that was just fixed) to your *local* copy of the repository, the next step is to *commit* it.

Step 4: Commit it and push it
------------------------------

In your Github Desktop, you will now find the changes that you made highlighted in **red** (removed) or **green** (added). 

The first thing that you have to do is to switch from the default - *Commit to Master* to *Commit to dev**. Always commit to your dev branch as it is the branch with the latest changes. Then give the changes you made a good and succinct title and hit the *Commit* button.

.. image:: https://raw.github.com/neuropsychology/NeuroKit/dev/docs/img/commit.png

**Committing** allows your changes to be saved in your *local* copy of the repository and in order to have the changes saved in your **remote** copy, you have to **push* the commit that you just made.

Step 4: Create pull request
----------------------------

The last step to make your contribution official is to create a pull request. 

.. image:: https://raw.github.com/neuropsychology/NeuroKit/dev/docs/img/pr.png

Go to your *remote* repository on Github page, the *New Pull Request* button is located right on top of the folders. Do remember to change your branch to *dev* since your commits were pushed to the dev branch previously. 

And now, all that is left is for the maintainers of the package to review your work and they can either request additional changes or merge it to the original repository. 

Useful links
=============
- `Understand Neurokit <https://neurokit2.readthedocs.io/en/latest/tutorials/understanding.html/>`_
- `Contributing guide <https://neurokit2.readthedocs.io/en/latest/contributing.html/>`_
