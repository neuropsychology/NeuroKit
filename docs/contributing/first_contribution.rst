Ideas for first contributions
================================

.. hint::
   Spotted a typo? Would like to add something or make a correction? Join us by contributing (`see our guides <https://neurokit2.readthedocs.io/en/latest/contributing/index.html>`_).
   
   
Now that you're familiar with `how to use GitHub <https://neurokit2.readthedocs.io/en/latest/contributing/contributing.html#how-to-use-github-to-contribute>`_, you're ready to get your hands dirty and contribute to open-science? But you're not sure **where to start or what to do**? We got you covered!

In this guide, we will discuss the two best types of contributions for beginners, as they are easy to make, super useful and safe (you cannot break the package 😏).

Talk about it
------------------

Contributing to the development of a package also means helping to popularize it, so that more people hear about it and use it. So do not hesitate to **talk about it on social media** (twitter, reddit, research gate, ...) and present it to your students or colleagues. Also, do not hesitate to write blogposts about it (or even make some videos if you're a YouTube influencer 😎). And let us know if you do that, we'll try to boost your outreach by retweeting, sharing and spreading it.


Look for *"good first contribution"* issues
-------------------------------------------

If you know how to code a bit, you can check out the issues that have been flagged as `good for first contribution <https://github.com/neuropsychology/NeuroKit/labels/good%20first%20contribution%20%3Asun_with_face%3A>`_. This means that they are issue or features ideas that we believe are accessible to beginners. If you're interested, do not hesitate to comment on these issues to know more, have more info or ask for guidance! We'll be really happy to help in any way we can ☺️.



Improving documentation
-------------------------

One of the easiest thing is to improve, complete or fix the documentation for functions. For instance the `ecg_simulate() <https://neurokit2.readthedocs.io/en/latest/functions.html#neurokit2.ecg_simulate>`_ function has a documentation with a general description, a description of the arguments, some example etc. As you've surely noticed, sometimes more details would be needed, some typos are present, or some references could be added.

The documentation for functions is located alongside the function *definition* (the code of the function). If you've read `understanding NeuroKit <https://neurokit2.readthedocs.io/en/latest/contributing/understanding.html>`_, you know that the code of the `ecg_simulate()` function is `here <https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/ecg/ecg_simulate.py>`_. And as you can see, just below the function name, there is a big *string* (starting and ending with `"""`) containing the documentation. 

This thing is called the *docstring*. 

If you modify it here, then it will be updated automatically on the website!




Adding tests
----------------

Tests are super important for programmers to make sure that the changes that we make at one location don't create unexpected changes at another place.

Adding them is a good first issue for new contributors, as it takes little time, doesn't require advanced programming skills and is a good occasion to discover functions and how they work.

By clicking on the `"coverage" badge <https://codecov.io/gh/neuropsychology/NeuroKit>`_ under the logo on the README page, then on the "neurokit2" folder button at the bottom, you can see the `breakdown of testing coverage <https://codecov.io/gh/neuropsychology/NeuroKit/tree/master/neurokit2>`_ for each submodules (folders), and if you click on one of them, the coverage for each individual file/function (`example here <https://codecov.io/gh/neuropsychology/NeuroKit/tree/master/neurokit2/stats>`_).

This percentage of coverage needs be improved ☺️

The common approach is to identify functions, methods or arguments that are not tested, and then try to write a small test to cover them (i.e., a small self-contained piece of code that will run through a given portion of code and which output is tested (e.g., `assert x == 3`) and depends on the correct functioning of that code), and then add this test to the appropriate `testing file <https://github.com/neuropsychology/NeuroKit/tree/master/tests>`_.

For instance, let's imagine the following function:

.. code-block:: python

    def domsfunction(x, method="great"):
        if method == "great": 
             z = x + 3
        else:
             z = x + 4
        return z


In order to test that function, I have to write some code that "runs through" it and put in a function which name starts with `test_*`, for instance:

.. code-block:: python

    def test_domsfunction():
        # Test default parameters
        output = domsfunction(1)
        assert output == 4


This will go through the function, which default method is `"great"`, therefore adds `3` to the input (here 1), and so the result *should* be 4. And the test makes sure that it is 4. However, we also need to add a second test  to cover the other method of the function (when `method != "great"`), for instance:

.. code-block:: python

    def test_domsfunction():
        # Test default parameters
        output = domsfunction(1)
        assert output == 4

        # Test other method
        output = domsfunction(1, method="whatever")
        assert isinstance(output, int)


I could have written `assert output == 5`, however, I decided instead to check the type of the output (whether it is an integer). That's the thing with testing, it requires to be creative, but also in more complex cases, to be clever about what and how to test. But it's an interesting challenge 😏 

You can see examples of tests in the existing `test files <https://github.com/neuropsychology/NeuroKit/tree/master/tests>`_.

And if you want to deepen your understanding of the topic, check-out this very accessible `pytest tutorial for data science <https://github.com/poldrack/pytest_tutorial>`_.




Adding examples and tutorials
----------------------------------

How to write
^^^^^^^^^^^^^^

The documentation that is on the `website <https://neurokit2.readthedocs.io/en/latest/>`_ is automatically built by the hosting website, readthedocs, from `reStructured Text (RST) files <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html>`_ (a syntax similar to markdown) or from `jupyter notebooks (.ipynb) <https://jupyter.org/>`_ Notebooks are preferred if your example contains code and images.


Where to add the files
^^^^^^^^^^^^^^^^^^^^^^^^

These documentation files that we need to write are located in the `/docs/ <https://github.com/neuropsychology/NeuroKit/tree/master/docs>`_ folder. For instance, if you want to add an example, you need to create a new file, for instance `myexample.rst`, in the `docs/examples/` folder.

If you want to add images to an `.rst` file, best is to put them in the `/docs/img/ <https://github.com/neuropsychology/NeuroKit/tree/master/docs/img>`_ folder and to reference their link.

However, in order for this file to be easily **accessible from the website**, you also need to add it to the **table of content** located in the `index <https://github.com/neuropsychology/NeuroKit/blob/master/docs/examples/index.rst>`_ file (just add the name of the file without the extension).

Do not hesitate to ask for more info by creating an `issue <https://github.com/neuropsychology/NeuroKit/issues>`_!

