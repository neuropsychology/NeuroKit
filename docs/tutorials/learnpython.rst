Get familiar with Python in 10 minutes
=========================================

.. hint::
   Spotted a typo? Would like to add something or make a correction? Join us by contributing (`see this tutorial <https://neurokit2.readthedocs.io/en/latest/contributing.html>`_).


You have no experience in programming? You are afraid of code? You feel betrayed because you didn't expect to code in psychology studies? **Relax!**

This tutorial will provide you with all you need to know to dive into the wonderful world of scientific programming. The goal here is not become a programmer, or a software designer, but rather to be able to use the power of programming to get some **scientific results** out.



Setup
---------------

The first thing you will need is to **install Python** on your computer (we have `tutorial for that <https://neurokit2.readthedocs.io/en/latest/installation.html>`_). In fact, this will do **two things**, installing Python (the *language*), and an *environment* to be able to use it. For this tutorial, we will assume you have something that looks like `Spyder <https://www.spyder-ide.org/>`_.

There is one important concept here to grasp: the difference between the **CONSOLE** and the **EDITOR**. The editor is where you write the code. It's basically a text editor (such as notepad), except that it automatically highlights the code. Importantly, you can directly *execute* a line of code (which is equivalent to copy it and paste it the *console*).

For instance, you can write `1+1` somewhere in the file in the editor pane. Now if select the piece of code you just wrote, and press `F9` (or `CTRL + ENTER`), it will **execute it**.


.. image:: https://raw.github.com/neuropsychology/Neurokit/master/docs/img/learnpython/learnpython_1.jpg


As a result, you should see in the console the order that you gave and below, the **output** (which is `2`). Now, take some time to explore the settings and turn the editor background to **BLACK**. Why? Because it's more comfortable for the eyes, but most importantly, because it's cool 😎.


.. image:: https://raw.github.com/neuropsychology/Neurokit/master/docs/img/learnpython/learnpython_2.jpg

**Congrats, you've become a programmer**, a wizard of the modern times.


You can now save the file (`CTRL + S`), which will be saved with a `.py` extension (i.e., a Python file). Try closing everything and reopening this file with the editor.


Variables
---------------

The second important concept is **variables**, which is a fancy name for something that you already know. Do you remember, from your mathematics classes, the famous *X*? This placeholder for any value? Well, *X* was a variable, i.e., the name refering to some other thing.

So we can *assign* a value to a *variable* using the `=` sign, for instance:

.. code-block:: python

    x = 2
    y = 3
    
Once we execute these two lines, Python will know that `x` refers to 2 and `y` to 3. We can now write:

.. code-block:: python

    print(x + y)

Which will print in the console the correct result.

.. image:: https://raw.github.com/neuropsychology/Neurokit/master/docs/img/learnpython/learnpython_3.jpg

We can also store the output in a third variable:

.. code-block:: python

    x = 2
    y = 3
   
    anothervariable = x * y
    print(anothervariable)


Variables and data types (classes)
----------------------------------


Keep in mind that there are more types, such as arrays and dataframes, that we will talk about later.



Functions
------------




Packages
-------------



Lists and vectors (arrays)
--------------------------


Indexing
------------


Control flow
----------------

.. code-block:: console

    this
        is
            indentation
            
            
Dataframes
-------------


- reading data

