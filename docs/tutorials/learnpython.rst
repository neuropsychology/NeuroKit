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

The next important thing to have in mind is that variables have **types**. Basic types include **integers** (numbers without decimals), **floats** (numbers with decimals) or **string** (character text). Depending on the type, the variables will not behave the same. For example, try:

.. code-block:: python

    print(1 + 2.0)
    print("1" + "2.0")
    
What happened here? Well, quotations are used to represent **strings** (text). So in the second line, the numbers that we added were not numbers, but text. And when you add strings together in Python, it *concatenates* them.

One can change the type of a variable with the following:

.. code-block:: python

    int(1.0)  # transform the input to an integer
    float(1)  # transform the input to a float
    str(1)  # transform the input into text
    
Also, here I used the hashtag symbol to **make comments**, i.e., writing stuff that won't be executed by Python. This is super useful to annotate each line of your code to remember what you do.


Lists and dictionnaries
------------------------

Two other important types are **lists** and **dictionnaries**. You can think of them as **containers**, as they contain multiple variables. The main difference between them is that in a **list**, you access the individual elements that it contains **by its order** (for instance, the third one), whereas in a **dictionnary**, you access an element by its name (also known as **key**), for example "the element named 'A'".

A list is created using square brackets, and a dictionnary using curly brackets. Importantly, in a dictionnary, you must specify a name to each element. Here's what it looks like:


.. code-block:: python

    mylist = [1, 2, 3]
    mydict = {"A": 1, "B": 2, "C": 3}


Keep in mind that there are more types of containers, such as *arrays* and *dataframes*, that we will talk about later.

Basic indexing
--------------------

There's no point in storing elements in containers if we cannot access them later on. As mentioned earlier, we can access elements from a **dictionnary** by its key within square brackets (note that here the square brackets don't mean *list*, just mean *within the previous container*).

.. code-block:: python

    mydict = {"A": 1, "B": 2, "C": 3}
    x = mydict["B"]
    print(x)

**Exercice time!** If you have followed this tutorial so far, you can guess what the following code will output:

.. code-block:: python

    mydict = {"1": 0, "2": 42, "x": 7}
    x = str(1 + 1)
    y = mydict[x]
    print(y)

If you guessed **42**, you're right, congrats! If you guessed **7**, you have likely confused the **variable** named `x` (which represents 1+1 converted to a character), with the character `"x"`. 



Indexing starts from 0
------------------------

As mentioned earliers, one can access elements from a list by its **order**. However, **and there is very important to remember** (the source of many beginner errors), in Python, **the order starts from 0**. That means that the **first element is the 0th**.

So if we want the 2nd element of the list, we have to ask for the 1th:

.. code-block:: python

    mylist = [1, 2, 3]
    x = mylist[1]
    print(x)
    


Control flow (if and else)
----------------------------

One important notion in programming is control flow. You want the code to do something different depending on a condition. For instance, if `x` is lower than 3, print "lower than 3". In Python, this is done as follows:



.. code-block:: python

    x = 2
    if x < 3:
        print("lower than 3")

One very important thing to notice is that the **if statement** corresponds to a "chunk" of code, as signified by the colon `:`. The chunk has to be written below, and has to be **indented** (you can ident a line or a chunk of code by pressing the `TAB` key). 

**What is identation?**


.. code-block:: console

    this
        is
            indentation
            

And this is very important in Python, if try runnning the following, it will **error**:

.. code-block:: python

    if 2 < 3:
    print("lower than 3")


Finally, **if** statements can be followed by **else** statements, which takes care of what happens if the condition is not fullfilled:


    x = 5
    if x < 3:
        print("lower")
    else:
        print("higher")

Again, note the **identation** and how the **else** statement creates a new idented chunk. 


For loops
----------


Functions
------------




Packages
-------------



Lists and vectors (arrays)
--------------------------

.. code-block:: console

    mylist = [1, 2, 3]
    for i in range(0, 2):
        mylist[i] = mylist[i] + 1
    print(mylist)

Conditional indexing
---------------------


DataFrames
------------



Reading data
-------------


