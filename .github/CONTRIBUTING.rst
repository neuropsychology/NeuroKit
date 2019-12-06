Contribution Guidelines 
========================


**All people are very much welcome to contribute to code, documentation, testing and suggestions.**

This package aims at being beginner-friendly. Even if you're new to this open-source way of life, new to coding and github stuff, we encourage you to try submitting pull requests (PRs). 

- **"I'd like to help, but I'm not good enough with programming yet"**

It's alright, don't worry! You can always dig in the code, in the documentation or tests. There are always some typos to fix, some docs to improve, some details to add, some code lines to document, some tests to add... **Even the smaller PRs are appreciated**.

- **"I'd like to help, but I don't know where to start"**

You can look around the **issue section** to find some features / ideas / bugs to start working on. You can also open a new issue **just to say that you're there, interested in helping out**. We might have some ideas adapted to your skills.

- **"I'm not sure if my suggestion or idea is worthwile"**

Enough with the impostor syndrom! All suggestions and opinions are good, and even if it's just a thought or so, it's always good to receive feedback.

- **"Why should I waste my time with this? Do I get any credit?"**

Software contributions are getting more and more valued in the academic world, so it is a good time to collaborate with us! Authors of substantial contributions will be added within the **authors** list. We're also very keen on including them to eventual academic publications.


**Anyway, starting is the most important! You will then enter a *whole new world, a new fantastic point of view*... So fork this repo, do some changes and submit them. We will then work together to make the best out of it :)**




How to submit a change? 
------------------------

- You don't know how much about code? You can contribute to [documentation](https://github.com/neuropsychology/NeuroKit.py/tree/master/docs) by creating tutorials, help and info!
- The master branch is [protected](https://help.github.com/articles/about-pull-request-reviews/): you should first start to create a new branch in your git clone, called for example "bugfix-functionX" or "feature-readEEG". Then, you add your commits to this branch, push it and create a pull request to merge it to master. In other words, avoid modifications from master to master.



Structure
---------------

- NeuroKit is currently organized into 5 major sections: Biosignals, M/EEG, statistics, miscellaneous and implementation of several statistical procedures used in psychology/neuroscience. However, this structure might be changed in order to be clarified or expanded (MRI processing, eye tracking, ...).
- API coherence has to be maintained/increased, with functions starting with the same prefix (`find_`, `read_`, `compute_`, `plot_`, `ecg_`, etc.) so that the user can easily find them by typing the "intuitive" prefix.




Code
---------------
- Authors of code contribution are invited to follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style sheet to write some nice (and readable) python.
- Contrary to Python recommandations, I prefer some nicely nested loops, rather than complex one-liners ["that" for s if h in i for t in range("don't") if "understand" is False].
- Please document and comment your code, so that the purpose of each step (or code line) is stated in a clear and understandable way.
- Avoid unnecessary use of "magic" functions (preceded by underscores, `_foo()`). I don't know I don't find them elegant. But it's personal. I might be wrong.



Ressources
---------------

- [Understanding the GitHub flow](https://guides.github.com/introduction/flow/)


