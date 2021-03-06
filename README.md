# IntProbDS
Julia code to accompany _Introduction to Probability for Data Science_ by Stanley H. Chan https://probability4datascience.com/index.html

This holds Julia translations of the Matlab and Python code that accompanies the book. This effort has been authorized by Stanley H. Chan.

To use this code:
* download or clone the archive to some directory.
* Go to that directory, and start Julia with ```julia --proj``` to let Julia know which packages are needed. 
* The first time you do this, enter ```] instantiate``` to install all of the needed packages, followed by CTRL-C to go back to the Julia prompt, once the packages have been installed (you may be told to exit and re-start Julia)
* once that's done, enter Julia again with ```julia --proj```, and you are ready to run the code.
* from the files for each chapter, copy the code block you desire to run, and paste it into the Julia REPL. In a few cases, you will need data files that are available from the book's web site, linked above. Make sure you download the needed files into the working directory before running those blocks of code.
