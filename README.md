### Bayesian Global Optimization Using Gaussian Processes
**Bachelor Thesis, 2014, ETHZ**

This repository contains implementations of the **Expected Improvement** and the **Gaussian Process Upper Confidence Bound algorithm** in *MATLAB*, which are part of my Bachelor thesis.

For a documentation of the code, please read Chapter 5 of my thesis [here](https://github.com/cglanzer/bayesian-global-optimization/blob/master/BT_Chapter_5.pdf). A complete version of my thesis can be found on my [website](https://people.math.ethz.ch/~cglanzer/files/bachelor_thesis_cglanzer.pdf).

To run the simulations, you need a recent version of MATLAB, including both the *Optimization Toolbox* and the *Statistics Toolbox*. The file *simulation.m* is a template showing how to run the algorithms on specific functions. The template is self-explaining and well commented. For details, please take a look at *EI.m* and *GPUCB.m*. Each function is commented and has a header which explains the parameters.

