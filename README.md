# Collapsar formalism

This repository contains the data required to reproduce the figures in the paper titled: _The effect of stellar rotation on black hole mass and spin_.

- The `Python_scripts` directory contains the `Python` files that were used to model the collapsar evolution. To run, they will require the `profile<number>.data`  file from MESA as an input.

- The `Collapsar dynamics - Section 2.2.ipynb` is a <a href="https://sagemanifolds.obspm.fr/" target="_blank">`SAGE`</a> notebook that contains the symbolic implementation of some of the calculations performed in Section 2.2 of the paper.

 - The `Plots_Stellar_rotation_and_collapsars.ipynb` file contains the code for generating those plots that consider fiducial values of the assumed free parameters such as $s, \alpha$ and  $\tilde{\eta}$.

 - The `Plots_Stellar_rotation_and_collapsars-Error_analysis.ipynb` file contains the code for generating fits for the black hole mass and spin spectrum. 

 - The `Plots_Stellar_rotation_and_collapsars-Non-fiducial.ipynb` file contains the code for those plots in the paper assume non-fiducial values for the above defined parameters. 

 - The figures will be saved in the `figures` directory.

 - Output data from the `MESA` models in stored in the `data` directory.

 - The `MESA_inlist` directory contains the `MESA` inlists used for generating helium star models. The structure of these inlist have been adapted from other works in the literature and the `MESA` `test_suites`.
 

