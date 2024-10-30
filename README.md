# ModelingVisibilityOfSaccadelikeMotion

This repository contains the source code and compiled markdown documents for the simple early-vision model to explain the experimental data presented in *Lawful kinematics link eye movements to the limits of high-speed perception* by Rolfs, Schweitzer, Castet, Watson, & Ohl. The preprint is available at https://doi.org/10.1101/2023.07.17.549281. 

## Contents
First and foremost, have a look at the markdown document [SLM_model_ver1.md](../main/SLM_model_ver1.md) for modeling code and results, as well as basic documentation. For a thorough description of the model, please see the above mentioned manuscript and link. 
The revised version of the manuscript features a grid-search analysis of model parameters which can be found in the markdown document [SLM_model_ver2.md](../main/SLM_model_ver2.md).

To run the code, simply execute the *.rmd* markdown files and make sure the relevant auxiliary function in this repository are present in the working directory. Moreover, you need the specification of the stimulus trajectory from one of the original experimental design files -- see *SLMFAG01_Block1.mat* -- also included in this repository, but, in principle, any trajectory can be supplied to the model.
The repository also contains output from two runs of 250 iterations each, for temporal response scales 12.5 and 15, respectively. Because of their larger file size, grid-search results can only be found at https://osf.io/89xmd/. 

## Requirements and instructions
The code was exectued with *R* version 4.4.1 on Kubuntu 22.04 (with *r-base* and dependencies installed from its official repositories), using a custom-built desktop computer with a Intel i9-10900K CPU, 64 GB working memory, and an Nvidia GeForce RTX 3060 graphics card (with CUDA 11.7 installed). See [run_on_this_R_version.txt](../main/run_on_this_R_version.txt) for the list of *R* packages and their versions used for compiling the markdown files. Markdown was compiled within *RStudio* version 2024.04.2. For installation instructions, please see the official maintainer documentation.

It is recommended to run the code if you have a Nvidia graphics card and CUDA installed, as the optimized code uses [torch](https://torch.mlverse.org/) for the convolutions in the visual processing function. 
If you don't have this hardware, no problem, you can set the variable *do_this_on_GPU* to FALSE, it will just take longer then. To experiment with the code, it is recommended to reduce *n_iterations* and skip the computation of the population response *do_compute_pop_mean* to speed up simulations. Depending on the number of iterations and the hardware used, simulations can take several days. 
To go directly to analyzing the data without going through the simulations, you can set *do_run_model* to FALSE and load a  previously saved image (e.g., from *visual_proc_on_SLMF_rev_sca12.5.rda*). 


![Example results](../main/SLM_model_ver1_files/figure-gfm/unnamed-chunk-20-1.svg)
