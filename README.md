# ModelingVisibilityOfSaccadelikeMotion

Simple early-vision model to explain the experimental data presented in Rolfs, Schweitzer, Castet, Watson, Ohl (in prep), doi: TBA, available at URLHERE. 

Have a look at the markdown document [SLM_model_ver1.md](../main/SLM_model_ver1.md) for modeling code and results. For a thorough description, please see the above mentioned manuscript and link. 

It is recommended to run the code if you have a Nvidia graphics card and CUDA installed, as the optimized code uses [torch](https://torch.mlverse.org/) for the convolutions in the visual processing function. 
If you don't have this hardware, no problem, you can set the variable *do_this_on_GPU* to FALSE, it will just take longer then. To experiment with the code, it is recommended to reduce *n_iterations* and skip the computation of the population response *do_compute_pop_mean* to speed up simulations. 
To go directly to analyzing the data without going through the simulations, you can set *do_run_model* to FALSE and load a  previously saved image. For instance, this repository contains two runs of 250 iterations each, for temporal response scales 12.5 and 15, respectively. 

![Example results](../main/SLM_model_ver1_files/figure-gfm/unnamed-chunk-20-1.svg)
