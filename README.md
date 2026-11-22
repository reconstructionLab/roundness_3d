# Grain roundness computation in 3D

This repository contains the code for both a new 3D roundness metric and the associated analyses, which we present in our paper (see citation below). This roundness metric is applicable for analyses in the Earth sciences (sedimentary geology, geomorphology) and engineering disciplines.

We include our new 3D roundness metric as well as implementations of other 3D roundness metrics (e.g., Wadell). We also include a workflow for processing CT imagery of grains and algorithms for generating the synthetic grains in our paper.

# Organization

src/ contains .py files
analyses/ contains folders for each figure in the paper, within which there are .ipynb files where we conducted analyses and generated plots

# How to use

src/roundness_3d.py contains functions for computing the 3D roundness metrics we implement, including a new method, std_curvature(). For most metric functions, you can pass either a binary volume (from which we construct a mesh) or a mesh (as vertices and faces).

## Dependencies

See requirements.txt, which lists the versions of all packages I used. Other versions of dependencies may work as well.

# Paper

This work is described in our paper, which you can cite as follows:

Yuan, H.D.A., Pamukcu, A., Meisenheimer, D., Keller, C.B., Strauss, J.V., Mehra, A.K. A curvature-based approach to measuring 3D grain roundness from low-resolution data. Submitted to The Journal of Sedimentary Research.