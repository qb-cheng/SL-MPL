# Overview

This repository contains codes and data for reproducing the results in the paper

### The effects of limited sampling and genetic drift on selection coefficient estimation from genetic time-series data
doi:

Qingbei Cheng<sup>1,+</sup>, Muhammad Saqib Sohail<sup>2,+,*</sup>, and Matthew R. McKay<sup>3,4,#</sup>

<sup>1</sup> Department of Electronic and Computer Engineering, Hong Kong University of Science and Technology, Hong Kong SAR, China.

<sup>2</sup> Department of Computer Science, Bahria University, Lahore, Pakistan.

<sup>3</sup> Department of Electrical and Electronic Engineering, University of Melbourne, Melbourne, Victoria, Australia.

<sup>4</sup> Department of Microbiology and Immunology, University of Melbourne, at The Peter Doherty Institute for Infection and Immunity, Melbourne, Victoria, Australia.

<sup>#</sup> Correspondence to [matthew.mckay@unimelb.edu.au](mailto:matthew.mckay@unimelb.edu.au) and [saqibsohail.bulc@bahria.edu.pk](mailto:saqibsohail.bulc@bahria.edu.pk)

# Contents

The codes were written in Python, with the required environment specifications provided in "environment.yml."

The scripts for generating and analyzing simulation data are identified by their respective file names.

To generate evolutionary trajectories under the Wright-Fisher model, first define a set ID within the evolutionary parameters in "GetCaseArg.py" and then execute "EvoGen.py" using that set ID. The simulated Wright-Fisher trajectories should be stored in the "PopRecords" folder.

Results related to the MPL-based estimator can subsequently be produced for the main and supplementary figures.

To obtain estimates using the Hidden Markov Model (HMM) framework, which serves as an extension of MPL, execute "MPL_HMM.py" initially.

Example figures are located in the "Figure" folder, while example HMM-based estimates can be found in the "ModelComp" folder.

# License

This project is under the MIT License.
