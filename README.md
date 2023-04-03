# Mechanistic-models-in-mycology
> Tara Hameed.

Simulation study of fitting **Mechanistic models** to different experimental designs **in mycology**.

## Table of contents
* [Introduction](#introduction)
* [Technologies](#technologies)
* [Setup](#setup)
* [Usage](#usage)
* [References](#references)
* [Acknowledgements](#acknowledgements)

## Introduction
This code can implement a simulation study that follows previously proposed statistical methods (Morris *et al*., Talts *et al.*) to quantify expected model fitting using parameter estimates' expected coverage and model predictions given a data set. The simulation study is repeatedly applied to a mathematical model of early-stage invasive aspergillosis (pulmonary lung infection) based on a model by (Tanaka *et al.*) and potential data that could be obtained by different experimental designs.

We considered the following 5 fake data sets:

1. Fake data that mimics the currently available real data (collated by Natasha Motsi, summarised in "data.csv").
2. Fake data that mimics a conventional experimental design.
3. Fake data that reflects the currently available real data (as in 1.) and the added fake dataset of the conventional experimental design (as in 2.).
4. Fake data that reflects the currently available real data (as in 1.) and the added fake dataset of the conventional experimental design with a model directed change. In this case, Neutrophils are measured in the experimental design, where they previously weren't.
5. Fake data that reflects the currently available real data (as in 1.) and the added fake dataset of the conventional experimental design with a model directed change. In this case, Neutrophils are measured in the experimental design, where they previously weren't, and samples are taken at 12, 32 and 72 hours instead of 24, 48 and 72 hours.

## Technologies
The code was developed using the coding language Julia v1.3.1 with the following package dependencies that can be found in [julia-packages.txt](julia-packages.txt). More information on managing julia packages can be found [here](https://pkgdocs.julialang.org/v1/managing-packages/).

## Setup
First either clone or download the repository to your machine.

Then, in order to run this project, you will need to install Julia with the packages listed in the [technologies](#technologies) section. Then copy the conda environment using the below command:

```
$ conda env create -f environment.yml
```

Or copy the conda environment using [spec-file.txt](spec-file.txt), following the instructions found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#cloning-an-environment).

## Usage

The [`scripts`](scripts) folder contains all of the files needed to either perform a simulation study on fake data or to fit to the real data. All of the scripts are intended to be run using the Imperial HPC, with corresponding submission scripts that should be created in a "run" folder.

In the scripts folder, the [`fake-data`](scripts/fake-data/) folder contains:
* [`pick-params.jl`](scripts/fake-data/pick-params.jl): script that picks 5 parameter sets for the analysis to be run for and saves them in the file [`parameters.csv`](data/parameters.csv). The parameter sets need result in a solvable ODE and ensure the healthy steady state of the ODE is stable.
* `fake-data-i.jl` for i in 1 to 5. These are scripts that generate fake data sets of the form described in the [Introduction](#introduction). When running these scripts they produce the data found in the [`fake-data`](fake-data) folder. Users will have to define a new system of ODEs to run this and subsequent scripts.
* [`fit-data`](scripts/fake-data/fit-data) folder that contains scripts `fit-data-i.jl` for i in 1 to 5, correpsonding to the fake data sets 1 to 5 listed in the [Introduction](#introduction) section. These scripts when run will fit the ODE to the 5 fake data sets for varying noise levels and for a specified [SciPy](https://docs.scipy.org/doc/scipy/reference/optimize.html) optimisation algorithm; either [Basin Hopping](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping), [Differential Evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution) or vanilla [Least Squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares).
* [`bootstrap`](scripts/fake-data/bootstrap/): contains scripts that are designed to calculate the bootstrap confidence intervals of the fit to the data.
* [`fim`](scripts/fake-data/sim/): contains scripts that are designed to calculate the Fisher Information Matrix of the 5 experimental designs listed in the [Introduction](#introduction) section.
* [`profile-likelihood`](scripts/fake-data/profile-likelihood/): contains scripts that are designed to calculate the profile confidence intervals of the fit to the data.

The [`real-data`](scripts/real-data/) folder contains:
* Files to read in and collate the real data that are contained in separate excel files in the [`data`](data) folder into a joint "data.csv" file, e.g. [`read-in-real-data.jl`](scripts/real-data/read-in-real-data.jl).


## References

- Morris, Tim P., Ian R. White, and Michael J. Crowther. "Using simulation studies to evaluate statistical methods." *Statistics in medicine* 38.11 (2019): 2074-2102.

- Talts, Sean, et al. "Validating Bayesian inference algorithms with simulation-based calibration." *arXiv preprint* arXiv:1804.06788 (2018).

- Tanaka, Reiko J., et al. "In silico modeling of spore inhalation reveals fungal persistence following low dose exposure." *Scientific reports* 5.1 (2015): 1-14.

## Acknowledgements

This work was supported by The Wellcome Trust [215358/Z/19/Z] and the National Centre for the Replacement Refinement & Reduction of Animals in Research (NC3Rs).
