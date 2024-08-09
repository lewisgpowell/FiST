# Fitting Superconducting Tunnelling (FiST)

This code is designed to fit theoretical models to superconducting tunnelling spectra.
The mathematical models are implemented in C++ with a python interface.
The curve fitting is then done in python based on the nlopt library for nonlinear optimization.

## Installation (10 minutes)
I have only tested the software tested on Windows for which binaries are provided.
However, it should be possible to compile the C++ code from source on other systems.

Clone the git repository

    git clone https://github.com/lewisgpowell/fist
    cd fist

Install python dependencies using conda https://conda.io/projects/conda/en/latest/user-guide/install/index.html

    conda create -n fist python=3.10
    conda activate fist
    conda install -c conda-forge nlopt numpy pandas

## Demo
Try the demo using some real data for a tunnel barrier in a parallel magnetic
field with the 'Maki model'
([K. Maki et al. 1964](https://academic.oup.com/ptp/article/32/1/29/1834620))
This takes about 20 minutes to run on a standard office computer.

    python demo_fits.py
