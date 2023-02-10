# VITE
a hierarchical VIsualization tool for Tree Ensemble.

This repository contains the source code and the data related to the paper Visualizing Tree Ensembles: from a black forest to a
white optimal tree by Giulia Di Teodoro, Marta Monaci and Laura Palagi, only for the visualization tool for tree ensemble. 
In particular, it contains the code for displaying the VITE plot in case you want to visualize a Random Forest.


## Install

You can clone the repository and install the package in your Python environment as follows:

```bash
git clone https://github.com/gditeodoro/VITE.git
pip install .
```

## Requirements

The file requirements.txt reports the list of packages that must be installed to run the code. You can add a package to your environment via pip or anaconda using either pip install "package" or conda install "package".

## Configuration and running

You just need to run `main.py`. 
The parameters are set according to the experiments in the paper, but you can simply modify them via `main.py`. 
The parameters that can be setted according to your needs are the following:
```
max_depth = ...
test_size= ...
max_features= ...
n_estimators= ...

x_tr, y_tr, x_ts, y_ts,data,scaler = readData("data/...",test_size=test_size)
```
In the folder `data` there are the dataset from UCI ML repository, used for experiments in the paper.

## Results

The output visualizations can be found in folder *figure/*.

## Examples

These are examples of the visualizations obtained in the case of the Cleveland database when training a Random Forest (RF) with maximum depth 3, 100 estimators and giving the RF the possibility to choose among all the possible features when looking for the best split at each node.

<p align="center">
  <img width="540" src="https://github.com/gditeodoro/VITE/blob/main/figure/heat_map.png"/>
</p>

<p align="center">
  <img width="1040" src="https://github.com/gditeodoro/VITE/blob/main/figure/tree_heatmap.png"/>
</p>

## Team

Contributors to this code:

* [Giulia Di Teodoro](https://github.com/gditeodoro)
* [Marta Monaci](https://github.com/m-monaci)

# License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

* [MIT License](https://opensource.org/licenses/mit-license.php)
* Copyright 2022 © Giulia Di Teodoro, Marta Monaci
