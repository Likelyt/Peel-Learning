# Peel Learning (PL) for Pathway-Related Outcome Prediction

Peel Learning (PL) is a Python implementation of Peel Learning for Pathway-Related Outcome Prediction.

## Version Requirement

•	Pytorch 0.4 or higher

•	Python 3.6 or higher

## Required Packages

The program will install the following packages:

•	NumPy http://www.numpy.org/

## Installation 

```python
python setup.py install
```

## Input Data Formats

Assuming n is the sample size and p is the number of predictors, the program takes 2 required input files:

•	A tab-delimited text file containing predictors and outcome (example_xy.txt): A data matrix with n rows and p+1 columns, outcome Y (either binary or continuous) as the 1st column, and predictors X in the remaining p columns. 

•	A tab-delimited text file containing a directed feature connection matrix (example_adjacency.txt): An adjacency matrix of the features’ directed connections with p rows and p columns. 

•	Both files do not need headers and row names.

## Run PL

```python
python PL_train.py example_x.txt example_y.txt example_adjacency.txt
```

## Hyperparameters Options
Seen in the main.py.

## Reference
@article{li2021peel,
  title={Peel Learning for Pathway-Related Outcome Prediction},
  author={Li, Yuantong and Wang, Fei and Yan, Mengying and Cantu, Edward and Yang, Fan Nils and Rao, Hengyi and Feng, Rui},
  journal={Bioinformatics},
  year={2021}
}











