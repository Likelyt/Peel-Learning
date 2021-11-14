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

•	A tab-delimited text file containing predictors X (example_x.txt): A data matrix with p rows and n columns, 

• A tab-delimited text file containing outcomes Y (example_y.txt) (either binary or continuous): A data vector with n rows.

•	A tab-delimited text file containing a directed feature connection matrix (example_adjacency.txt): An adjacency matrix of the features’ directed connections with p rows and p columns. 

•	All files do not need headers and row names.

## Run PL

```python
python main_real.py -task real-data -data_x_path x_path -data_y_path y_path -adj_matrix_path adj_path -iteration 100 -lr 1 -layers 5
```

## Hyperparameters Options
Seen in the main_real.py.

## Reference
We have a [paper](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btab402/6286960) you can cite:
```
@article{li2021peel,
  title={Peel Learning for Pathway-Related Outcome Prediction},
  author={Li, Yuantong and Wang, Fei and Yan, Mengying and Cantu, Edward and Yang, Fan Nils and Rao, Hengyi and Feng, Rui},
  journal={Bioinformatics},
  year={2021}
}
```










