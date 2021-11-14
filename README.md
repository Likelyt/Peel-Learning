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

## Hyperparameters Options
Seen in the main_real.py.

Some examples: iteration = 100, lr = 1, layers = 5.

The default parameters and hyperparameter are set as below:

### File save path
•	out_dir='/home/li3551/stat/bioinformatics/code/data' [Put your directory here]

### Optimizer setup
•	optimizer='SGD'

•	beta_1=0.9

•	beta_2=0.999 

• decay_rate=0.975

• momentum=0.9

• end_epoch=1

• eps=1e-08

• iteration=200

• np_random_seed=12345

• lr=1.0

### PL algorithm setup
•	coefficient_share='True'

•	format_type='residual'

•	layers=5

### Objective function setup
• objective='classification'

•	criteria='C_E'

•	eval_metric='cross_entropy'

•	balance_data='balance'

•	balance_weight=0.25

## Run PL
Before you run the code, pls change line 14 and 15 to your own directory.

sys_path = 'xxx'

os.chdir(sys_path)

sys.path.append(sys_path)

### Provide you data path
x_path = 'x_path', eg:'data/brca/x_brca.txt'

y_path = 'y_path', eg:'data/brca/y_brca.txt'

adj_path = 'adj_path', eg:'data/brca/adj_brca.txt'

### For example
```python
python main_real.py -task real-data -data_x_path x_path -data_y_path y_path -adj_matrix_path adj_path -iteration 100 -lr 1 -layers 5
```

## Reference
Pls consider cite our [paper](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btab402/6286960):
```
@article{li2021peel,
  title={Peel Learning for Pathway-Related Outcome Prediction},
  author={Li, Yuantong and Wang, Fei and Yan, Mengying and Cantu, Edward and Yang, Fan Nils and Rao, Hengyi and Feng, Rui},
  journal={Bioinformatics},
  year={2021}
}
```










