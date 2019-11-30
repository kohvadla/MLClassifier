# Implemented classifiers

* Neural Network: Densely connected, feed forward (MLP)
* Boosted Decision Tree (BDT): XGBoost

# Framework and ML libraries

* Framework: scikit-learn API and libraries for pre-processing of data, and for building, fitting and running the classifiers
* Data handling: pandas DataFrame
* Neural network: Keras interface using the TensorFlow backend
* BDT: xgboost
* Visualization: seaborn and matplotlib

# How to use the classifier package

The workflow is implemented in `main.py`, which imports and calls functionality from the scripts in `modules/`.

## Pre-process input dataset and store pandas DataFrame to file

Use the existing functions in `modules/importData.py` to read in and make the appropriate pre-selections of the input dataset in user defined chunks, before storing the resulting pandas DataFrame to an HDF5 file. Then, run
```
main.py --prepare_hdf5
```
f.ex. using the Jupyter notebook `run_package.py`, where you also can run other functionality of the package.
Type
```
main.py -h
```
to see all available command line options.

## Train/run a neural network

Open the Jupyter notebook `run_neural_network.ipynb` and run/modify the relevant cells with the appropriate options.


## Train/run an XGBoost BDT

Open the Jupyter notebook `run_xgboost.ipynb` and run/modify the relevant cells with the appropriate options.
