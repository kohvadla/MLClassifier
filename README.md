Based on the `dqews` package written by James Catmore – https://gitlab.cern.ch/jcatmore/dqews

# Available classifiers

* AdaBoost BDT
* XGBoost BDT
* Densely connected, shallow, feed forward neural network

# How to use the classifier package

Enter the names of all the input features to the classifier in `featureList.py`, possibly including event weight feature

For AdaBoost BDT, run
```
python classifiers.py --adaboost
```
for XGBoost BDT, run
```
python classifiers.py --xgboost
```
or for neural network, run
```
python classifiers.py --nn
```
To list all available command line options, type
```
python classifiers.py -h
```
