Based on the `dqews` package written by James Catmore – https://gitlab.cern.ch/jcatmore/dqews

# How to use the classifier package

* Enter the names of all the input features to the classifier in `featureList.py`, possibly including event weight feature

* Available classifiers: 
  * For AdaBoost BDT, run
```
python classifiers.py --adaboost
```
  * For XGBoost BDT, run
```
python classifiers.py --xgboost
```
  * Or for neural network, run
```
python classifiers.py --nn
```
* To list all available command line options, type
```
python classifiers.py -h
```
