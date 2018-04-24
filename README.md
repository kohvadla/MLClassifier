Based on the `dqews` package written by James Catmore – https://gitlab.cern.ch/jcatmore/dqews

# How to use the classifier package

* Enter the names of all the input features to the classifier in `featuresLists.py` (with event weights as the last feature)

* For BDTs, run
```
python classifiers.py --bdt
```
or for neural networks, run
```
python classifiers.py --nn
```
To list all available command line options, type
```
python classifiers.py -h
```
