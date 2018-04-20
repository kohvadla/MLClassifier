Based on the `dqews` package written by James Catmore – https://gitlab.cern.ch/jcatmore/dqews

# How to use the classifier package

## `featuresLists.py`
* Enter the names of all the input features to the classifier (with event weights as the last feature)

## `classifiers.py`
* For BDTs, run

`python classifiers.py --bdt`

* Or for neural networks, run

`python classifiers.py --nn`

* To list all available command line options, type

`python classifiers.py -h`
