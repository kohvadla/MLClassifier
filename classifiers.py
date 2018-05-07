from __future__ import division

import sys
import time
import argparse
import pickle
import itertools

import h5py

import numpy as np
import pandas as pd

import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import preprocessing, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report, make_scorer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from featureList import features 

import logging

logging.basicConfig(
                    format="%(message)s",
                    level=logging.DEBUG,
                    stream=sys.stdout)

###############################################
# MAIN PROGRAM

t_start = time.time()

runBDT = False
runXGBoost = False
runNN = False
runTraining = True
use_event_weights = True
use_class_weights = True
log_y = True
doGridSearchCV = False
plot_validation_curve = False
plot_learning_curve = False

parser = argparse.ArgumentParser(description='Run classifier -- BDT or neural network')
group = parser.add_mutually_exclusive_group()
group.add_argument('-a', '--adaboost', action='store_true', help='Run adaptive BDT')
group.add_argument('-x', '--xgboost', action='store_true', help='Run gradient BDT')
group.add_argument('-n', '--nn', action='store_true', help='Run neural network')
parser.add_argument('-t', '--no_training', action='store_false', help='Do not train the selected classifier')
parser.add_argument('-e', '--no_event_weights', action='store_false', help='Do not apply event weights to training examples')
parser.add_argument('-c', '--no_class_weights', action='store_false', help='Undersample dominant class to balance dataset, instead of applying class weights to all available training examples')
parser.add_argument('-y', '--no_log_y', action='store_false', help='Do not use log scale on y-axis')
parser.add_argument('-g', '--doGridSearchCV', action='store_true', help='Perform tuning of hyperparameters using k-fold cross-validation')
parser.add_argument('-v', '--validation_curve', action='store_true', help='Plot validation curve')
parser.add_argument('-l', '--learning_curve', action='store_true', help='Plot learning curve')

args = parser.parse_args()

if args.adaboost:
    runBDT = args.adaboost
elif args.xgboost:
    runBDT = args.xgboost
    runXGBoost = args.xgboost
elif args.nn:
    runNN = args.nn
elif (args.adaboost and args.xgboost) or (args.adaboost and args.nn) or (args.xgboost and args.nn):
    runBDT = True
    runXGBoost = False
    runNN = False
    print "\nINVALID CLASSIFIER CHOICE! Multiple classifiers specified. Running AdaBoost BDT as default."
else:
    sys.exit("Classifier argument not given! Choose either -a for AdaBoost BDT, -x for XGBoost BDT or -n for neural network.")
if not(args.no_training):
    runTraining = args.no_training
if not(args.no_event_weights):
    use_event_weights = args.no_event_weights
if not(args.no_class_weights):
    use_class_weights = args.no_class_weights
if not(args.no_log_y):
    log_y = args.no_log_y
if args.doGridSearchCV:
    doGridSearchCV = args.doGridSearchCV
if args.validation_curve:
    plot_validation_curve = args.validation_curve
if args.learning_curve:
    plot_learning_curve = args.learning_curve


# Build background arrays
bkgFile = h5py.File("../ewk/hdf5_files/2L_bkg_flat_ext.h5","r")
X_bkg_dset = bkgFile['FlatTree'][:]  # structured array from the FlatTree dataset

structured_array_features = list(X_bkg_dset.dtype.names)
deselectedFeatures = ['n_bjets', 'met_phi', 'dsid', 'event_weight']

in_selectedFeatures = np.array([i not in deselectedFeatures for i in structured_array_features])
selectedFeatures = ( np.array(structured_array_features)[in_selectedFeatures] ).tolist()
n_selectedFeatures = len(features) - len(deselectedFeatures)
print "len(structured_array_features)",len(structured_array_features)
print "in_selectedFeatures.shape",in_selectedFeatures.shape
print "\nselectedFeatures =", selectedFeatures 
print "n_selectedFeatures =", n_selectedFeatures 

X_bkg_sel_arr = np.array( X_bkg_dset[selectedFeatures].tolist() )
X_bkg_ew_arr = np.array( X_bkg_dset['event_weight'].tolist() )

print "before imp: np.any(X_bkg_sel_arr == -999)",np.any(X_bkg_sel_arr == -999)
bkgFile.close()

# Build signal arrays
sigFile = h5py.File("../ewk/hdf5_files/2L_sig_flat_ext.h5","r")
X_sig_dset = sigFile['FlatTree'][:]
X_sig_sel_arr = np.array( X_sig_dset[selectedFeatures].tolist() )
X_sig_ew_arr = np.array( X_sig_dset['event_weight'].tolist() )
sigFile.close()

seed = 42

if use_class_weights:
    class_weight = 'balanced'
    n_samples_bkg = None
else:
    class_weight = None
    n_samples_bkg = X_sig_sel_arr.shape[0]

X_sig_sel_shuffled = shuffle(X_sig_sel_arr, random_state=seed, n_samples=None)
X_sig_ew_shuffled = shuffle(X_sig_ew_arr, random_state=seed, n_samples=None)
X_bkg_sel_shuffled = shuffle(X_bkg_sel_arr, random_state=seed, n_samples=n_samples_bkg)
X_bkg_ew_shuffled = shuffle(X_bkg_ew_arr, random_state=seed, n_samples=n_samples_bkg)
print "before imp: np.any(X_bkg_shuffled == -999)",np.any(X_bkg_sel_shuffled == -999)

X = np.concatenate((X_bkg_sel_shuffled, X_sig_sel_shuffled), 0)
event_weights = np.concatenate((X_bkg_ew_shuffled, X_sig_ew_shuffled), 0)
print "before imp: np.any(X == -999)",np.any(X == -999)

# Make array of labels
y_bkg = np.zeros(X_bkg_sel_shuffled.shape[0])
y_sig = np.ones(X_sig_sel_shuffled.shape[0])
y = np.concatenate((y_bkg, y_sig),0)
y = np.ravel(y)

classes = np.unique(y)
class_weight_vect = compute_class_weight(class_weight, classes, y)
class_weight_dict = {0: class_weight_vect[0], 1: class_weight_vect[1]}
scale_pos_weight = len(y)/np.sum(y)
print "class_weight_vect",class_weight_vect
print "class_weight_dict",class_weight_dict
print "scale_pos_weight",scale_pos_weight

# Replane -999
imp = Imputer(missing_values=-999, strategy='mean', axis=0)
imp.fit_transform(X)
print "after imp: np.any(X == -999)",np.any(X == -999)

# Split dataset in train and test sets
test_size=0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=None) #, shuffle=True
y_test = y_test.astype(int) # convert labels from float to int
event_weights_train, event_weights_test, y_ew_train, y_ew_test = train_test_split(event_weights, y, test_size=test_size, 
                                                                                    random_state=seed, stratify=None) #, shuffle=True
y_ew_test = y_ew_test.astype(int) # convert labels from float to int

print ""
print "Number of background examples for training =", y_train[y_train==0].shape[0]  
print "Number of background examples for testing =", y_test[y_test==0].shape[0]  
print "Number of signal examples for training =", y_train[y_train==1].shape[0]  
print "Number of signal examples for testing =", y_test[y_test==1].shape[0]  


# Feature scaling

min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

if use_event_weights:
    sample_weight = event_weights_train
    print "\nINFO  Applying event weights to the examples during training"
else:
    sample_weight = None
    print "\nINFO  Not applying event weights to the examples during training"

if doGridSearchCV:
    print "INFO  Performing grid search for hyperparameters"
if plot_validation_curve:
    print "INFO  Plotting validation curve"
if plot_learning_curve:
    print "INFO  Plotting learning curve"

train_scores_vc_mean = train_scores_vc_std = 0
valid_scores_vc_mean = valid_scores_vc_std = 0
train_scores_lc_mean = train_scores_lc_std = 0
valid_scores_lc_mean = valid_scores_lc_std = 0
train_sizes = [0.5, 0.75, 1.0]

# BDT TRAINING AND TESTING

n_estimators = [10, 100, 1000]
learning_rate = [0.1, 1.0, 10.0]

param_range_bdt = n_estimators
param_name_bdt = "n_estimators"

global model

if runBDT:

    if runTraining:

        if runXGBoost:
            model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective="binary:logistic", scale_pos_weight=scale_pos_weight)

            if doGridSearchCV:
                tuned_parameters = [{'n_estimators': n_estimators, 'learning_rate': learning_rate}]
                model = GridSearchCV( model, tuned_parameters, cv=3, scoring=None )

        else: # run AdaBoost
            model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, class_weight=class_weight_dict), n_estimators=100, learning_rate=1.0)

            if doGridSearchCV:
                tuned_parameters = [{'base_estimator': [DecisionTreeClassifier(max_depth=1, class_weight=class_weight_dict)], 
                                    'n_estimators': n_estimators, 'learning_rate': learning_rate}]
                model = GridSearchCV( model, tuned_parameters, cv=3, scoring=None )

        params = model.get_params()
        print "\nmodel.get_params()",params

        print "\nBuilding and training BDT"

        model.fit(X_train,y_train,sample_weight=sample_weight)

        if plot_validation_curve:
            train_scores, valid_scores = validation_curve(model, X_train, y_train, param_name=param_name_bdt, param_range=param_range_bdt, 
                                                            cv=3, scoring=None, n_jobs=1, verbose=0)
            train_scores_vc_mean = np.mean(train_scores, axis=1)
            train_scores_vc_std = np.std(train_scores, axis=1)
            valid_scores_vc_mean = np.mean(valid_scores, axis=1)
            valid_scores_vc_std = np.std(valid_scores, axis=1)

        if plot_learning_curve:
            train_sizes, train_scores, valid_scores = learning_curve(model, X_train, y_train, train_sizes=train_sizes, 
                                                                        cv=3, scoring=None, n_jobs=1, verbose=0)
            train_scores_lc_mean = np.mean(train_scores, axis=1)
            train_scores_lc_std = np.std(train_scores, axis=1)
            valid_scores_lc_mean = np.mean(valid_scores, axis=1)
            valid_scores_lc_std = np.std(valid_scores, axis=1)

        joblib.dump(model, 'bdt_AC18.pkl')

    if not runTraining:
        print "\nReading in pre-trained BDT"
        model = joblib.load('bdt_AC18.pkl')

    # Training scores
    pred_train = model.predict(X_train)
    if runXGBoost: output_train = model.predict_proba(X_train)[:,1]
    else: output_train = model.decision_function(X_train)

    # Validation scores
    pred_test = model.predict(X_test)
    if runXGBoost: output_test = model.predict_proba(X_test)[:,1]
    else: output_test = model.decision_function(X_test)


# NEURAL NETWORK TRAINING AND TESTING
# Set up neural network

batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 100, 1000]

param_range_nn = epochs
param_name_nn = "epochs"

if runNN:

    def create_model():
        model = Sequential()
        model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    if runTraining:
        print "\nBuilding and training neural network"

        if doGridSearchCV:
            model = KerasClassifier(build_fn=create_model, verbose=0)

            param_grid = dict(batch_size=batch_size, epochs=epochs)
            model = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

            grid_result = model.fit(X_train, y_train, epochs=100, batch_size=100, sample_weight=sample_weight, class_weight=class_weight_dict)

        elif not doGridSearchCV:
            if plot_validation_curve:
                model = KerasClassifier(build_fn=create_model, verbose=0)
            else:
                model = Sequential()
                model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation="relu"))
                model.add(Dense(1, activation="sigmoid"))
                model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

            model.fit(X_train, y_train, epochs=100, batch_size=100, sample_weight=sample_weight, class_weight=class_weight_dict)

            if plot_validation_curve or plot_learning_curve:
                if plot_validation_curve:
                    train_scores, valid_scores = validation_curve(model, X_train, y_train, param_name=param_name_nn, param_range=param_range_nn, 
                                                                    cv=3, scoring="accuracy", n_jobs=1, verbose=0)
                    train_scores_vc_mean = np.mean(train_scores, axis=1)
                    train_scores_vc_std = np.std(train_scores, axis=1)
                    valid_scores_vc_mean = np.mean(valid_scores, axis=1)
                    valid_scores_vc_std = np.std(valid_scores, axis=1)

                if plot_learning_curve:
                    train_sizes, train_scores, valid_scores = learning_curve(model, X_train, y_train, train_sizes=train_sizes, 
                                                                                cv=3, scoring=None, n_jobs=1, verbose=0)
                    train_scores_lc_mean = np.mean(train_scores, axis=1)
                    train_scores_lc_std = np.std(train_scores, axis=1)
                    valid_scores_lc_mean = np.mean(valid_scores, axis=1)
                    valid_scores_lc_std = np.std(valid_scores, axis=1)

            else:
                model.save("nn_AC18.h5")

    elif not runTraining:
        print "Reading in pre-trained neural network"
        model = load_model("nn_AC18.h5")

    # Get class and probability predictions
    if doGridSearchCV or plot_validation_curve or plot_learning_curve:
        # Training
        pred_train = model.predict(X_train)
        output_train = model.predict_proba(X_train)[:,1]

        # Testing
        pred_test = model.predict(X_test)
        output_test = model.predict_proba(X_test)[:,1]
    else:
        # Training
        pred_train = model.predict_classes(X_train,batch_size=100)
        output_train = model.predict_proba(X_train,batch_size=100)
        # Testing
        pred_test = model.predict_classes(X_test,batch_size=100)
        output_test = model.predict_proba(X_test,batch_size=100)


pred_train = np.ravel(pred_train)
output_train = np.ravel(output_train)
pred_test = np.ravel(pred_test)
output_test = np.ravel(output_test)

# Print results of grid search
if doGridSearchCV:
    print "Best parameters set found on development set:"
    print ""
    print "model.best_params_", model.best_params_
    print ""
    print "Grid scores on development set:"
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print "{0:0.3f} (+/-{1:0.03f}) for {2!r}".format(mean, std, params)
    print ""
    df = pd.DataFrame.from_dict(model.cv_results_)
    print "pandas DataFrame of cv results"
    print df
    print ""

    print "Detailed classification report:"
    print ""
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print ""
    y_true, y_pred = y_test, model.predict(X_test)
    print classification_report(y_true, y_pred)

# Define name of output file
if runXGBoost: 
    output_filename = 'plots/xgboost'
elif runBDT: 
    output_filename = 'plots/adaboost'
elif runNN: 
    output_filename = 'plots/nn'
output_filename += '_AC18_ext_shuffled'
if use_event_weights:
    output_filename += '_ew'
if plot_validation_curve:
    output_filename += '_vc'
if plot_learning_curve:
    output_filename += '_lc'
output_filename += '.pdf'
pdf_pages = PdfPages(output_filename)
np.set_printoptions(threshold=np.nan)

# Plotting - probabilities
figA, axsA = plt.subplots()
axsA.set_ylabel("Events normalized to one")
if runXGBoost: 
    bins = np.linspace(0.0, 1.0, 30)
    axsA.set_title("XGBoost BDT")
    axsA.set_xlabel("Signal probability")
elif runBDT: 
    bins = np.linspace(-1.0, 1.0, 60)
    axsA.set_title("AdaBoost BDT")
    axsA.set_xlabel("BDT score")
elif runNN: 
    bins = np.linspace(0.0, 1.0, 30)
    axsA.set_title("Neural network")
    axsA.set_xlabel("Signal probability")
# Plot training output
axsA.hist(output_train[y_train==0], bins, alpha=0.2, histtype='stepfilled', facecolor='blue', label='Background trained', normed=True)
axsA.hist(output_train[y_train==1], bins, alpha=0.2, histtype='stepfilled', facecolor='red', label='Signal trained', normed=True)
# Plot test output
axsA.hist(output_test[y_test==0], bins, alpha=1, histtype='step', linestyle='--', edgecolor='blue', label='Background tested', normed=True)
axsA.hist(output_test[y_test==1], bins, alpha=1, histtype='step', linestyle='--', edgecolor='red', label='Signal tested', normed=True)
if log_y: axsA.set_yscale('log') #, nonposy='clip')
axsA.legend(loc="best")
pdf_pages.savefig(figA)

# Plotting probabilities with event weights
figA2, axsA2 = plt.subplots()
axsA2.set_ylabel("Weighted events normalized to one")
if runXGBoost: 
    bins = np.linspace(0.0, 1.0, 30)
    axsA2.set_title("XGBoost BDT")
    axsA2.set_xlabel("Signal probability")
elif runBDT: 
    bins = np.linspace(-1.0, 1.0, 60)
    axsA2.set_title("AdaBoost BDT")
    axsA2.set_xlabel("BDT score")
elif runNN: 
    bins = np.linspace(0.0, 1.0, 30)
    axsA2.set_title("Neural network")
    axsA2.set_xlabel("Signal probability")
# Plot training output
acceptance_bkg = X_sig_sel_arr.shape[0]/X_bkg_sel_arr.shape[0]
sf_bkg = 1.#/acceptance_bkg
sf_sig = 1.
print "acceptance_bkg",acceptance_bkg
print "y_train.shape",y_train.shape
print "output_train[y_train==0].shape",output_train[y_train==0].shape
print "event_weights_train[y_train==0].shape",event_weights_train[y_train==0].shape
axsA2.hist(output_train[y_train==0], bins, weights=event_weights_train[y_train==0]*sf_bkg, alpha=0.2, histtype='stepfilled', facecolor='blue', label='Background trained', normed=True)
axsA2.hist(output_train[y_train==1], bins, weights=event_weights_train[y_train==1]*sf_sig, alpha=0.2, histtype='stepfilled', facecolor='red', label='Signal trained', normed=True)
# Plot test output
axsA2.hist(output_test[y_test==0], bins, weights=event_weights_test[y_test==0]*sf_bkg, alpha=1, histtype='step', linestyle='--', edgecolor='blue', label='Background tested', normed=True)
axsA2.hist(output_test[y_test==1], bins, weights=event_weights_test[y_test==1]*sf_sig, alpha=1, histtype='step', linestyle='--', edgecolor='red', label='Signal tested', normed=True)
if log_y: axsA2.set_yscale('log') #, nonposy='clip')
axsA2.legend(loc="best")
pdf_pages.savefig(figA2)


# Plotting - performance curves
# ROC
fpr, tpr, thresholds = roc_curve(y_test, output_test, pos_label=1)
auc = roc_auc_score(y_test, output_test)
figB, axB1 = plt.subplots()
#axB1,axB2 = axsB.ravel()
axB1.plot(fpr, tpr, label='ROC curve')
axB1.plot([0, 1], [0, 1], 'k--')
axB1.set_xlim([0.0, 1.0])
axB1.set_ylim([0.0, 1.05])
axB1.set_xlabel('False Signal Rate')
axB1.set_ylabel('True Signal Rate')
if runXGBoost: axB1.set_title('XGBoost BDT')
elif runBDT: axB1.set_title('AdaBoost BDT')
elif runNN: axB1.set_title('Neural network')
axB1.text(0.4,0.2,"AUC = %.4f" % auc,fontsize=15)
pdf_pages.savefig(figB)

# BDT
# Variable importances
if runBDT and not (doGridSearchCV or plot_validation_curve):
    y_pos = np.arange(X_train.shape[1])
    figC, axC1 = plt.subplots(1,1)
    axC1.barh(y_pos, 100.0*model.feature_importances_, align='center', alpha=0.4)
    #axC1.set_ylim([0,n_selectedFeatures])
    axC1.set_yticks(y_pos)
    axC1.set_yticklabels(np.array(structured_array_features)[in_selectedFeatures],fontsize=5)
    axC1.set_xlabel('Relative importance, %')
    axC1.set_title("Estimated variable importance using outputs (BDT)")
    plt.gca().invert_yaxis()
    pdf_pages.savefig(figC)

# NEURAL NETWORK
# Assess variable importance using weights method
if runNN and not (doGridSearchCV or plot_validation_curve):
    weights = np.array([])
    for layer in model.layers:
        if layer.name =="dense_1":
            weights = layer.get_weights()[0]
    # Ecol. Modelling 160 (2003) 249-264
    sumWeights = np.sum(np.absolute(weights),axis=0)
    Q = np.absolute(weights)/sumWeights
    R = 100.0 * np.sum(Q,axis=1) / np.sum(np.sum(Q,axis=0))
    y_pos = np.arange(X_train.shape[1])
    figC, axC = plt.subplots()
    axC.barh(y_pos, R, align='center', alpha=0.4)
    #axC.set_ylim([0,n_selectedFeatures])
    axC.set_yticks(y_pos)
    axC.set_yticklabels(np.array(structured_array_features)[in_selectedFeatures],fontsize=5)
    axC.set_xlabel('Relative importance, %')
    axC.set_title('Estimated variable importance using input-hidden weights (ecol.model)')
    plt.gca().invert_yaxis()
    pdf_pages.savefig(figC)

# Function taken from scikit-learn website
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    #ax = plt.subplot(111)
    #im = ax.
    #im = 
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes('right', size="5%", pad=0.05)
    plt.title(title)
    #plt.colorbar(im, use_gridspec=True) #cax=cax)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Plot confusion matrices for training and test performance
class_names = ['Background', 'Signal']

#if not doGridSearchCV:
# Compute confusion matrix for training
cnf_matrix_train = confusion_matrix(y_train, pred_train)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix for training
print "\n----- TRAINING -----"
figD, (axsD1, axsD2) = plt.subplots(2,1)
#figD = plt.figure()
plt.suptitle('Confusion matrices for training set')
plt.subplot(1,2,1)
plot_confusion_matrix(cnf_matrix_train, classes=class_names,
                      title='Counts')
                      #title='Confusion matrix, without normalization')

# Plot normalized confusion matrix for training
print ""
plt.subplot(1,2,2)
plot_confusion_matrix(cnf_matrix_train, classes=class_names, normalize=True,
                      title='Normalized')
                      #title='Normalized confusion matrix')

pdf_pages.savefig(figD)


# Compute confusion matrix for test
cnf_matrix_test = confusion_matrix(y_test, pred_test)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix for test
print "\n----- TESTING -----"
figE, (axsE1, axsE2) = plt.subplots(1,2)
plt.suptitle('Confusion matrices for test set')
plt.subplot(1,2,1)
plot_confusion_matrix(cnf_matrix_test, classes=class_names,
                      title='Counts')
                      #title='Confusion matrix, without normalization')

# Plot normalized confusion matrix for test
print ""
plt.subplot(1,2,2)
plot_confusion_matrix(cnf_matrix_test, classes=class_names, normalize=True,
                      title='Normalized')
                      #title='Normalized confusion matrix')

pdf_pages.savefig(figE)

if plot_validation_curve:
    # Plot validation curves
    figF, axsF = plt.subplots()
    # Training score
    if runBDT:
        param_range = param_range_bdt
        param_name = param_name_bdt
    elif runNN:
        param_range = param_range_nn
        param_name = param_name_nn
    axsF.plot(param_range, train_scores_vc_mean, label="Training score", color="darkorange", lw=2)
    axsF.fill_between(epochs, train_scores_vc_mean - train_scores_vc_std, train_scores_vc_mean + train_scores_vc_std, alpha=0.2, color="darkorange", lw=2)
    # Test score
    axsF.plot(param_range, valid_scores_vc_mean, label="Cross-validation score", color="navy", lw=2)
    axsF.fill_between(epochs, valid_scores_vc_mean - valid_scores_vc_std, valid_scores_vc_mean + valid_scores_vc_std, alpha=0.2, color="navy", lw=2)
    axsF.set_xlabel(param_name)
    axsF.set_ylabel('Score')
    axsF.legend(loc="best")
    axsF.set_title('Validation curves')
    pdf_pages.savefig(figF)

if plot_learning_curve:
    # Plot learning curves
    figG, axsG = plt.subplots()
    # 68% CL bands
    #if runBDT:
    #elif runNN:
    axsG.fill_between(train_sizes, train_scores_lc_mean - train_scores_lc_std, train_scores_lc_mean + train_scores_lc_std, alpha=0.2, color="r", lw=2)
    axsG.fill_between(train_sizes, valid_scores_lc_mean - valid_scores_lc_std, valid_scores_lc_mean + valid_scores_lc_std, alpha=0.2, color="g", lw=2)
    # Training and validation scores
    axsG.plot(train_sizes, train_scores_lc_mean, 'o-', label="Training score", color="r", lw=2)
    axsG.plot(train_sizes, valid_scores_lc_mean, 'o-', label="Cross-validation score", color="g", lw=2)
    axsG.set_xlabel("Training examples")
    axsG.set_ylabel('Score')
    axsG.legend(loc="best")
    axsG.set_title('Learning curves')
    pdf_pages.savefig(figG)


pdf_pages.close()

print ""
print "Training sample...."
print "  Signal identified as signal (%)        : ",100.0*np.sum(pred_train[y_train==1]==1.0)/(X_train[y_train==1].shape[0])
print "  Signal identified as background (%)    : ",100.0*np.sum(pred_train[y_train==1]==0.0)/(X_train[y_train==1].shape[0])
print "  Background identified as signal (%)    : ",100.0*np.sum(pred_train[y_train==0]==1.0)/(X_train[y_train==0].shape[0])
print "  Background identified as background (%): ",100.0*np.sum(pred_train[y_train==0]==0.0)/(X_train[y_train==0].shape[0])
print ""
print "Testing sample...."
print "  Signal identified as signal (%)        : ",100.0*np.sum(pred_test[y_test==1]==1.0)/(X_test[y_test==1].shape[0])
print "  Signal identified as background (%)    : ",100.0*np.sum(pred_test[y_test==1]==0.0)/(X_test[y_test==1].shape[0])
print "  Background identified as signal (%)    : ",100.0*np.sum(pred_test[y_test==0]==1.0)/(X_test[y_test==0].shape[0])
print "  Background identified as background (%): ",100.0*np.sum(pred_test[y_test==0]==0.0)/(X_test[y_test==0].shape[0])

print "\nArea under ROC = ",auc

#event_weights_test = np.ones(event_weights_test.shape)
S_test_sum_weights = np.sum( event_weights_test[ np.multiply(pred_test==1, y_test==1) == 1 ] )
print "\nS_test_sum_weights",S_test_sum_weights
B_test_sum_weights = np.sum( event_weights_test[ np.multiply(pred_test==1, y_test==0) == 1 ] )
print "B_test_sum_weights",B_test_sum_weights

S = S_test_sum_weights #/acceptance_sig
B = B_test_sum_weights #/acceptance_bkg

N = S + B
print "\nS",S
print "B",B
print "np.sqrt(B+10)",np.sqrt(B+10)
print "N",N
print "np.sqrt(N)",np.sqrt(N)

print "\nSignificance in SR: S/sqrt(B+10) =",float(S)/np.sqrt(B+10)
print "Significance in SR: S/sqrt(N) =",float(S)/np.sqrt(N)

# Approximate median significance
#def ams(y_true, y_pred, b_r=10):
#    s = np.sum( event_weights_test[ np.multiply( y_pred==1, y_true==1 ) == 1 ] )
#    b = np.sum( event_weights_test[ np.multiply( y_pred==1, y_true==0 ) == 1 ] )
#    ams_score = np.sqrt( 2*( (s + b + b_r)*np.log( 1 + s/(b + b_r) ) - s ) )
#    print "s",s
#    print "b",b
#    print "ams_score",ams_score
#    return ams_score

#def ams_train(y_true, y_pred, b_r=10):
#    s = K.sum( sample_weight[ (y_pred==1)*(y_true==1) == 1 ] )
#    b = K.sum( sample_weight[ (y_pred==1)*(y_true==0) == 1 ] )
#    ams_train_score = K.sqrt( 2*( (s + b + b_r)*K.log( 1 + s/(b + b_r) ) - s ) )
#    return ams_train_score

# Make ams scoring function to evaluate classifier performance
#ams_score = make_scorer(ams, greater_is_better=True, needs_proba=False, needs_threshold=True)
#ams_score = ams(model, y_test, pred_test)

# From yandex Reproducible Experiment Platform (REP)
def ams(s, b, br=10.):
    """
    Regularized approximate median significance

    :param s: amount of signal passed
    :param b: amount of background passed
    :param br: regularization
    """
    radicand = 2 * ((s + b + br) * np.log(1.0 + s / (b + br)) - s)
    return np.sqrt(radicand)

print "\nams(S, B)",ams(S, B)

print "\nPlots saved to",output_filename

t_end = time.time()
print "Process time:",t_end - t_start
