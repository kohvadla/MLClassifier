from __future__ import division

import sys
import time
import argparse
import pickle
import itertools

import ROOT
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
from matplotlib.colors import to_rgba 
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

# Close figure windows, if there are any open
plt.close()

# Start timer
t_start = time.time()

#sum_weights_tot_bkg = 145592.7984953417
#sum_weights_tot_sig = 159.2633907347918

# Default values for command line options
runBDT = False
runXGBoost = False
runNN = False
runTraining = True
use_event_weights = True
use_class_weights = False
log_y = True
doGridSearchCV = False
plot_validation_curve = False
plot_learning_curve = False
do_cut_scan = False
region = 'inclusive'
final_state = '2L'

parser = argparse.ArgumentParser(description='Run classifier -- BDT or neural network')
group = parser.add_mutually_exclusive_group()
group.add_argument('-a', '--adaboost', action='store_true', help='Run adaptive BDT')
group.add_argument('-x', '--xgboost', action='store_true', help='Run gradient BDT')
group.add_argument('-n', '--nn', action='store_true', help='Run neural network')
parser.add_argument('-t', '--no_training', action='store_false', help='Do not train the selected classifier')
parser.add_argument('-e', '--no_event_weights', action='store_false', help='Do not apply event weights to training examples')
parser.add_argument('-c', '--class_weights', action='store_true', help='Apply class weights to account for unbalanced dataset')
parser.add_argument('-y', '--no_log_y', action='store_false', help='Do not use log scale on y-axis')
parser.add_argument('-g', '--doGridSearchCV', action='store_true', help='Perform tuning of hyperparameters using k-fold cross-validation')
parser.add_argument('-v', '--validation_curve', action='store_true', help='Plot validation curve')
parser.add_argument('-l', '--learning_curve', action='store_true', help='Plot learning curve')
parser.add_argument('-s', '--cut_scan', action='store_true', help='Do AMS significance scan on classifier output')
parser.add_argument('-r', '--region', type=str, nargs='?', help='Phase space region: 2L2J or 2L2J-ISR or incl', default='inclusive')
parser.add_argument('-f', '--final_state', type=str, nargs='?', help='Final state: 2L or 3L', default='2L')

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
    runTraining = False
if not(args.no_event_weights):
    use_event_weights = False
if args.class_weights:
    use_class_weights = True
if not(args.no_log_y):
    log_y = False
if args.doGridSearchCV:
    doGridSearchCV = True
if args.validation_curve:
    plot_validation_curve = True
if args.learning_curve:
    plot_learning_curve = True
if args.cut_scan:
    do_cut_scan = True
if args.region == '2L2J':
    region = args.region
elif args.region == '2L2J-ISR':
    region = args.region
if args.final_state == '3L':
    final_state = '3L'

if region == 'inclusive': 
    region_id = '2L'
elif region == '2L2J': 
    region_id = '2L2J'
elif region == '2L2J-ISR':
    region_id = '2L2J_ISR' 

ntuple_id = 'flat_ext'

# Build background arrays
bkgFilename = "../../../ewk/hdf5_files/"+final_state+"_bkg_"+ntuple_id+".h5"
bkgFile = h5py.File(bkgFilename,"r")
X_bkg_dset = bkgFile['FlatTree'][:]  # structured array from the FlatTree dataset

structured_array_features = list(X_bkg_dset.dtype.names)
deselected_features = ['dsid', 'event_weight', 'lep_charge1', 'lep_charge2', 'n_bjets']
if region == '2L2J': 
    deselected_features.extend(['jet_pt3', 'jet_eta3', 'jet_phi3', 'jet_m3'])

#print "\nstructured_array_features", structured_array_features
print "\ndeselected_features", deselected_features

in_selected_features = np.array([i not in deselected_features for i in structured_array_features])
selected_features = ( np.array(structured_array_features)[in_selected_features] ).tolist()
n_selected_features = len(structured_array_features) - len(deselected_features)

print "\nselected_features =", selected_features 
print "\nn_selected_features =", n_selected_features 

X_bkg_sel_arr = np.array( X_bkg_dset[selected_features].tolist() )
X_bkg_ew_arr = np.array( X_bkg_dset['event_weight'].tolist() )

bkgFile.close()

# Build signal arrays
sigFilename = "../../../ewk/hdf5_files/"+final_state+"_sig_"+ntuple_id+".h5"
sigFile = h5py.File(sigFilename,"r")
X_sig_dset = sigFile['FlatTree'][:]

X_sig_sel_arr = np.array( X_sig_dset[selected_features].tolist() )
X_sig_ew_arr = np.array( X_sig_dset['event_weight'].tolist() )

sigFile.close()

print "\nRead in background file:",bkgFilename
print "Read in signal file:",sigFilename

if use_class_weights:
    class_weight = 'balanced'
    n_samples_bkg = None
else:
    class_weight = None
    n_samples_bkg = X_sig_sel_arr.shape[0]

seed = 42

X_sig_sel_shuffled = shuffle(X_sig_sel_arr, random_state=seed, n_samples=None)
X_sig_ew_shuffled = shuffle(X_sig_ew_arr, random_state=seed, n_samples=None)

X_bkg_sel_shuffled = shuffle(X_bkg_sel_arr, random_state=seed, n_samples=n_samples_bkg)
X_bkg_ew_shuffled = shuffle(X_bkg_ew_arr, random_state=seed, n_samples=n_samples_bkg)

X = np.concatenate((X_bkg_sel_shuffled, X_sig_sel_shuffled), 0)
event_weights = np.concatenate((X_bkg_ew_shuffled, X_sig_ew_shuffled), 0)

# Make array of labels
y_bkg = np.zeros(X_bkg_sel_shuffled.shape[0])
y_sig = np.ones(X_sig_sel_shuffled.shape[0])
y = np.concatenate((y_bkg, y_sig),0)
y = np.ravel(y)

classes = np.unique(y)
class_weight_vect = compute_class_weight(class_weight, classes, y)
class_weight_dict = {0: class_weight_vect[0], 1: class_weight_vect[1]}

if use_class_weights:
    scale_pos_weight = len(y)/np.sum(y)
else:
    scale_pos_weight = 1.

print "\nclass_weight_vect",class_weight_vect
print "class_weight_dict",class_weight_dict
print "\nscale_pos_weight",scale_pos_weight

# Replane -999
imp = Imputer(missing_values=-999, strategy='mean', axis=0)
X = imp.fit_transform(X)

print "\nbefore imp: np.any(X_bkg_sel_arr == -999)",np.any(X_bkg_sel_arr == -999)
print "before imp: np.any(X_bkg_shuffled == -999)",np.any(X_bkg_sel_shuffled == -999)
print "after imp: np.any(X == -999)",np.any(X == -999)

# Split dataset in train and test sets
test_size=0.33

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y) #, shuffle=True
y_test = y_test.astype(int) # convert labels from float to int

event_weights_train, event_weights_test, y_ew_train, y_ew_test = train_test_split(event_weights, y, test_size=test_size, 
                                                                                    random_state=seed, stratify=y) #, shuffle=True
y_ew_test = y_ew_test.astype(int) # convert labels from float to int

print ""
print "# training examples:"
print "Background  :", y_train[y_train==0].shape[0]  
print "Signal      :", y_train[y_train==1].shape[0]  
print ""
print "# test examples:"
print "Background  :", y_test[y_test==0].shape[0]  
print "Signal      :", y_test[y_test==1].shape[0]  


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

# List of parameter values to grid search
max_depth = [1, 2, 3, 4, 5]
n_estimators = [50, 100, 200, 500, 1000]
learning_rate = [0.001, 0.01, 0.1, 0.5, 1.0]

# Specify one of the above parameter lists to plot validation curve for
param_name_bdt = "max_depth"  # Name of parameter in the API
param_range_bdt = max_depth   # Name of the variable holding the list of parameters
#param_name_bdt = "n_estimators"  # Name of parameter in the API
#param_range_bdt = n_estimators   # Name of the variable holding the list of parameters
#param_name_bdt = "learning_rate"  # Name of parameter in the API
#param_range_bdt = learning_rate   # Name of the variable holding the list of parameters

global model, trained_model_filename

if runBDT:

    if runTraining:

        if runXGBoost:  # best parameter values: max_depth = 5, learning_rate=0.5, n_estimators=200
            model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective="binary:logistic", scale_pos_weight=scale_pos_weight)

            if doGridSearchCV:
                tuned_parameters = [{'max_depth': max_depth, 'n_estimators': n_estimators, 'learning_rate': learning_rate}]
                model = GridSearchCV( model, tuned_parameters, cv=3, scoring=None, fit_params={'sample_weight': sample_weight} )

        else: # run AdaBoost
            model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, class_weight=class_weight_dict), n_estimators=100, learning_rate=1.0)

            if doGridSearchCV:
                tuned_parameters = [{'base_estimator': [DecisionTreeClassifier(max_depth=1, class_weight=class_weight_dict)], 
                                    'n_estimators': n_estimators, 'learning_rate': learning_rate}]
                model = GridSearchCV( model, tuned_parameters, cv=3, scoring=None, fit_params={'sample_weight': sample_weight} )

        params = model.get_params()
        print "\nmodel.get_params()",params

        if doGridSearchCV:
            model.fit( X_train, y_train )
        else:
            model.fit( X_train, y_train, sample_weight=sample_weight )
            
        print "\nBuilding and training BDT\n"


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

        if runXGBoost: 
            trained_model_filename = 'xgboost_'+final_state+'_AC18.pkl'
        else: 
            trained_model_filename = 'adaboost_'+final_state+'_AC18.pkl'
        joblib.dump(model, trained_model_filename)

    if not runTraining:
        if runXGBoost:
            model = joblib.load('xgboost_'+final_state+'_AC18.pkl')
            print "\nReading in pre-trained BDT:",'xgboost_'+final_state+'_AC18.pkl\n'
        else:
            model = joblib.load('adaboost_'+final_state+'_AC18.pkl')
            print "\nReading in pre-trained BDT:",'adaboost_'+final_state+'_AC18.pkl\n'

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
        print "\nBuilding and training neural network\n"

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
        print "\nReading in pre-trained neural network\n"
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
    output_filename = 'xgboost'
elif runBDT: 
    output_filename = 'adaboost'
elif runNN: 
    output_filename = 'nn'
output_filename += '_'+final_state+'_'+ntuple_id
if use_event_weights:
    output_filename += '_ew'
if use_class_weights:
    output_filename += '_cw'
if plot_validation_curve:
    output_filename += '_vc'
if plot_learning_curve:
    output_filename += '_lc'

plot_filename = 'plots/'+output_filename+'.pdf'
pdf_pages = PdfPages(plot_filename)
np.set_printoptions(threshold=np.nan)


# Approximate median significance
# ---- From yandex Reproducible Experiment Platform (REP)
def ams(s, b, br=10.):
    """
    Regularized approximate median significance

    :param s: amount of signal passed
    :param b: amount of background passed
    :param br: regularization
    """
    radicand = 2 * ((s + b + br) * np.log(1.0 + s / (b + br)) - s)
    return np.sqrt(radicand)


# Yields

clf_cut = 0.
if not do_cut_scan:
    if final_state == '2L':
        if runXGBoost: clf_cut = 0.9   # optimized cut on xgboost output
        elif runBDT: clf_cut = 0.0  # optimized cut on adaboost output
    elif final_state == '3L':
        if runXGBoost: clf_cut = 0.9   # optimized cut on xgboost output
        elif runBDT: clf_cut = 0.0  # optimized cut on adaboost output
cut_optimized = clf_cut
ams_optimized_br0 = 0.
ams_optimized_br10 = 0.
S_optimized = 0.
B_optimized = 0.
acceptance_bkg = 1.
acceptance_sig = 1.

print "X_bkg_dset.shape[0]",X_bkg_dset.shape[0]
print "X_bkg_sel_shuffled.shape[0]",X_bkg_sel_shuffled.shape[0]

n_train_test_bkg = X_bkg_sel_shuffled.shape[0]
n_tot_bkg = X_bkg_dset.shape[0]
n_sel_bkg_frac = n_train_test_bkg / n_tot_bkg

acceptance_sig = 1./3  # fraction of selected dataset used for testing
acceptance_bkg = 1./3  # fraction of selected dataset used for testing
if not use_class_weights:
    acceptance_bkg *= n_sel_bkg_frac  # fraction of full dataset selected for training and testing

if do_cut_scan:

    if runBDT and not runXGBoost:
        cut_range = [0.0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    else:
        cut_range = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]

    for cut in cut_range:
        sum_weights_test_sig = np.sum( event_weights_test[ np.multiply(output_test>cut, y_test==1) == 1 ] )
        sum_weights_test_bkg = np.sum( event_weights_test[ np.multiply(output_test>cut, y_test==0) == 1 ] )

        print "\n--------------------------------------"

        if sum_weights_test_sig < 0.:
            sum_weights_test_sig = 0.
        if sum_weights_test_bkg < 1e-10:
            continue

        print "sum_weights_test_sig",sum_weights_test_sig
        print "sum_weights_test_bkg",sum_weights_test_bkg

        print "acceptance_sig",acceptance_sig
        print "acceptance_bkg",acceptance_bkg

        S = sum_weights_test_sig/acceptance_sig
        B = sum_weights_test_bkg/acceptance_bkg

        if cut == cut_range[0]:
            S_optimized = S
            B_optimized = B

        print "S =",S,", B =",B
        print "sum_weights_test_sig/acceptance_sig =",sum_weights_test_sig/acceptance_sig

        ams_scan_br0 = ams(S, B, 0)
        ams_scan_br10 = ams(S, B, 10)

        print "cut =",cut,", ams_br10 = ",ams_scan_br10

        if ams_scan_br10 > ams_optimized_br10 and B >= 1.: #sum_weights_test_bkg >= 1.:
            ams_optimized_br0 = ams_scan_br0
            ams_optimized_br10 = ams_scan_br10
            cut_optimized = cut
            S_optimized = S
            B_optimized = B
            print "new optimal cut:",cut,", with ams_br10:",ams_scan_br10,", S =",S,", B =",B
else:
    sum_weights_test_sig = np.sum( event_weights_test[ np.multiply(output_test>cut_optimized, y_test==1) == 1 ] )
    sum_weights_test_bkg = np.sum( event_weights_test[ np.multiply(output_test>cut_optimized, y_test==0) == 1 ] )
    
    S_optimized = sum_weights_test_sig / acceptance_sig
    B_optimized = sum_weights_test_bkg / acceptance_bkg

    ams_optimized_br0 = ams(S_optimized, B_optimized, 0)
    ams_optimized_br10 = ams(S_optimized, B_optimized, 10)



# Plotting - probabilities
figA, axsA = plt.subplots()
axsA.set_ylabel("Events normalized to one")
if runXGBoost: 
    bins = np.linspace(0.0, 1.0, 30)
    axsA.set_title("XGBoost BDT")
    axsA.set_xlabel("Signal probability")
elif runBDT: 
    bins = np.linspace(-1.0, 1.0, 40)
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
if log_y: axsA.set_yscale('log', nonposy='clip')
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
    bins = np.linspace(-1.0, 1.0, 40)
    axsA2.set_title("AdaBoost BDT")
    axsA2.set_xlabel("BDT score")
elif runNN: 
    bins = np.linspace(0.0, 1.0, 30)
    axsA2.set_title("Neural network")
    axsA2.set_xlabel("Signal probability")
# Plot training output
sf_bkg = 1.#/acceptance_bkg
sf_sig = 1.
axsA2.hist(output_train[y_train==0], bins, weights=event_weights_train[y_train==0]*sf_bkg, alpha=0.2, histtype='stepfilled', facecolor='blue', label='Background trained', normed=True)
axsA2.hist(output_train[y_train==1], bins, weights=event_weights_train[y_train==1]*sf_sig, alpha=0.2, histtype='stepfilled', facecolor='red', label='Signal trained', normed=True)
# Plot test output
axsA2.hist(output_test[y_test==0], bins, weights=event_weights_test[y_test==0]*sf_bkg, alpha=1, histtype='step', linestyle='--', edgecolor='blue', label='Background tested', normed=True)
axsA2.hist(output_test[y_test==1], bins, weights=event_weights_test[y_test==1]*sf_sig, alpha=1, histtype='step', linestyle='--', edgecolor='red', label='Signal tested', normed=True)
if log_y: axsA2.set_yscale('log', nonposy='clip')
axsA2.legend(loc="best")
pdf_pages.savefig(figA2)

# Plotting weighted events scaled to 36.1/fb
figA3, axsA3 = plt.subplots()
axsA3.set_ylabel("Events scaled to 36.1/fb")
if runXGBoost: 
    bins = np.linspace(0.0, 1.0, 30)
    axsA3.set_title("XGBoost BDT")
    axsA3.set_xlabel("Signal probability")
elif runBDT: 
    bins = np.linspace(-1.0, 1.0, 40)
    axsA3.set_title("AdaBoost BDT")
    axsA3.set_xlabel("BDT score")
elif runNN: 
    bins = np.linspace(0.0, 1.0, 30)
    axsA3.set_title("Neural network")
    axsA3.set_xlabel("Signal probability")
# Plot training output
sf_bkg = 1./acceptance_bkg
sf_sig = 1./acceptance_sig
axsA3.hist(output_train[y_train==0], bins, weights=event_weights_train[y_train==0]*sf_bkg, alpha=0.2, histtype='stepfilled', facecolor='blue', label='Background trained')
axsA3.hist(output_train[y_train==1], bins, weights=event_weights_train[y_train==1]*sf_sig, alpha=0.2, histtype='stepfilled', facecolor='red', label='Signal trained')
# Plot test output
axsA3.hist(output_test[y_test==0], bins, weights=sf_bkg*(event_weights_test[y_test==0]), alpha=1, histtype='step', linestyle='--', edgecolor='blue', label='Background tested')
axsA3.hist(output_test[y_test==1], bins, weights=sf_sig*(event_weights_test[y_test==1]), alpha=1, histtype='step', linestyle='--', edgecolor='red', label='Signal tested')
if log_y: axsA3.set_yscale('log', nonposy='clip')
axsA3.legend(loc="best")
pdf_pages.savefig(figA3)

## Plotting probabilities with event weights
#figA4, axsA4 = plt.subplots()
#axsA4.set_ylabel("Events scaled to 36.1/fb")
#if runXGBoost: 
#    bins = np.linspace(0.0, 1.0, 30)
#    axsA4.set_title("XGBoost BDT")
#    axsA4.set_xlabel("Signal probability")
#elif runBDT: 
#    bins = np.linspace(-1.0, 1.0, 40)
#    axsA4.set_title("AdaBoost BDT")
#    axsA4.set_xlabel("BDT score")
#elif runNN: 
#    bins = np.linspace(0.0, 1.0, 30)
#    axsA4.set_title("Neural network")
#    axsA4.set_xlabel("Signal probability")
## Plot training output
#sf_bkg = 1./acceptance_bkg
#sf_sig = 1./acceptance_sig
#print "output_train[y_train==0].shape",output_train[y_train==0].shape
#print "event_weights_train[y_train==0].shape",event_weights_train[y_train==0].shape
#
#axsA4.hist( [output_train[y_train==0], output_train[y_train==1] ], 
#		bins, 
#		weights=[ sf_bkg*event_weights_train[y_train==0], sf_bkg*event_weights_train[y_train==1] ], 
#		alpha=0.2, 
#		histtype='bar', 
#		stacked=False, 
#		color=[to_rgba('blue'), to_rgba('red')], 
#		label=['Background trained', 'Signal trained'])
#
## Plot test output
#axsA4.hist( [output_test[y_test==0], output_test[y_test==1]], 
#		bins, 
#		weights=[ sf_bkg*event_weights_test[y_test==0], sf_bkg*event_weights_test[y_test==1] ], 
#		alpha=1, 
#		histtype='step', 
#		stacked=False, 
#		linestyle='--',
#		color=[to_rgba('blue'), to_rgba('red')], 
#		label=['Background tested', 'Signal tested'])
#
#if log_y: axsA4.set_yscale('log') #, nonposy='clip')
#axsA4.legend(loc="best")
#pdf_pages.savefig(figA4)


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
    #axC1.set_ylim([0,n_selected_features])
    axC1.set_yticks(y_pos)
    axC1.set_yticklabels(np.array(structured_array_features)[in_selected_features],fontsize=5)
    axC1.set_xlabel('Relative importance, %')
    axC1.set_title("Estimated variable importance using outputs (BDT)")
    plt.gca().invert_yaxis()
    pdf_pages.savefig(figC)

# NEURAL NETWORK
# Assess variable importance using weights method
if runNN and not (doGridSearchCV or plot_validation_curve):
    weights = np.array([])
    layer_counter = 0
    for layer in model.layers:
        layer_counter += 1
        print "layer.name",layer.name
        #if layer.name =="dense_1":
        if layer_counter == 1:
            weights = layer.get_weights()[0]
    # Ecol. Modelling 160 (2003) 249-264
    sumWeights = np.sum(np.absolute(weights),axis=0)
    Q = np.absolute(weights)/sumWeights
    R = 100.0 * np.sum(Q,axis=1) / np.sum(np.sum(Q,axis=0))
    y_pos = np.arange(X_train.shape[1])
    figC, axC = plt.subplots()
    axC.barh(y_pos, R, align='center', alpha=0.4)
    #axC.set_ylim([0,n_selected_features])
    axC.set_yticks(y_pos)
    axC.set_yticklabels(np.array(structured_array_features)[in_selected_features],fontsize=5)
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
    axsF.plot( param_range, train_scores_vc_mean, 'o-', label="Training score", color="darkorange", lw=2)
    axsF.fill_between( param_range, train_scores_vc_mean - train_scores_vc_std, train_scores_vc_mean + train_scores_vc_std, alpha=0.2, color="darkorange", lw=2)
    # Test score
    axsF.plot( param_range, valid_scores_vc_mean, 'o-', label="Cross-validation score", color="navy", lw=2)
    axsF.fill_between( param_range, valid_scores_vc_mean - valid_scores_vc_std, valid_scores_vc_mean + valid_scores_vc_std, alpha=0.2, color="navy", lw=2)
    axsF.set_xlabel(param_name)
    axsF.set_ylabel('Score')
    axsF.legend(loc="best")
    axsF.set_title('Validation curves')
    axsF.set_ylim(0., 1.)
    pdf_pages.savefig(figF)

if plot_learning_curve:
    # Plot learning curves
    figG, axsG = plt.subplots()
    # 68% CL bands
    #if runBDT:
    #elif runNN:
    axsG.fill_between( train_sizes, train_scores_lc_mean - train_scores_lc_std, train_scores_lc_mean + train_scores_lc_std, alpha=0.2, color="r", lw=2)
    axsG.fill_between( train_sizes, valid_scores_lc_mean - valid_scores_lc_std, valid_scores_lc_mean + valid_scores_lc_std, alpha=0.2, color="g", lw=2)
    # Training and validation scores
    axsG.plot( train_sizes, train_scores_lc_mean, 'o-', label="Training score", color="r", lw=2)
    axsG.plot( train_sizes, valid_scores_lc_mean, 'o-', label="Cross-validation score", color="g", lw=2)
    axsG.set_xlabel("Training examples")
    axsG.set_ylabel('Score')
    axsG.legend(loc="best")
    axsG.set_title('Learning curves')
    pdf_pages.savefig(figG)


pdf_pages.close()

plt.show()

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

print "\nsum_weights_test_sig",sum_weights_test_sig
print "sum_weights_test_bkg",sum_weights_test_bkg

print "\nacceptance_sig",acceptance_sig
print "acceptance_bkg",acceptance_bkg

N = S_optimized + B_optimized
#print "\nS",S_optimized
#print "B",B_optimized
##print "np.sqrt(B+10)",np.sqrt(B+10)
#print "N",N
#print "np.sqrt(N)",np.sqrt(N)

B_optimized_sumw2 = np.sum( (event_weights_test[y_test==0])**2 )#/acceptance_bkg)**2 )
S_optimized_sumw2 = np.sum( (event_weights_test[y_test==1])**2 )#/acceptance_sig)**2 )

B_optimized_sumwOverAccept2 = np.sum( (event_weights_test[y_test==0]/acceptance_bkg)**2 )
S_optimized_sumwOverAccept2 = np.sum( (event_weights_test[y_test==1]/acceptance_sig)**2 )

#print "\nevent_weights_test[event_weights_test>1.]",event_weights_test[event_weights_test>1.]
#print "event_weights_test[y_test==1]",event_weights_test[y_test==1]

# Save SR yields in one-bin histogram
ROOT_filename = "output_"+output_filename+".root"
f = ROOT.TFile(ROOT_filename,"recreate")
ROOT.TH1.SetDefaultSumw2()
h = ROOT.TH1D("SR","SR yields;;Events",2,0,4)

# Background yield +/- sum of event weights squared
h.SetBinContent(1, B_optimized)
h.SetBinError(1, ROOT.TMath.Sqrt(B_optimized_sumw2))
h.GetXaxis().SetBinLabel(1, "Background +/- sqrt(sumw2)")

# Signal yield +/- sum of event weights squared
h.SetBinContent(2, S_optimized)
h.SetBinError(2, ROOT.TMath.Sqrt(S_optimized_sumw2))
h.GetXaxis().SetBinLabel(2, "Signal +/- sqrt(sumw2)")

# Background yield +/- sum of event weights over acceptance squared
h.SetBinContent(3, B_optimized)
h.SetBinError(3, ROOT.TMath.Sqrt(B_optimized_sumwOverAccept2))
h.GetXaxis().SetBinLabel(3, "Background +/- sqrt(sumOverAccept2)")

# Signal yield +/- sum of event weights over acceptance squared
h.SetBinContent(4, S_optimized)
h.SetBinError(4, ROOT.TMath.Sqrt(S_optimized_sumwOverAccept2))
h.GetXaxis().SetBinLabel(4, "Signal +/- sqrt(sumOverAccept2)")

#c = ROOT.TCanvas("c_SR","SR canvas",800,600)
h.Draw()
#c.Draw()
f.Write()
f.Close()
#c.Close()

#print "\nSignificance in SR: S/sqrt(B+10) =",float(S)/np.sqrt(B+10)

print "\n*************************************************"
print "Optimized cut value = ",cut_optimized
print "*************************************************"
print "Event yields in optimized SR:"
print "S =",S_optimized,"+/-",S_optimized_sumw2
print "B =",B_optimized,"+/-",B_optimized_sumw2
print "*************************************************"
print "Optimized AMS score (br=0) = ",ams_optimized_br0
print "Optimized AMS score (br=10) = ",ams_optimized_br10
print "S/sqrt(N) =",float(S_optimized)/np.sqrt(N)
print "*************************************************"

if runTraining: print "\nTrained classifier model saved to:",trained_model_filename
print "\nPlots saved to:",plot_filename
print "\nROOT histogram saved to:",ROOT_filename

t_end = time.time()
print "\nProcess time:",t_end - t_start
