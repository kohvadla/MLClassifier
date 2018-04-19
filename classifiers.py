from __future__ import division

import sys
import argparse
import pickle
import itertools

import h5py

import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import preprocessing, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from featuresLists import features 

import logging

logging.basicConfig(
                    format="%(message)s",
                    level=logging.DEBUG,
                    stream=sys.stdout)

###############################################
# MAIN PROGRAM

runBDT = False
runNN = False
runTraining = True
use_event_weights = True
log_y = True
doGridSearchCV = False

parser = argparse.ArgumentParser(description='Run classifier -- BDT or neural network')
group = parser.add_mutually_exclusive_group()
group.add_argument('-b', '--bdt', action='store_true', help='Run with BDT')
group.add_argument('-n', '--nn', action='store_true', help='Run with neural network')
parser.add_argument('-t', '--no_training', action='store_false', help='Do not train the selected classifier')
parser.add_argument('-w', '--no_event_weights', action='store_false', help='Do not apply event weights to training examples')
parser.add_argument('-l', '--no_log_y', action='store_false', help='Do not use log scale on y-axis')
parser.add_argument('-g', '--doGridSearchCV', action='store_true', help='Perform tuning of hyperparameters using k-fold cross-validation')

args = parser.parse_args()

if args.bdt:
    runBDT = args.bdt
elif args.nn:
    runNN = args.nn
elif args.bdt and args.nn:
    runBDT = args.bdt
    runNN = False
    print "\nINVALID CLASSIFIER CHOICE! Both bdt and nn are chosen. Reverting to run only bdt."
else:
    sys.exit("Classifier argument not given! Choose either -b for BDT or -n for neural network.")
if not(args.no_training):
    runTraining = args.no_training
if not(args.no_event_weights):
    use_event_weights = args.no_event_weights
if not(args.no_log_y):
    log_y = args.no_log_y
if args.doGridSearchCV:
    doGridSearchCV = args.doGridSearchCV

if use_event_weights:
    print "\nINFO Applying event weights to the examples during training"
else:
    print "\nINFO Not applying event weights to the examples during training"

if doGridSearchCV:
    print "INFO Performing grid search for hyperparameters"

#selectedFeatures = features
unselectedFeatures = ['dsid']
event_weight_feature = ['event_weight']

in_selectedFeatures = np.array([i not in unselectedFeatures for i in features])
selectedFeatures = (np.array(features)[in_selectedFeatures]).tolist()
print "\nselectedFeatures =", selectedFeatures 
n_selectedFeatures = len(features) - len(unselectedFeatures)
#print "n_selectedFeatures =", n_selectedFeatures 


# Build background arrays
bkgFile = h5py.File("../ewk/hdf5_files/2L_bkg_flat.h5","r")
X_bkg = bkgFile['FlatTree'][:,in_selectedFeatures]
bkgFile.close()


# Build signal arrays
sigFile = h5py.File("../ewk/hdf5_files/2L_sig_flat.h5","r")
X_sig = sigFile['FlatTree'][:,in_selectedFeatures]
sigFile.close()

# Draw a number of signal and background samples, equal to the size of the signal dataset, randomly from the datasets
X_sig_shuffled = shuffle(X_sig, random_state=None, n_samples=None)
X_bkg_shuffled = shuffle(X_bkg, random_state=None, n_samples=X_sig_shuffled.shape[0])

X = np.concatenate((X_bkg_shuffled, X_sig_shuffled), 0)

# Make array of labels
y_bkg = np.zeros(X_bkg_shuffled.shape[0])
y_sig = np.ones(X_sig_shuffled.shape[0])
y = np.concatenate((y_bkg, y_sig),0)
y = np.ravel(y)

# Split dataset in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True, stratify=None)
y_test = y_test.astype(int) # convert labels from float to int

# Remove event weights from training features and store in separate array
event_weights = X_train[:,-1] #X[:,is_event_weight]
X_train = X_train[:,:-1] #np.delete(X, is_event_weight, 1)
X_test = X_test[:,:-1] #np.delete(X, is_event_weight, 1)

print ""
print "Number of background examples for training =", y_train[y_train==0].shape[0]  
print "Number of background examples for testing =", y_test[y_test==0].shape[0]  
print "Number of signal examples for training =", y_train[y_train==1].shape[0]  
print "Number of signal examples for testing =", y_test[y_test==1].shape[0]  
print ""


# Feature scaling
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)


#for score in scores:  
# BDT TRAINING AND TESTING
if runBDT:
    if runTraining:
        print "Building and training BDT"

        if doGridSearchCV:
            tuned_parameters = [{'base_estimator': [DecisionTreeClassifier(max_depth=1)], 'n_estimators': [10, 100, 1000], 
                                    'learning_rate': [0.1, 1.0, 10.0]}]
            clf = GridSearchCV( AdaBoostClassifier(), tuned_parameters, cv=3, scoring=None )
        else:
            clf = AdaBoostClassifier( base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, 
                                      learning_rate=1.0, algorithm='SAMME.R', random_state=None )

        params = clf.get_params()
        print "params",params

        if use_event_weights:
            clf.fit(X_train,y_train,sample_weight=event_weights)
        else:
            clf.fit(X_train,y_train)
        joblib.dump(clf, 'bdt_AC18.pkl')

    if not runTraining:
        print "Reading in pre-trained BDT"
        clf = joblib.load('bdt_AC18.pkl')

    # Training scores
    pred_train = clf.predict(X_train)
    output_train = clf.decision_function(X_train)

    # Validation scores
    pred_test = clf.predict(X_test)
    output_test = clf.decision_function(X_test)


# NEURAL NETWORK TRAINING AND TESTING
# Set up neural network
if runNN:
    if runTraining:
        print "Building and training neural network"

        if doGridSearchCV:
            def create_model():
                model = Sequential()
                model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation="relu"))
                model.add(Dense(1, activation="sigmoid"))
                model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
                return model

            model = KerasClassifier(build_fn=create_model, verbose=0)

            batch_size = [10, 20, 40, 60, 80, 100]
            epochs = [10, 50, 100]
            param_grid = dict(batch_size=batch_size, epochs=epochs)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

            if use_event_weights:
                grid_result = grid.fit(X_train, y_train, epochs=100, batch_size=100, sample_weight=event_weights) #, validation_split=0.33)
            else:
                grid_result = grid.fit(X_train,y_train, epochs=100, batch_size=100)

        elif not doGridSearchCV:
            model = Sequential()
            model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation="relu"))
            model.add(Dense(1, activation="sigmoid"))
            model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

            if use_event_weights:
                model.fit(X_train, y_train, epochs=100, batch_size=100, sample_weight=event_weights) #, validation_split=0.33)
            else:
                model.fit(X_train,y_train, epochs=100, batch_size=100)

            model.save("nn_AC18.h5")

    elif not runTraining:
        print "Reading in pre-trained neural network"
        model = load_model("nn_AC18.h5")

    # Get class and probability predictions
    if doGridSearchCV:
        # Training
        pred_train = grid.predict(X_train)
        output_train = grid.predict_proba(X_train)
        for i_row_train in range(output_train.shape[0]):
            if y_train[i_row_train] == 0:
                output_train[i_row_train] = output_train[i_row_train,0]
            elif y_train[i_row_train] == 1:
                output_train[i_row_train] = output_train[i_row_train,1]
        output_train = output_train[:,0]

        # Testing
        pred_test = grid.predict(X_test)
        output_test = grid.predict_proba(X_test) # array of shape (n,2)
        for i_row_test in range(output_test.shape[0]):
            if y_test[i_row_test] == 0:
                output_test[i_row_test] = output_test[i_row_test,0]
            elif y_test[i_row_test] == 1:
                output_test[i_row_test] = output_test[i_row_test,1]
        output_test = output_test[:,0]
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
    if runNN:
        clf = grid
    print "Best parameters set found on development set:"
    print ""
    print "clf.best_params_", clf.best_params_
    print ""
    print "Grid scores on development set:"
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print "{0:0.3f} (+/-{1:0.03f}) for {2!r}".format(mean, std, params)
    print ""

    print "Detailed classification report:"
    print ""
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print ""
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, y_pred)
    print ""

#if not doGridSearchCV:
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

# Plotting - probabilities
#print probabilities_train[(y_train==0.0).reshape(2*nEvents,)]

if runBDT or runNN:
    if use_event_weights:
        output_string = 'AC18_shuffled_train_test_split_ew'
    else:
        output_string = 'AC18_shuffled_train_test_split'
    if runBDT: 
        output_filename = 'plots/bdt_'+output_string+'.pdf'
    elif runNN: 
        output_filename = 'plots/nn_'+output_string+'.pdf'
    pdf_pages = PdfPages(output_filename)
np.set_printoptions(threshold=np.nan)

figA, axsA = plt.subplots()
axsA.set_ylabel("Events")
if runBDT: 
    axsA.set_xlabel("BDT response")
    bins = np.linspace(-1.0, 1.0, 50)
elif runNN: 
    axsA.set_xlabel("NN signal probability")
    bins = np.linspace(0.0, 1.0, 25)
# Plot training output
#if not doGridSearchCV:
axsA.hist(output_train[y_train==0], bins, alpha=0.3, histtype='stepfilled', facecolor='blue', label='Background trained', normed=True)
axsA.hist(output_train[y_train==1], bins, alpha=0.3, histtype='stepfilled', facecolor='red', label='Signal trained', normed=True)
# Plot test output
axsA.hist(output_test[y_test==0], bins, alpha=1, histtype='step', linestyle='--', edgecolor='blue', label='Background tested', normed=True)
axsA.hist(output_test[y_test==1], bins, alpha=1, histtype='step', linestyle='--', edgecolor='red', label='Signal tested', normed=True)
if log_y: axsA.set_yscale('log') #, nonposy='clip')
axsA.legend()
pdf_pages.savefig(figA)


# Plotting - performance curves
# ROC
fpr, tpr, thresholds = roc_curve(y_test, output_test, pos_label=1)
auc = roc_auc_score(y_test, output_test)
print "\nArea under ROC = ",auc
figB, axB1 = plt.subplots()
#axB1,axB2 = axsB.ravel()
axB1.plot(fpr, tpr, label='ROC curve')
axB1.plot([0, 1], [0, 1], 'k--')
axB1.set_xlim([0.0, 1.0])
axB1.set_ylim([0.0, 1.05])
axB1.set_xlabel('False Signal Rate')
axB1.set_ylabel('True Signal Rate')
if runBDT: axB1.set_title('BDT')
elif runNN: axB1.set_title('Neural network')
axB1.text(0.4,0.2,"AUC = %.4f" % auc,fontsize=15)
pdf_pages.savefig(figB)

# BDT
# Variable importances
if runBDT and not doGridSearchCV:
    y_pos = np.arange(X_train.shape[1])
    figC, axC1 = plt.subplots(1,1)
    axC1.barh(y_pos, 100.0*clf.feature_importances_, align='center', alpha=0.4)
    #axC1.set_ylim([0,n_selectedFeatures])
    axC1.set_yticks(y_pos)
    axC1.set_yticklabels(np.array(features)[in_selectedFeatures],fontsize=5)
    axC1.set_xlabel('Relative importance, %')
    axC1.set_title("Estimated variable importance using outputs (BDT)")
    plt.gca().invert_yaxis()
    pdf_pages.savefig(figC)

# NEURAL NETWORK
# Assess variable importance using weights method
if runNN and not doGridSearchCV:
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
    axC.set_yticklabels(np.array(features)[in_selectedFeatures],fontsize=5)
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

pdf_pages.close()

print "\nPlots saved to",output_filename
