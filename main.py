#!/usr/bin/env python3

from __future__ import division

import os
import sys
import time
import argparse
import pickle
#import h5py
import itertools

from ROOT import TLorentzVector, RooStats
import uproot
import numpy as np
import pandas as pd
#from pandas import HDFStore
import seaborn as sns

import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import preprocessing, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report, make_scorer, log_loss#, balanced_accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.utils import shuffle, resample
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba 
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

import logging

logging.basicConfig(
                    format="%(message)s",
                    level=logging.DEBUG,
                    stream=sys.stdout)


from modules.importData import *
from modules.plotting import *
from modules.samples import l_bkg, d_sig
from modules.models import create_model
#from hepFunctions import invariantMass


h5_group = ""

def main():

  # Start timer
  t_start = time.time()

  # Command line options
  parser = argparse.ArgumentParser()
  group_model = parser.add_mutually_exclusive_group() 
  group_model.add_argument('-x', '--xgboost', action='store_true', help='Run gradient BDT')
  group_model.add_argument('-n', '--nn', action='store_true', help='Run neural network')
  group_model.add_argument('-p', '--prepare_hdf5', type=str, nargs='?', default='', help='Prepare input datasets for ML and store in HDF5 file; options: "2L2J" or "2L3J+"')
  group_read_dataset = parser.add_mutually_exclusive_group() 
  group_read_dataset.add_argument('-r', '--read_hdf5', action='store_true', help='Read prepared datasets from HDF5 file')
  #group_read_dataset.add_argument('-d', '--direct_read', action='store_true', help='Read unprepared datasets from ROOT file')
  parser.add_argument('-l', '--load_pretrained_model', action='store_true', help='Load pre-trained classifier model, i.e. only run on test data')
  #parser.add_argument('-B', '--N_sig_events', type=lambda x: int(float(x)), default=0, help='Number of signal events to read from the dataset')
  #parser.add_argument('-S', '--N_bkg_events', type=lambda x: int(float(x)), default=0, help='Number of background events to read from the dataset for each class')
  parser.add_argument('-s', '--signal_region', type=str, nargs='?', default='int', help='Choose signal region: low-2J, int-2J, high-2J, low-3J+, int-3J+, high-3J+')
  parser.add_argument('-b', '--balanced', type=int, nargs='?', default=-1, help='Balance dataset for training; 0: oversample signal, 1: undersample background')
  parser.add_argument('-m', '--multiclass', action='store_true', help='Use multiple background classes in addition to the signal class')
  parser.add_argument('-w', '--event_weight', action='store_true', help='Apply event weights during training')
  parser.add_argument('-c', '--class_weight', action='store_true', help='Apply class weights to account for unbalanced dataset')
  parser.add_argument('-t', '--do_train', action='store_true', help='Train the classifier')
  parser.add_argument('-T', '--do_test', action='store_true', help='Test the classifier on data it has not been trained on')
  parser.add_argument('-e', '--train_even', action='store_true', help='Use even run numbers for training and odd run numbers for testing')
  parser.add_argument('-o', '--train_odd', action='store_true', help='Use odd run numbers for training and even run numbers for testing')
  parser.add_argument('-C', '--doCV', action='store_true', help='Perform a k-fold cross-validation on the training set during training')
  parser.add_argument('-O', '--oversample', action='store_true', help='Balance imbalanced dataset using oversampling')
  parser.add_argument('-U', '--undersample', action='store_true', help='Balance imbalanced dataset using undersampling')
  parser.add_argument('--n_nodes', type=int, nargs='?', default=20, help='Number of nodes in each hidden neural network layer')
  parser.add_argument('--n_hidden_layers', type=int, nargs='?', default=1, help='Number of nodes in each hidden neural network layer')
  parser.add_argument('--dropout', type=float, nargs='?', default=0., help='Use dropout regularization on neural network layers to reduce overfitting')
  parser.add_argument('--L1', type=float, nargs='?', default=0., help='Use L1 regularization on neural network weights to reduce overfitting')
  parser.add_argument('--L2', type=float, nargs='?', default=0., help='Use L2 regularization (weights decay) on neural network weights to reduce overfitting')
  parser.add_argument('--lr', type=float, nargs='?', default=0.001, help='Set learning rate for the neural network or BDT optimizer')
  parser.add_argument('--batch_size', type=int, nargs='?', default=32, help='Number of events to use for each weight update')
  parser.add_argument('--epochs', type=lambda x: int(float(x)), nargs='?', default=1, help='Number of passes through the training set')
  parser.add_argument('--max_depth', type=int, nargs='?', default=3, help='Maximum tree depth for BDT')
  parser.add_argument('--n_estimators', type=lambda x: int(float(x)), nargs='?', default=100, help='Number of trees in BDT ensemble')
  parser.add_argument('--gamma', type=float, nargs='?', default=0, help='Minimum loss reduction required to make a further partition on a leaf node of the XGBoost tree')
  parser.add_argument('--min_child_weight', type=float, nargs='?', default=1, help='Minimum sum of instance weight(hessian) needed in a child')
  parser.add_argument('--max_delta_step', type=float, nargs='?', default=0, help='Maximum delta step we allow each tree’s weight estimation to be')
  parser.add_argument('--subsample', type=float, nargs='?', default=1, help='Subsample ratio of the training instance')
  parser.add_argument('--colsample_bytree', type=float, nargs='?', default=1, help='Subsample ratio of columns when constructing each tree')
  parser.add_argument('--colsample_bylevel', type=float, nargs='?', default=1, help='Subsample ratio of columns for each level')
  parser.add_argument('--colsample_bynode', type=float, nargs='?', default=1, help='Subsample ratio of columns for each node')
  parser.add_argument('-G', '--doGridSearchCV', action='store_true', help='Perform a grid search for optimal hyperparameter values using cross-validation')
  parser.add_argument('-V', '--plot_validation_curve', action='store_true', help='Calculate and plot perforance score as function of number of training events')
  parser.add_argument('-L', '--plot_learning_curve', action='store_true', help='Calculate and plot perforance score for different values of a chosen hyperparameter')
  args = parser.parse_args()

  # Set which sample types to prepare HDF5s for
  use_sig = 1
  use_bkg = 1
  use_data = 0

  # Where to put preprocessed datasets
  preproc_dir = 'preprocessed_datasets/'
  preproc_suffix = ''
  if args.prepare_hdf5:
    preproc_suffix = '_group_{}_preprocessed.h5'.format(args.prepare_hdf5)
  elif '2J' in args.signal_region:
    preproc_suffix = '_group_2L2J_preprocessed.h5'
  elif '3J+' in args.signal_region:
    preproc_suffix = '_group_2L3J+_preprocessed.h5'
  filename_sig_low_preprocessed = preproc_dir + 'sig_low' + preproc_suffix
  filename_sig_int_preprocessed = preproc_dir + 'sig_int' + preproc_suffix
  filename_sig_high_preprocessed = preproc_dir + 'sig_high' + preproc_suffix
  filename_sig_preprocessed = filename_sig_low_preprocessed
  filename_bkg_preprocessed = preproc_dir + 'bkg' + preproc_suffix
  filename_data_preprocessed = preproc_dir + 'data' + preproc_suffix

  # Where to put output
  output_dir = 'output/'
  #trained_model_dir = 'trained_models/'
  trained_model_dir = output_dir
  trained_model_xgb_suffix = '2LJets_trained_model.joblib'
  trained_model_nn_suffix = '2LJets_trained_model.h5'

  # Counters
  n_events_read = n_events_kept = 0
  n_events_read_sample = n_events_kept_sample = 0
  n_events_read_sample_type = n_events_kept_sample_type = 0

  if args.xgboost:
    output_dir += 'xgboost/latest/xgb_'
    trained_model_dir += 'xgboost/latest/xgb_'
  elif args.nn:
    output_dir += 'neural_network/latest/nn_'
    trained_model_dir += 'neural_network/latest/nn_'

  if 'low' in args.signal_region:
    output_dir += 'low_'
    trained_model_dir += 'low_'
  elif 'int' in args.signal_region:
    output_dir += 'int_'
    trained_model_dir += 'int_'
  elif 'high' in args.signal_region:
    output_dir += 'high_'
    trained_model_dir += 'high_'

  if args.train_even:
    output_dir += 'trainEven_'
    trained_model_dir += 'trainEven_'
  elif args.train_odd:
    output_dir += 'trainOdd_'
    trained_model_dir += 'trainOdd_'

  if args.xgboost:
    trained_model_path = trained_model_dir + trained_model_xgb_suffix
  elif args.nn:
    trained_model_path = trained_model_dir + trained_model_nn_suffix

  global df_sig_feat, df_bkg_feat, df_data_feat

  l_sig = []
  if use_sig:
    if 'low' in args.signal_region:
      l_sig = d_sig['low']
      filename_sig_preprocessed = filename_sig_low_preprocessed
    elif 'int' in args.signal_region:
    #elif args.signal_region == 'int':
      l_sig = d_sig['int']
      filename_sig_preprocessed = filename_sig_int_preprocessed
    elif 'high' in args.signal_region:
      l_sig = d_sig['high']
      filename_sig_preprocessed = filename_sig_high_preprocessed

    d_sig_infile = {'low': filename_sig_low_preprocessed, 
                    'int': filename_sig_int_preprocessed, 
                    'high': filename_sig_high_preprocessed}

  class Logger(object):
      def __init__(self):
          self.terminal = sys.stdout
          self.log = open(output_dir+".log", "w")

      def write(self, message):
          self.terminal.write(message)
          self.log.write(message)  

      def flush(self):
          #this flush method is needed for python 3 compatibility.
          #this handles the flush command by doing nothing.
          #you might want to specify some extra behavior here.
          pass    

  sys.stdout = Logger()

  if args.prepare_hdf5:
    """Read input dataset in chunks, select features and perform cuts,
    before storing DataFrame in HDF5 file"""

    # Prepare and store signal dataset
    if use_sig:
      prepareHDF5(filename_sig_low_preprocessed, d_sig['low'], sample_type='sig', selection=args.prepare_hdf5, chunk_size=1e5, n_chunks=None, entrystart=0)
      prepareHDF5(filename_sig_int_preprocessed, d_sig['int'], sample_type='sig', selection=args.prepare_hdf5, chunk_size=1e5, n_chunks=None, entrystart=0)
      prepareHDF5(filename_sig_high_preprocessed, d_sig['high'], sample_type='sig', selection=args.prepare_hdf5, chunk_size=1e5, n_chunks=None, entrystart=0)

    # Prepare and store background dataset
    if use_bkg:
      prepareHDF5(filename_bkg_preprocessed, l_bkg, sample_type='bkg', selection=args.prepare_hdf5, chunk_size=1e6, n_chunks=None, entrystart=0)
      #prepareHDF5(filename_bkg_preprocessed, l_bkg, sample_type='bkg', selection=args.prepare_hdf5, chunk_size=1e4, n_chunks=1, entrystart=0)

    # Prepare and store real dataset
    if use_data:
      prepareHDF5(filename_data_preprocessed, l_data, sample_type='data', selection=args.prepare_hdf5, chunk_size=1e5, n_chunks=None, entrystart=0)

    return

  elif args.read_hdf5:

    if use_sig:
      # Read in preprocessed signal DataFrame from HDF5 file
      df_sig_feat = pd.DataFrame({})

      for key_sig, value_sig_infile in d_sig_infile.items():
        if key_sig in args.signal_region:
          print("\nReading in file:", value_sig_infile)
          sig_store = pd.HDFStore(value_sig_infile)
          for i_sig in sig_store.keys(): #d_sig[key_sig]:
            if len(df_sig_feat) is 0:
              df_sig_feat = sig_store[i_sig]#.astype('float64')
              df_sig_feat['group'] = i_sig
            else:
              df_sig_sample = sig_store[i_sig]#.astype('float64')
              df_sig_sample['group'] = i_sig
              df_sig_feat = df_sig_feat.append(df_sig_sample)

      if 'mTl3' in df_sig_feat:
        df_sig_feat.drop(columns='mTl3', inplace=True)

      print("\ndf_sig_feat.head():\n", df_sig_feat.head())
      sig_store.close()
      print("Closed store")

    if use_bkg:
      # Read in preprocessed background DataFrame from HDF5 file
      df_bkg_feat = pd.DataFrame({})

      print("\nReading in file:", filename_bkg_preprocessed)
      bkg_store = pd.HDFStore(filename_bkg_preprocessed)
      for i_bkg in bkg_store.keys(): #l_bkg:
        if len(df_bkg_feat) is 0:
          df_bkg_feat = bkg_store[i_bkg]#.astype('float64')
          df_bkg_feat['group'] = i_bkg
        else:
          df_bkg_sample = bkg_store[i_bkg]#.astype('float64')
          df_bkg_sample['group'] = i_bkg
          df_bkg_feat = df_bkg_feat.append(df_bkg_sample)

      if 'mTl3' in df_bkg_feat:
        df_bkg_feat.drop(columns='mTl3', inplace=True)

      print("\ndf_bkg_feat.head():\n", df_bkg_feat.head())
      bkg_store.close()
      print("Closed store")

    if use_data:
      # Read in preprocessed DataFrame of real data from HDF5 file
      data_store = pd.HDFStore(filename_data_preprocessed)
      df_data_feat = data_store['data']
      print("\ndf_data_feat.head():\n", df_data_feat.head())
      data_store.close()
      print("Closed store")

  elif args.direct_read:
    """Read the input dataset for direct use, without reading in chunks
    and storing to output file"""

    print("Not available at the moment")
    return

    #entry_start = 0
    #sig_entry_stop = 1e4
    #bkg_entry_stop = 1e4

    ## import signal dataset
    #df_sig = importOpenData(sample_type="sig", entrystart=entry_start, entrystop=sig_entry_stop)
    #df_sig = shuffle(df_sig)  # shuffle the rows/events
    #df_sig_feat = selectFeatures(df_sig, l_features)
    #df_sig_feat = df_sig_feat*1  # multiplying by 1 to convert booleans to integers
    #df_sig_feat["eventweight"] = getEventWeights(df_sig, l_eventweights)

    ## import background dataset
    #df_bkg = importOpenData(sample_type="bkg", entrystart=entry_start, entrystop=bkg_entry_stop)
    #df_bkg = shuffle(df_bkg)  # shuffle the rows/events
    #df_bkg_feat = selectFeatures(df_bkg, l_features)
    #df_bkg_feat = df_bkg_feat*1  # multiplying by 1 to convert booleans to integers
    #df_bkg_feat["eventweight"] = getEventWeights(df_bkg, l_eventweights)

    ## import data
    ##df_data = importOpenData(sample_type="data", entrystart=entry_start, entrystop=entry_stop)

  if 'low' in args.signal_region:
    print('\nBefore xsec correction: df_sig_feat.query("DatasetNumber == 396210").loc[:,"eventweight"]\n', df_sig_feat.query("DatasetNumber == 396210").loc[:,"eventweight"].head())
    df_sig_feat.loc[df_sig_feat.DatasetNumber==396210,'eventweight'] = df_sig_feat.loc[df_sig_feat.DatasetNumber==396210,'eventweight'] * 0.08836675497457203
    print('\nAfter xsec correction: df_sig_feat.query("DatasetNumber == 396210").loc[:,"eventweight"]\n', df_sig_feat.query("DatasetNumber == 396210").loc[:,"eventweight"].head())

  # Preselection cuts
  l_presel = ['met_Sign > 2', 'mt2leplsp_0 > 10']
  #df_sig_feat.query('&'.join(l_presel), inplace=True)

  print("\n======================================")
  print("df_sig_feat.shape =", df_sig_feat.shape)
  print("df_bkg_feat.shape =", df_bkg_feat.shape)
  print("======================================")

  # make array of features
  df_X = pd.concat([df_bkg_feat, df_sig_feat], axis=0)#, sort=False)

  print("\ndf_X.isna().sum().sum()", df_X.isna().sum().sum())

  #print("\ndf_X.dtypes", df_X.dtypes)
  #col_float32 = (df_X.dtypes == 'float32').values
  #df_X.iloc[:, col_float32] = df_X.iloc[:, col_float32].astype('float64')
  #print("\nAfter converting all columns to float64:\ndf_X.dtypes", df_X.dtypes)

  # make array of labels
  y_bkg = np.zeros(len(df_bkg_feat))
  y_sig = np.ones(len(df_sig_feat))
  y = np.concatenate((y_bkg, y_sig), axis=0).astype(int)
  df_X['ylabel'] = y

  if args.multiclass:
    df_X.loc[df_X.group=='Zjets', 'ylabel'] = 2
    df_X.loc[df_X.group=='diboson', 'ylabel'] = 3
    df_X = df_X.query('group=="diboson" | group=="Zjets" | ylabel==1')
    Y = df_X.ylabel
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_multi = np_utils.to_categorical(encoded_Y)

  # Split the dataset in train and test sets
  test_size = 0.5
  seed = 42

  df_X_even = df_X.query("RandomRunNumber % 2 == 0")
  df_X_odd  = df_X.query("RandomRunNumber % 2 == 1")

  df_X_even = shuffle(df_X_even)
  df_X_odd = shuffle(df_X_odd)

  if args.train_even:
    X_train = df_X_even
    X_test = df_X_odd
  elif args.train_odd:
    X_train = df_X_odd
    X_test = df_X_even

  # Balance dataset by resampling: equal number of signal and background events
  if args.balanced >= 0:
    # Oversample signal
    if args.balanced is 0:
      N_train_sig = len(X_train.query('ylabel==0'))
    # Undersample background
    elif args.balanced is 1:
      N_train_sig = len(X_train.query('ylabel==1'))
    N_train_bkg = N_train_sig
    # Draw balanced training datasets where the number of signal and background events are equal
    X_train_sig = resample(X_train.query('ylabel==1'), replace=True, n_samples=N_train_sig, random_state=42)#, stratify=None)
    X_train_bkg = resample(X_train.query('ylabel==0'), replace=True, n_samples=N_train_bkg, random_state=42)#, stratify=None)
    X_train = pd.concat([X_train_bkg, X_train_sig], axis=0)

  print("\n---------- After balancing ----------")
  print("args.balanced =", args.balanced)
  print("X_train.query('ylabel==1').shape =", X_train.query('ylabel==1').shape)
  print("X_train.query('ylabel==1').shape =", X_train.query('ylabel==0').shape)
  print("---------------------------------------")

  #X_train_bkg = resample(X_train.query('group==Zjets'), replace=True, n_samples=N_train_bkg, random_state=42)#, stratify=None)
  #X_train = X_train.query('group=="diboson" | ylabel==1')

  # Draw validation set as subsample of test set, for quicker evaluation of validation loss during training
  n_val_samples = 1e5
  X_val = resample(X_test, replace=False, n_samples=n_val_samples, random_state=42, stratify=X_test.ylabel)
  y_val = X_val.ylabel

  y_train = X_train.ylabel
  y_test = X_test.ylabel

  # Making a copy of the DFs with only feature columns
  X_train_feat_only = X_train.copy()
  X_test_feat_only = X_test.copy()
  X_val_feat_only = X_val.copy()
  l_non_features = ['DatasetNumber', 'RandomRunNumber', 'eventweight', 'group', 'ylabel']
  X_train_feat_only.drop(l_non_features, axis=1, inplace=True)
  X_test_feat_only.drop(l_non_features, axis=1, inplace=True)
  X_val_feat_only.drop(l_non_features, axis=1, inplace=True)

  print("\nX_train_feat_only:", X_train_feat_only.columns)
  print("X_test_feat_only:", X_test_feat_only.columns)
  print("X_val_feat_only:", X_val_feat_only.columns)

  print("\nX_train_feat_only:", X_train_feat_only.shape)
  print("X_test_feat_only:", X_test_feat_only.shape)
  print("X_val_feat_only:", X_val_feat_only.shape)

  # Feature scaling
  # Scale all variables to the interval [0,1]
  #scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
  scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
  print("\nscaler.fit_transform(X_train_feat_only)")
  X_train_scaled = scaler.fit_transform(X_train_feat_only)
  print("scaler.transform(X_test_feat_only)")
  X_test_scaled = scaler.transform(X_test_feat_only)
  print("scaler.transform(X_val_feat_only)")
  X_val_scaled = scaler.transform(X_val_feat_only)

  
  print("\n\n//////////////////// ML part ////////////////////////")

  global model
  scale_pos_weight = 1
  event_weight = None
  class_weight = None
  class_weight_dict = {}

  if args.event_weight:
    event_weight = X_train.eventweight
    #event_weight = eventweight_train_resampled

  if args.class_weight:
    if args.xgboost:
      # XGBoost: Scale signal events up by a factor n_bkg_train_events / n_sig_train_events
      scale_pos_weight = len(X_train[X_train.ylabel == 0]) / len(X_train[X_train.ylabel == 1]) 
      #scale_pos_weight = 10
    else:
      # sciki-learn: Scale overrespresented sample down (bkg) and underrepresented sample up (sig)
      class_weight = "balanced"
  else:
    class_weight = None

  print("\n# bkg train events / # sig train events = {0:d} / {1:d}".format(len(X_train[X_train.ylabel == 0]), len(X_train[X_train.ylabel == 1])))
  print("scale_pos_weight =", scale_pos_weight)

  classes = np.unique(y)
  class_weight_vect = compute_class_weight(class_weight, classes, y)
  class_weight_dict = {0: class_weight_vect[0], 1: class_weight_vect[1]}

  # Initialize variables for storing CV output
  valid_score = test_score = fit_time = score_time = 0
  # Initialize variables for storing validation and learning curve output
  train_scores_vc_mean = train_scores_vc_std = 0
  valid_scores_vc_mean = valid_scores_vc_std = 0
  train_scores_lc_mean = train_scores_lc_std = 0
  valid_scores_lc_mean = valid_scores_lc_std = 0

  # List of training set sizes for plotting of learning curve
  train_sizes = [0.5, 0.75, 1.0]

  # List of parameter values for hyperparameter grid search
  # XGBoost
  max_depth = [5, 6, 8, 10]
  n_estimators = [50, 100, 200, 500, 1000]
  learning_rate = [0.001, 0.01, 0.1, 0.5, 1.0]
  reg_alpha = [0, 0.001, 0.01, 0.1, 1.]
  reg_lambda = [0, 0.001, 0.01, 0.1, 1.]

  d_param_grid_xgb = {'max_depth': max_depth,
                      'n_estimators': n_estimators,
                      'learning_rate': learning_rate,
                      'reg_alpha': reg_alpha,
                      'reg_lambda': reg_lambda
                      }

  # Specify one of the above parameter lists to plot validation curve for
  param_name_xgb = 'max_depth'
  param_range_xgb = d_param_grid_xgb[param_name_xgb]

  # Neural network
  n_hidden_layers = [1, 3, 5, 7, 10]
  n_nodes = [10, 20, 50, 100, 500]
  batch_size = [8, 16, 32, 64, 128]
  epochs = [10, 50, 100, 500, 1000]
  #kernel_regularizer = [l1_l2(l1=1e-6, l2=1e-6), l1_l2(l1=1e-6, l2=1e-5), l1_l2(l1=1e-5, l2=1e-6), l1_l2(l1=1e-5, l2=1e-5)]
  d_param_grid_nn = {'n_hidden_layers': [1] #n_hidden_layers,
                     #'n_nodes': #n_nodes,
                     #'batch_size': batch_size,
                     #'epochs': epochs,
                     #'kernel_regularizer': kernel_regularizer
                    }

  # Specify one of the above parameter lists to plot validation curve for
  param_name_nn = 'n_hidden_layers'
  param_range_nn = d_param_grid_nn[param_name_nn]

  if args.xgboost:
    param_range = param_range_xgb
    param_name = param_name_xgb
  elif args.nn:
    param_range = param_range_nn
    param_name = param_name_nn


  # Run XGBoost BDT
  if args.xgboost:

    if args.multiclass:
      objective = 'multi:softmax'
      eval_metric = 'mlogloss'
    else:
      objective = 'binary:logistic'
      eval_metric = 'logloss'
      #eval_metric = 'auc'

    max_depth = args.max_depth
    lr = args.lr
    n_estimators = args.n_estimators
    gamma = args.gamma
    min_child_weight = args.min_child_weight
    max_delta_step = args.max_delta_step
    subsample = args.subsample
    colsample_bytree = args.colsample_bytree
    colsample_bylevel = args.colsample_bylevel
    colsample_bynode = args.colsample_bynode
    reg_alpha = args.L1
    reg_lambda = args.L2

    if not args.load_pretrained_model:
      model = XGBClassifier(max_depth=max_depth, 
                            learning_rate=lr,
                            n_estimators=n_estimators, 
                            verbosity=1,
                            objective=objective, 
                            n_jobs=-1,
                            gamma=gamma,
                            min_child_weight=min_child_weight,
                            max_delta_step=max_delta_step,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            colsample_bylevel=colsample_bylevel,
                            colsample_bynode=colsample_bynode,
                            reg_alpha=reg_alpha,  # L1 regularization
                            reg_lambda=reg_alpha, # L2 regularization
                            scale_pos_weight=scale_pos_weight)

      print("\nmodel.get_params()\n", model.get_params())

      if not args.plot_validation_curve and not args.plot_learning_curve:

        if args.doGridSearchCV:
          model = GridSearchCV(model, d_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    
        print("\nTraining XGBoost BDT...")

        if args.doCV:

          cv_results = cross_validate(model, X_train_scaled, y_train, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1, return_train_score=True)

          valid_score = cv_results['test_score']
          train_score = cv_results['train_score']
          fit_time = cv_results['fit_time']
          score_time = cv_results['score_time']
          fit_time = cv_results['fit_time']

        else:
          model.fit(X_train_scaled, y_train, 
                    sample_weight=event_weight, 
                    eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
                    #eval_set=[(X_val_scaled, y_val)],
                    eval_metric=eval_metric,
                    early_stopping_rounds=20,
                    verbose=True)

          evals_result = model.evals_result()
          sns.set()
          ax = sns.lineplot(x=range(0, len(evals_result['validation_0'][eval_metric])), y=evals_result['validation_0'][eval_metric], label='Training loss')
          ax = sns.lineplot(x=range(0, len(evals_result['validation_1'][eval_metric])), y=evals_result['validation_1'][eval_metric], label='Validation loss')
          ax.set(xlabel='Epochs', ylabel='Loss')
          plt.show()

        print("\nTraining done!")

        if args.doGridSearchCV:
          joblib.dump(model.best_estimator_, trained_model_path)
        else:
          joblib.dump(model, trained_model_path)
        print("\nSaving the trained XGBoost BDT:", trained_model_path)

    elif args.load_pretrained_model:
      print("\nReading in pre-trained XGBoost BDT:", trained_model_path)
      model = joblib.load(trained_model_path)


  # Run neural network
  elif args.nn:

    n_inputs = X_train_scaled.shape[1]
    n_nodes = args.n_nodes
    n_hidden_layers = args.n_hidden_layers
    dropout_rate = args.dropout
    batch_size = args.batch_size
    epochs = args.epochs
    l1 = args.L1
    l2 = args.L2
    lr = args.lr

    if not args.load_pretrained_model:
      print("\nBuilding and training neural network")

      es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

      model = KerasClassifier(build_fn=create_model,
                              n_inputs=n_inputs,
                              n_hidden_layers=n_hidden_layers,
                              n_nodes=n_nodes,
                              dropout_rate=dropout_rate,
                              l1=l1,
                              l2=l2,
                              lr=lr,
                              batch_size=batch_size, 
                              epochs=epochs, 
                              verbose=1,
                              )

      if not args.plot_validation_curve and not args.plot_learning_curve:

        if args.doGridSearchCV:
          param_grid = d_param_grid_nn
          model = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)

        history = model.fit(X_train_scaled, y_train, 
                            sample_weight=event_weight, 
                            class_weight=class_weight_dict,
                            verbose=1,
                            callbacks=[es],
                            validation_data=(X_val_scaled, y_val)
                            #validation_data=(X_test_scaled, y_test)
                            )

        print("\nmodel.model.summary()\n", model.model.summary())

        if not args.doGridSearchCV:
          d_val_loss = {'Training loss': history.history['loss'], 'Validation loss': history.history['val_loss']}
          df_val_loss = pd.DataFrame(d_val_loss)

          sns.set()
          ax = sns.lineplot(data=df_val_loss)
          ax.set(xlabel='Epochs', ylabel='Loss')
          plt.show()

        if args.doGridSearchCV:
          model.best_estimator_.model.save(trained_model_path)
        else:
          model.model.save(trained_model_path)
        print("\nSaving the trained neural network:", trained_model_path)

    elif args.load_pretrained_model:
      print("\nReading in pre-trained neural network:", trained_model_path)
      model = load_model(trained_model_path)

  if not args.plot_validation_curve and not args.plot_learning_curve:

    # Print results of grid search
    if args.doGridSearchCV:
      print("Best parameters set found on development set:")
      print("")
      print("model.best_params_", model.best_params_)
      print("")
      print("Grid scores on development set:")
      means = model.cv_results_['mean_test_score']
      stds = model.cv_results_['std_test_score']
      for mean, std, params in zip(means, stds, model.cv_results_['params']):
          print("{0:0.3f} (+/-{1:0.03f}) for {2!r}".format(mean, std, params))
      print("")
      df = pd.DataFrame.from_dict(model.cv_results_)
      print("pandas DataFrame of cv results")
      print(df)
      print("")

    # Get predicted signal probabilities for train and test sets
    output_train = model.predict_proba(X_train_scaled)
    output_test = model.predict_proba(X_test_scaled)
    #X_train = X_train.copy()
    #X_test = X_test.copy()

    if args.multiclass:
      output_test = output_test.reshape(output_test.shape[0], 3)
      print("output_train", len(output_train[0]))

      for i_output in range(len(output_train[0])):
        X_train["output"+str(i_output)] = output_train[:,i_output]
        X_test["output"+str(i_output)] = output_test[:,i_output]

    elif output_train.shape[1] is 2:
      print("output_train[:10,1]", output_train[:10,1])
      X_train["output"] = output_train[:,1]
      X_test["output"] = output_test[:,1]

    else:
      X_train["output"] = output_train
      X_test["output"] = output_test


    print("\n\n//////////////////// Plotting part ////////////////////////\n")

    if not args.multiclass:
      print("len(X_train.query('ylabel==0').loc[:,'eventweight'])", len(X_train.query('ylabel==0').loc[:,'eventweight']))
      print("len(X_train.query('ylabel==0').loc[:,'output'])", len(X_train.query('ylabel==0').loc[:,'output']))
      print("X_train.query('ylabel==0').loc[:,'eventweight']", X_train.query("ylabel==0").loc[:,"eventweight"].head())
      print("X_train.query('ylabel==0').loc[:,'output']", X_train.query("ylabel==0").loc[:,"output"].head())

      print("X_train[['eventweight', 'output']].min(): \n", X_train[['eventweight', 'output']].min())
      print("X_train[['eventweight', 'output']].max(): \n", X_train[['eventweight', 'output']].max())
    
    l_X_train_bkg = [X_train.query('group=="/bkg/'+i_bkg+'"').filter(like='output') for i_bkg in l_bkg]
    l_ew_train_bkg = [X_train.query('group=="/bkg/'+i_bkg+'"').loc[:,'eventweight'] for i_bkg in l_bkg]
    l_X_test_bkg = [X_test.query('group=="/bkg/'+i_bkg+'"').filter(like='output') for i_bkg in l_bkg]
    l_ew_test_bkg = [X_test.query('group=="/bkg/'+i_bkg+'"').loc[:,'eventweight'] for i_bkg in l_bkg]

    l_X_train_sig = [X_train.query('ylabel==1 & group=="/sig/'+i_sig+'"').filter(like='output') for i_sig in l_sig]
    l_ew_train_sig = [X_train.query('ylabel==1 & group=="/sig/'+i_sig+'"').loc[:,'eventweight'] for i_sig in l_sig]
    l_X_test_sig = [X_test.query('ylabel==1 & group=="/sig/'+i_sig+'"').filter(like='output') for i_sig in l_sig]
    l_ew_test_sig = [X_test.query('ylabel==1 & group=="/sig/'+i_sig+'"').loc[:,'eventweight'] for i_sig in l_sig]

    d_X_train_bkg = dict(zip(l_bkg, l_X_train_bkg))
    d_ew_train_bkg = dict(zip(l_bkg, l_ew_train_bkg))
    d_X_test_bkg = dict(zip(l_bkg, l_X_test_bkg))
    d_ew_test_bkg = dict(zip(l_bkg, l_ew_test_bkg))

    # Plot unweighted training and test output
    #plt.figure(1)
    #plotTrainTestOutput(d_X_train_bkg, None,
    #                    X_train.query("ylabel==1").loc[:,"output"], None,
    #                    d_X_test_bkg, None,
    #                    X_test.query("ylabel==1").loc[:,"output"], None)
    #plotTrainTestOutput(d_X_train_bkg, None,
    #                    X_train.query("ylabel==1").loc[:,"output"], None,
    #                    d_X_test_bkg, None,
    #                    X_test.query("ylabel==1").loc[:,"output"], None)
    #plt.savefig(output_dir + 'hist1_train_test_unweighted.pdf')

    # Plot weighted train and test output, with test set multiplied by 2 to match number of events in training set
    plt.figure()
    #for i_output in range(output_train.shape[1]):
    plotTrainTestOutput(d_X_train_bkg, d_ew_train_bkg,
                        X_train.query("ylabel==1").filter(like='output'), X_train.query("ylabel==1").loc[:,"eventweight"],
                        d_X_test_bkg, d_ew_test_bkg,
                        X_test.query("ylabel==1").filter(like='output'), X_test.query("ylabel==1").loc[:,"eventweight"],
                        args.signal_region)
    plt.savefig(output_dir + 'hist_train_test_weighted_comparison.pdf')

    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
    if 'low' in args.signal_region:
      plt.figure()
      plotFinalTestOutput(d_X_test_bkg,
                          d_ew_test_bkg,
                          X_test.query("ylabel==1 & (DatasetNumber==392330 | DatasetNumber==396210)").filter(like='output'),
                          X_test.query("ylabel==1 & (DatasetNumber==392330 | DatasetNumber==396210)").loc[:,"eventweight"],
                          args.signal_region,
                          figure_text='(200, 100) GeV')
      plt.savefig(output_dir + 'hist_test_392330_396210_C1N2_WZ_2L2J_200_100_weighted.pdf')
    elif 'int' in args.signal_region:
      plt.figure()
      plotFinalTestOutput(d_X_test_bkg,
                          d_ew_test_bkg,
                          X_test.query("ylabel==1 & DatasetNumber==392325").loc[:,"output"],
                          X_test.query("ylabel==1 & DatasetNumber==392325").loc[:,"eventweight"],
                          args.signal_region,
                          figure_text='(500, 200) GeV')
      plt.savefig(output_dir + 'hist_test_392325_C1N2_WZ_2L2J_500_200_weighted.pdf')
    elif 'high' in args.signal_region:
      plt.figure()
      plotFinalTestOutput(d_X_test_bkg,
                          d_ew_test_bkg,
                          X_test.query("ylabel==1 & DatasetNumber==392356").loc[:,"output"],
                          X_test.query("ylabel==1 & DatasetNumber==392356").loc[:,"eventweight"],
                          args.signal_region,
                          figure_text='(600, 0) GeV')
      plt.savefig(output_dir + 'hist5_test_392356_C1N2_WZ_2L2J_600_0_weighted.pdf')


    if args.xgboost and not args.doGridSearchCV:
      # Plot feature importance
      print("model.feature_importances_", model.feature_importances_)
      print("np.sum(model.feature_importances_)", np.sum(model.feature_importances_))
      if args.multiclass:
        l_feat_drop = ['DatasetNumber', 'RandomRunNumber', 'eventweight', 'group', 'ylabel', 'output0', 'output1', 'output2']
      else:
        l_feat_drop = ['DatasetNumber', 'RandomRunNumber', 'eventweight', 'group', 'ylabel', 'output']
      s_feat_importance = pd.Series(model.feature_importances_, index=X_train.drop(l_feat_drop, axis=1).columns)
      print("X_train.drop(l_feat_drop, axis=1).columns\n", X_train.drop(l_feat_drop, axis=1).columns)
      s_feat_importance.sort_values(ascending=False, inplace=True)

      plt.figure()
      sns.set(style="ticks", color_codes=True)
      n_top_feat_importance = 20
      ax = sns.barplot(x=s_feat_importance[:n_top_feat_importance]*100, y=s_feat_importance[:n_top_feat_importance].index)#, palette="Blues_r")
      #ax.set_yticklabels(s_feat_importance.index)
      ax.set(xlabel="Feature importance [%]")
      plt.savefig(output_dir + 'feature_importance.pdf')


    if not args.multiclass:
      # Plot ROC curve
      fpr, tpr, thresholds = metrics.roc_curve(X_test.loc[:,"ylabel"], X_test.loc[:,"output"])
      auc = metrics.roc_auc_score(X_test.loc[:,"ylabel"], X_test.loc[:,"output"])

      plt.figure()
      ax = sns.lineplot(x=tpr, y=1-fpr, estimator=None, label='ROC curve: AUC = %0.2f' % auc)
      plt.plot([1,0], [0,1], linestyle="--")
      ax.set(xlabel="Signal efficiency", ylabel="Background efficiency")
      plt.savefig(output_dir + 'ROC_curve_AUC_sigEff_vs_1minBkgEff.pdf')

      plt.figure()
      ax = sns.lineplot(x=tpr, y=1/(fpr), estimator=None, label='ROC curve: AUC = %0.2f' % auc)
      #plt.plot([0,1], [0,1], linestyle="--")
      ax.set(xlabel="Signal efficiency", ylabel="Background rejection = 1/(1 - bkg eff.)", yscale='log')
      plt.savefig(output_dir + 'ROC_curve_AUC_sigEff_vs_bkgRej.pdf')


    plt.show()


    # Signal significance
    print("\n///////////////// Signal significance /////////////////")

    def significance(cut_string_sig, cut_string_bkg, rel_unc=0.3):
      sig_exp = np.sum(X_test.query("ylabel == 1 & "+cut_string_sig).loc[:,"eventweight"])
      bkg_exp = np.sum(X_test.query("(ylabel == 0 | ylabel == 2 | ylabel == 3) & "+cut_string_bkg).loc[:,"eventweight"])
      Z_N_exp = RooStats.NumberCountingUtils.BinomialExpZ(sig_exp, bkg_exp, rel_unc)
      return [sig_exp, bkg_exp, Z_N_exp]

    #cut_string_DSID = 'DatasetNumber == {0:d}'.format(dsid)
    if 'low' in args.signal_region: 
      key = '(200, 100)'
      cut_string_DSID = '(DatasetNumber == 392330 | DatasetNumber == 396210)'
    elif 'int' in args.signal_region: 
      key = '(500, 200)'
      cut_string_DSID = 'DatasetNumber == 392325'
    elif 'high' in args.signal_region: 
      key = '(600, 0)'
      cut_string_DSID = 'DatasetNumber == 392356'

    l_cuts = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    global cut_optimal
    cut_optimal = 0
    Z_N_optimal = 0
    for cut in l_cuts:

      if args.multiclass:
        cut_string_SR = 'output0 > {:f}'.format(cut)
      else:
        cut_string_SR = 'output > {:f}'.format(cut)
      cut_string_bkg = cut_string_SR
      cut_string_sig = cut_string_SR + " & " + cut_string_DSID
      print('\ncut_string_sig:', cut_string_sig)
      print('cut_string_bkg:', cut_string_bkg)

      [sig_exp, bkg_exp, Z_N_exp] = significance(cut_string_sig, cut_string_bkg, rel_unc=0.3)
      print("---", key)
      print("S_exp =", sig_exp)
      print("B_exp =", bkg_exp)
      for i in range(len(l_X_train_bkg)):
        l_cut_strings = ['ylabel == 0', 'group == "/bkg/{}"'.format(l_bkg[i]), cut_string_bkg]
        B_exp_i = np.sum(X_test.query('&'.join(l_cut_strings)).loc[:,"eventweight"])
        print("  {0}: {1}".format(l_bkg[i], B_exp_i))
      print("Z_N_exp =", Z_N_exp)

      if sig_exp >= 3 and bkg_exp >= 1:
        if Z_N_exp > Z_N_optimal:
          Z_N_optimal = Z_N_exp
          cut_optimal = cut

    # Print the optimal SR values
    if args.multiclass:
      cut_string_SR = 'output0 > {:f}'.format(cut_optimal)
    else:
      cut_string_SR = 'output > {:f}'.format(cut_optimal)
    cut_string_bkg = cut_string_SR
    cut_string_sig = cut_string_SR + " & " + cut_string_DSID
    print('\ncut_string_sig:', cut_string_sig)
    print('cut_string_bkg:', cut_string_bkg)


    [sig_exp, bkg_exp, Z_N_exp] = significance(cut_string_sig, cut_string_bkg, rel_unc=0.3)
    print("---", key)
    print("Optimal cut =", cut_optimal)
    print("S_exp =", sig_exp)
    print("B_exp =", bkg_exp)
    for i in range(len(l_X_train_bkg)):
      l_cut_strings = ['ylabel == 0', 'group == "/bkg/{}"'.format(l_bkg[i]), cut_string_bkg]
      B_exp_i = np.sum(X_test.query('&'.join(l_cut_strings)).loc[:,"eventweight"])
      print("  {0}: {1}".format(l_bkg[i], B_exp_i))
    print("Z_N_exp =", Z_N_exp)



  if args.plot_validation_curve:
    print("\nCalculating validation curve...")
    train_scores, valid_scores = validation_curve(model, X_train_scaled, y_train, 
                                                  param_name=param_name, param_range=param_range,
                                                  cv=3, 
                                                  scoring='roc_auc', 
                                                  n_jobs=-1,
                                                  verbose=11)

    train_scores_vc_mean = np.mean(train_scores, axis=1)
    train_scores_vc_std = np.std(train_scores, axis=1)
    valid_scores_vc_mean = np.mean(valid_scores, axis=1)
    valid_scores_vc_std = np.std(valid_scores, axis=1)
 
    # Plot validation curves
    figF, axsF = plt.subplots()
    # Training score
    axsF.plot( param_range, train_scores_vc_mean, 'o-', label="Training score", color="darkorange", lw=2)
    axsF.fill_between( param_range, train_scores_vc_mean - train_scores_vc_std, train_scores_vc_mean + train_scores_vc_std, alpha=0.2, color="darkorange", lw=2)
    # Test score
    axsF.plot( param_range, valid_scores_vc_mean, 'o-', label="Cross-validation score", color="navy", lw=2)
    axsF.fill_between( param_range, valid_scores_vc_mean - valid_scores_vc_std, valid_scores_vc_mean + valid_scores_vc_std, alpha=0.2, color="navy", lw=2)
    axsF.set_xlabel(param_name)
    axsF.set_ylabel('Score')
    axsF.legend(loc="best")
    axsF.set_title('Validation curves')
    #axsF.set_ylim(0., 1.)
    plt.savefig(output_dir + 'validation_curve_{}.pdf'.format(param_name))
    plt.show()

  if args.plot_learning_curve:
    print("\nCalculating learning curve...")
    train_sizes, train_scores, valid_scores = learning_curve(model, X_train_scaled, y_train, train_sizes=train_sizes,
                                                             cv=3, scoring='roc_auc', n_jobs=1, verbose=3)
    train_scores_lc_mean = np.mean(train_scores, axis=1)
    train_scores_lc_std = np.std(train_scores, axis=1)
    valid_scores_lc_mean = np.mean(valid_scores, axis=1)
    valid_scores_lc_std = np.std(valid_scores, axis=1)

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
    #axsG.set_ylim(0., 1.)
    plt.savefig(output_dir + 'learning_curve.pdf')
    plt.show()


  # Stop timer
  t_end = time.time()
  print("\nProcess time: {:4.2f} s".format(t_end - t_start))


def custom_logloss(y_pred, y_true):
  return 'custom_logloss', log_loss(y_true, y_pred, sample_weight=None, eps=1e-7)

if __name__ == "__main__":
  main()
