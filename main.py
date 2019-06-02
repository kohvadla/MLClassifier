#!/usr/bin/env python3

from __future__ import division

import os
import sys
import time
import argparse
import pickle
import itertools

from ROOT import TLorentzVector, RooStats
import uproot
import numpy as np
import pandas as pd
#from pandas import HDFStore
import seaborn as sns

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
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba 
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

import logging

logging.basicConfig(
                    format="%(message)s",
                    level=logging.DEBUG,
                    stream=sys.stdout)


from importData import *
from plotting import *
#from hepFunctions import invariantMass


def main():

  # Start timer
  t_start = time.time()

  parser = argparse.ArgumentParser()
  group_model = parser.add_mutually_exclusive_group() 
  group_model.add_argument('-x', '--xgboost', action='store_true', help='Run gradient BDT')
  group_model.add_argument('-n', '--nn', action='store_true', help='Run neural network')
  group_read_dataset = parser.add_mutually_exclusive_group() 
  group_read_dataset.add_argument('-p', '--prepare_hdf5', action='store_true', help='Prepare input datasets for ML and store in HDF5 file')
  group_read_dataset.add_argument('-r', '--read_hdf5', action='store_true', help='Read prepared datasets from HDF5 file')
  group_read_dataset.add_argument('-d', '--direct_read', action='store_true', help='Read unprepared datasets from ROOT file')
  parser.add_argument('-B', '--N_sig_events', type=lambda x: int(float(x)), default=1e3, help='Number of signal events to read from the dataset')
  parser.add_argument('-S', '--N_bkg_events', type=lambda x: int(float(x)), default=1e3, help='Number of background events to read from the dataset for each class')
  parser.add_argument('-e', '--event_weight', action='store_true', help='Apply event weights during training')
  parser.add_argument('-c', '--class_weight', action='store_true', help='Apply class weights to account for unbalanced dataset')
  args = parser.parse_args()

  global df_sig_feat, df_bkg_feat, df_data_feat
  filename_preprocessed = 'openData_preprocessed.h5'

  if args.prepare_hdf5:
    """Read input dataset in chunks, select features and perform cuts,
    before storing DataFrame in HDF5 file"""

    # Import OpenData ntuples as flat pandas DataFrames
    exists = os.path.isfile(filename_preprocessed)
    if exists:
      os.remove(filename_preprocessed)
      print("Removed existing file", filename_preprocessed)
    store = pd.HDFStore(filename_preprocessed)
    print("Created new store with name", filename_preprocessed)

    #n_events_chunk = -1
    #print("\nIn main(): n_events_chunk =", n_events_chunk)

    # Prepare and store signal dataset
    # -- C1N2 via WZ to 2L2J:     119 000 events in total
    store = prepareInput(store, sample_type='sig', filename=filename_preprocessed, chunk_size=1e5, n_chunks=None, entrystart=0)
    # Prepare and store background dataset
    # -- diboson WZqqll 361607: 1 469 000 events in total
    store = prepareInput(store, sample_type='bkg', filename=filename_preprocessed, chunk_size=1e5, n_chunks=None, entrystart=0)
    store = prepareInput(store, sample_type='data', filename=filename_preprocessed, chunk_size=1e5, n_chunks=None, entrystart=0)

    print("\nReturned from prepareInput()")
    print("\nstore:\n", store)
    print("\nstore.keys()", store.keys())

    df_sig_feat = store['sig']
    df_bkg_feat = store['bkg']
    #df_data_feat = store['data']
    print("\ndf_sig_feat.head():\n", df_sig_feat.head())
    print("\ndf_bkg_feat.head():\n", df_bkg_feat.head())

    store.close()
    print("Closed store")

  elif args.read_hdf5:
    store = pd.HDFStore(filename_preprocessed)

    df_sig_feat = store['sig']
    df_bkg_feat = store['bkg']
    #df_data_feat = store['data']
    print("\ndf_sig_feat.head():\n", df_sig_feat.head())
    print("\ndf_bkg_feat.head():\n", df_bkg_feat.head())

    store.close()
    print("Closed store")

  elif args.direct_read:
    """Read the input dataset for direct use, without reading in chunks
    and storing to output file"""

    entry_start = 0
    sig_entry_stop = 1e4
    bkg_entry_stop = 1e4

    # import signal dataset
    df_sig = importOpenData(sample_type="sig", entrystart=entry_start, entrystop=sig_entry_stop)
    df_sig = shuffle(df_sig)  # shuffle the rows/events
    df_sig_feat = selectFeatures(df_sig, l_features)
    df_sig_feat = df_sig_feat*1  # multiplying by 1 to convert booleans to integers
    df_sig_feat["eventweight"] = getEventWeights(df_sig, l_eventweights)

    # import background dataset
    df_bkg = importOpenData(sample_type="bkg", entrystart=entry_start, entrystop=bkg_entry_stop)
    df_bkg = shuffle(df_bkg)  # shuffle the rows/events
    df_bkg_feat = selectFeatures(df_bkg, l_features)
    df_bkg_feat = df_bkg_feat*1  # multiplying by 1 to convert booleans to integers
    df_bkg_feat["eventweight"] = getEventWeights(df_bkg, l_eventweights)

    ## import data
    ##df_data = importOpenData(sample_type="data", entrystart=entry_start, entrystop=entry_stop)

  print("\n======================================")
  print("df_sig_feat.shape        =", df_sig_feat.shape)
  print("df_bkg_feat.shape        =", df_bkg_feat.shape)
  print("======================================")

  # make array of features
  df_X = pd.concat([df_bkg_feat, df_sig_feat], axis=0)#, sort=False)

  # make array of labels
  y_bkg = np.zeros(len(df_bkg_feat))
  y_sig = np.ones(len(df_sig_feat))
  y = np.concatenate((y_bkg, y_sig), axis=0).astype(int)
  df_X["ylabel"] = y

  print("\n- Input arrays:")
  print("df_X.drop(['eventweight', 'ylabel']) :", df_X.drop(["eventweight", "ylabel"], axis=1).shape)
  print("df_X.eventweight                     :", df_X.eventweight.shape)
  print("df_X.ylabel                          :", df_X.ylabel.shape)

  # Split the dataset in train and test sets
  test_size = 0.33
  seed = 42

  X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=test_size, random_state=seed, stratify=y)

  print("\nX_train:", X_train.columns)
  print("X_test:", X_test.columns)

  print("\nX_train:", X_train.shape)
  print("X_test:", X_test.shape)
  print("y_train:", y_train.shape)
  print("y_test:", y_test.shape)

  print("\nisinstance(X_train, pd.DataFrame):", isinstance(X_train, pd.DataFrame))
  print("isinstance(X_test, pd.DataFrame):", isinstance(X_test, pd.DataFrame))
  print("isinstance(y_train, pd.Series):", isinstance(y_train, pd.Series))
  print("isinstance(y_test, pd.Series):", isinstance(y_test, pd.Series))

  # Making a copy of the train and test DFs with only feature columns
  X_train_feat_only = X_train.copy()
  X_test_feat_only = X_test.copy()
  X_train_feat_only.drop(["channelNumber", "eventweight", "ylabel"], axis=1, inplace=True)
  X_test_feat_only.drop(["channelNumber", "eventweight", "ylabel"], axis=1, inplace=True)

  # Feature scaling
  # Scale all variables to the interval [0,1]
  min_max_scaler = preprocessing.MinMaxScaler()
  X_train_scaled = min_max_scaler.fit_transform(X_train_feat_only)
  X_test_scaled = min_max_scaler.transform(X_test_feat_only)

  
  print("\n\n//////////////////// ML part ////////////////////////")

  global model
  scale_pos_weight = 1
  event_weight = None
  class_weight = None

  if args.event_weight:
    event_weight = X_train.eventweight

  if args.class_weight:
    if args.xgboost:
      # XGBoost: Scale signal events up by a factor n_bkg_train_events / n_sig_train_events
      scale_pos_weight = len(X_train[X_train.ylabel == 0]) / len(X_train[X_train.ylabel == 1]) 
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

  # Run XGBoost BDT
  if args.xgboost:
    print("\nRunning XGBoost BDT")

    model = XGBClassifier(max_depth=3, 
                          learning_rate=0.1, 
                          n_estimators=100, 
                          verbosity=3, 
                          objective="binary:logistic", 
                          scale_pos_weight=scale_pos_weight)
    model.fit(X_train_scaled, y_train, 
              sample_weight=event_weight)
    print("\nBuilding and training BDT\n")
    print("\nX_train_scaled.shape\n", X_train_scaled.shape)

  # Run neural network
  elif args.nn:
    print("\nRunning neural network")

  # Get predicted signal probabilities for train and test sets
  output_train = model.predict_proba(X_train_scaled)
  output_test = model.predict_proba(X_test_scaled)
  X_train = X_train.copy()
  X_test = X_test.copy()
  print("\nBefore X_train['output'] = output_train[:,1]: X_train =\n", X_train.head())
  X_train["output"] = output_train[:,1]
  X_test["output"] = output_test[:,1]
  print("\nAfter X_train['output'] = output_train[:,1]: X_train =\n", X_train.head())


  print("\n\n//////////////////// Plotting part ////////////////////////\n")

  # Set seaborn style
  sns.set(color_codes=True)

  print("len(X_train.query('ylabel==0').loc[:,'eventweight'])", len(X_train.query('ylabel==0').loc[:,'eventweight']))
  print("len(X_train.query('ylabel==0').loc[:,'output'])", len(X_train.query('ylabel==0').loc[:,'output']))
  print("X_train.query('ylabel==0').loc[:,'eventweight']", X_train.query("ylabel==0").loc[:,"eventweight"].head())
  print("X_train.query('ylabel==0').loc[:,'output']", X_train.query("ylabel==0").loc[:,"output"].head())

  print("X_train[['eventweight', 'output']].min(): \n", X_train[['eventweight', 'output']].min())
  print("X_train[['eventweight', 'output']].max(): \n", X_train[['eventweight', 'output']].max())
  
  # Plot unweghted training (2/3) and test (1/3) output
  plt.figure(1)
  plotTrainTestOutput(X_train.query("ylabel==0").loc[:,"output"], None,
                      X_train.query("ylabel==1").loc[:,"output"], None,
                      X_test.query("ylabel==0").loc[:,"output"], None,
                      X_test.query("ylabel==1").loc[:,"output"], None)
  plt.savefig('hist1_train_test_unweighted.pdf')

  # Plot weighted train and test output, with test set multiplied by 2 to match number of events in training set
  plt.figure(2)
  plotTrainTestOutput(X_train.query("ylabel==0").loc[:,"output"],
                      X_train.query("ylabel==0").loc[:,"eventweight"],
                      X_train.query("ylabel==1 & channelNumber==392323").loc[:,"output"],
                      X_train.query("ylabel==1 & channelNumber==392323").loc[:,"eventweight"],
                      X_test.query("ylabel==0").loc[:,"output"],
                      2*X_test.query("ylabel==0").loc[:,"eventweight"],
                      X_test.query("ylabel==1 & channelNumber==392323").loc[:,"output"],
                      2*X_test.query("ylabel==1 & channelNumber==392323").loc[:,"eventweight"])
  plt.savefig('hist2_train_test_weighted_comparison.pdf')

  # Plot final signal vs background estimate for test set, scaled to 10.6/fb
  plt.figure(3)
  plotFinalTestOutput(X_test.query("ylabel==0").loc[:,"output"],
                      3*X_test.query("ylabel==0").loc[:,"eventweight"],
                      X_test.query("ylabel==1 & channelNumber==392330").loc[:,"output"],
                      3*X_test.query("ylabel==1 & channelNumber==392330").loc[:,"eventweight"])#,
                      #figure_text='10~fb$^{-1}$')
  plt.savefig('hist3_test_392330_C1N2_WZ_2L2J_200_100_weighted_corrected.pdf')

  plt.figure(4)
  plotFinalTestOutput(X_test.query("ylabel==0").loc[:,"output"],
                      3*X_test.query("ylabel==0").loc[:,"eventweight"],
                      X_test.query("ylabel==1 & channelNumber==392304").loc[:,"output"],
                      3*X_test.query("ylabel==1 & channelNumber==392304").loc[:,"eventweight"])#,
                      #figure_text='10~fb$^{-1}$')
  plt.savefig('hist4_test_392304_C1N2_WZ_2L2J_300_100_weighted_corrected.pdf')

  plt.figure(5)
  plotFinalTestOutput(X_test.query("ylabel==0").loc[:,"output"],
                      3*X_test.query("ylabel==0").loc[:,"eventweight"],
                      X_test.query("ylabel==1 & channelNumber==392323").loc[:,"output"],
                      3*X_test.query("ylabel==1 & channelNumber==392323").loc[:,"eventweight"])#,
                      #figure_text='10~fb$^{-1}$')
  plt.savefig('hist5_test_392323_C1N2_WZ_2L2J_500_0_weighted_corrected.pdf')


  # Plot feature importance
  print("model.feature_importances_", model.feature_importances_)
  print("np.sum(model.feature_importances_)", np.sum(model.feature_importances_))
  s_feat_importance = pd.Series(model.feature_importances_, index=X_train.drop(["channelNumber", "eventweight", "ylabel", "output"], axis=1).columns)
  print("X_train.drop(['eventweight', 'ylabel'], axis=1).columns\n", X_train.drop(["channelNumber", "eventweight", "ylabel", "output"], axis=1).columns)

  plt.figure(6)
  sns.set(style="ticks", color_codes=True)
  ax = sns.barplot(x=s_feat_importance*100, y=s_feat_importance.index)#, palette="Blues_r")
  #ax.set_yticklabels(s_feat_importance.index)
  ax.set(xlabel="Feature importance [%]")
  plt.savefig('feature_importance.pdf')


  # Plot ROC curve
  plt.figure(7)
  fpr, tpr, thresholds = metrics.roc_curve(X_test.loc[:,"ylabel"], X_test.loc[:,"output"])
  auc = metrics.roc_auc_score(X_test.loc[:,"ylabel"], X_test.loc[:,"output"])
  ax = sns.lineplot(x=fpr, y=tpr, estimator=None, label='ROC curve: AUC = %0.2f' % auc)
  plt.plot([0,1], [0,1], linestyle="--")
  ax.set(xlabel="False positive rate", ylabel="True positive rate")
  plt.savefig('ROC_curve_AUC.pdf')

  plt.show()


  # Signal significance
  print("\n///////////////// Signal significance /////////////////")

  def significance(cut_string_sig, cut_string_bkg, rel_unc=0.3):
    sig_exp = np.sum(3*X_test.query("ylabel == 1 & "+cut_string_sig).loc[:,"eventweight"])
    bkg_exp = np.sum(3*X_test.query("ylabel == 0 & "+cut_string_bkg).loc[:,"eventweight"])
    Z_N_exp = RooStats.NumberCountingUtils.BinomialExpZ(sig_exp, bkg_exp, rel_unc)
    return [sig_exp, bkg_exp, Z_N_exp]

  DSID = {'(200, 100)':392330, '(300, 100)':392304, '(500, 0)':392323}
  cut = 0.5
  
  for key, dsid in DSID.items():
    cut_string_SR = 'output > {:f}'.format(cut)
    cut_string_DSID = 'channelNumber == {0:d}'.format(dsid)
    cut_string_bkg = cut_string_SR
    cut_string_sig = cut_string_SR + " & " + cut_string_DSID
    print('\ncut_string_sig:', cut_string_sig)
    print('cut_string_bkg:', cut_string_bkg)

    [sig_exp, bkg_exp, Z_N_exp] = significance(cut_string_sig, cut_string_bkg, rel_unc=0.3)
    print("---", key)
    print("S_exp =", sig_exp)
    print("B_exp =", bkg_exp)
    print("Z_N_exp =", Z_N_exp)

  # Stop timer
  t_end = time.time()
  print("\nProcess time: {:4.2f} s".format(t_end - t_start))


if __name__ == "__main__":
  main()
