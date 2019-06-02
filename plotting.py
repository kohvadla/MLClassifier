import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plotTrainTestOutput(X_train_bkg, ew_train_bkg, X_train_sig, ew_train_sig, X_test_bkg, ew_test_bkg, X_test_sig, ew_test_sig):
  ax = sns.distplot(X_train_bkg, #X_train.query("ylabel==0").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_train_bkg #X_train.query("ylabel==0").loc[:,"eventweight"]
                              }, 
                    color="tab:blue",
                    label="bkg")
  sns.distplot(X_train_sig, #X_train.query("ylabel==1").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_train_sig #X_train.query("ylabel==1").loc[:,"eventweight"]
                              }, 
                    color="tab:orange",
                    label="sig", 
                    ax=ax)
  sns.distplot(X_test_bkg, #X_test.query("ylabel==0").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"step", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"dashed", 
                              "linewidth":2, 
                              'alpha':1, 
                              "weights":ew_test_bkg #X_test.query("ylabel==0").loc[:,"eventweight"]
                              }, 
                    color="tab:blue",
                    #label="bkg test", 
                    ax=ax)
  sns.distplot(X_test_sig, #X_test.query("ylabel==1").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"step", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"dashed", 
                              "linewidth":2, 
                              'alpha':1, 
                              "weights":ew_test_sig #X_test.query("ylabel==1").loc[:,"eventweight"]
                              }, 
                    color="tab:orange",
                    #label="sig test", 
                    ax=ax)
  ax.set(xlabel="XGBoost output", ylabel="Entries", yscale="log")

  handles, labels = plt.gca().get_legend_handles_labels()
  leg_train = mpatches.Patch(color='black', linestyle='solid', fill=True, alpha=0.5, label='train')
  leg_test = mpatches.Patch(color='black', linestyle='dashed', fill=False, alpha=0.5, label='test')
  plt.legend(handles=[leg_train, leg_test, handles[0], handles[1]], labels=["train", "test", labels[0], labels[1]], ncol=2)

  return



def plotFinalTestOutput(X_test_bkg, ew_test_bkg, X_test_sig, ew_test_sig, figure_text=''):
  ax = sns.distplot(X_test_bkg, #X_test.query("ylabel==0").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_test_bkg #X_test.query("ylabel==0").loc[:,"eventweight"]
                              }, 
                    color="tab:blue",
                    label="bkg")
  sns.distplot(X_test_sig, #X_test.query("ylabel==1").loc[:,"output"], 
                    hist=True, 
                    bins=20, 
                    kde=False, 
                    rug=False, 
                    hist_kws={"histtype":"bar", 
                              "range":(0,1), 
                              "density":False, 
                              "linestyle":"solid", 
                              "linewidth":2, 
                              "weights":ew_test_sig #X_test.query("ylabel==1").loc[:,"eventweight"]
                              }, 
                    color="tab:orange",
                    label="sig", 
                    ax=ax)
  ax.set(xlabel="XGBoost output", ylabel="Entries", yscale="log")
  #if figure_text:
  #  plt.rc('text', usetex='True')
  #  plt.text(0.5, 0.9, figure_text)
  plt.legend()

  return
