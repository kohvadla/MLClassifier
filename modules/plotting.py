import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from modules.samples import l_bkg

# Set seaborn style
sns.set(color_codes=True)
colpal = sns.color_palette()
c_blue = colpal[0]
c_orange = colpal[1]
c_green = colpal[2]
c_red = colpal[3]

# Color Brewer palettes (reversed, i.e. from dark to light)
red_pal = sns.color_palette('Reds_r', n_colors=3)
orange_pal = sns.color_palette('Oranges_r', n_colors=3)
green_pal = sns.color_palette('Greens_r', n_colors=3)
blue_pal = sns.color_palette('Blues_r', n_colors=3)
purple_pal = sns.color_palette('Purples_r', n_colors=3)
grey_pal = sns.color_palette('Greys_r', n_colors=3)
RdPu_pal = sns.color_palette('RdPu_r', n_colors=3)

# Map sample groups to a color
d_bkg_col = {'ttbar':     blue_pal[0],
             'singleTop': blue_pal[1],
             'topOther':  blue_pal[2],
             'lowMassDY': green_pal[1],
             'Zjets':     green_pal[0],
             'Wjets':     green_pal[2],
             'diboson':   orange_pal[1],
             'triboson':  orange_pal[2],
             'higgs':     RdPu_pal[0],
             }

# List of bkg colors ordered according to the order in l_bkg
l_bkg_col = [d_bkg_col[bkg] for bkg in l_bkg]


def plotTrainTestOutput(d_X_train_bkg, d_ew_train_bkg, X_train_sig, ew_train_sig, d_X_test_bkg, d_ew_test_bkg, X_test_sig, ew_test_sig, signal_region):
  
  n_classes = d_X_train_bkg['Zjets'].shape[1]

  if n_classes is 1:
    l_X_train_bkg_arrs = [d_X_train_bkg[i_bkg]['output'].squeeze() for i_bkg in l_bkg]
    l_X_test_bkg_arrs = [d_X_test_bkg[i_bkg]['output'].squeeze() for i_bkg in l_bkg]
  else:
    l_X_train_bkg_arrs = [[d_X_train_bkg[i_bkg]['output'+str(i_class)] for i_bkg in l_bkg] for i_class in range(n_classes)]
    l_X_test_bkg_arrs = [[d_X_test_bkg[i_bkg]['output'+str(i_class)] for i_bkg in l_bkg] for i_class in range(n_classes)]

  if d_ew_train_bkg is not None:
    l_ew_train_bkg_arrs = [d_ew_train_bkg[i_bkg] for i_bkg in l_bkg]
  else:
   l_ew_train_bkg_arrs = None
  if d_ew_test_bkg is not None:
    l_ew_test_bkg_arrs = [d_ew_test_bkg[i_bkg] for i_bkg in l_bkg]
  else:
    l_ew_test_bkg_arrs = None

  signal_group_label = 'C1N2 '+ signal_region
  if 'low' in signal_region:
    signal_label = '(200, 100) GeV'
  elif 'int' in signal_region:
    signal_label = '(500, 200) GeV'
  elif 'high' in signal_region:
    signal_label = '(600, 0) GeV'

  bins = np.linspace(0., 1., 20)
  l_classes = ['Signal', 'Zjets', 'Diboson']

  fig, ax = plt.subplots(n_classes, 1, figsize=[6.4, 4.8*n_classes])
  for i_class in range(n_classes):

    if n_classes > 1:
      ax = ax[i_class]
      
      l_X_train_bkg_arrs = l_X_train_bkg_arrs[i_class]
      l_X_test_bkg_arrs = l_X_test_bkg_arrs[i_class]
      X_train_sig = X_train_sig['output'+str(i_class)]
      X_test_sig = X_test_sig['output'+str(i_class)]

      ax.set_xlabel(l_classes[i_class]+' probability')
    else:
      ax.set_xlabel('Signal probability')

    ax.set_yscale('symlog')
    ax.set_ylim([0, 1E7])
    ax.set_ylabel('Entries')

    n_train_bkg = sum(X_train_bkg.size for X_train_bkg in l_X_train_bkg_arrs)
    n_test_bkg = sum(X_test_bkg.size for X_test_bkg in l_X_test_bkg_arrs)
    n_train_sig = len(X_train_sig)
    n_test_sig = len(X_test_sig)
    sf_train_bkg = n_test_bkg/n_train_bkg
    sf_train_sig = n_test_sig/n_train_sig

    ax.hist(l_X_train_bkg_arrs,
             bins=bins,
             density=False,
             weights=[sf_train_bkg*i_ew_bkg_arr for i_ew_bkg_arr in l_ew_train_bkg_arrs],
             histtype='stepfilled',
             #color=colpal[:len(l_bkg)][::-1],
             color=l_bkg_col,
             alpha=0.75,
             label=l_bkg,
             stacked=True,
             linewidth=2)

    ax.hist(X_train_sig.squeeze(),
             bins=bins,
             density=False,
             weights=sf_train_sig*ew_train_sig,
             histtype='step',
             color=c_red,
             alpha=0.75,
             label=signal_group_label,
             stacked=False,
             linewidth=2,
             #hatch='//'
             )

    ax.hist(l_X_test_bkg_arrs,
             bins=bins,
             density=False,
             weights=l_ew_test_bkg_arrs,
             histtype='step',
             #color=colpal[:len(l_bkg)][::-1],
             color=l_bkg_col,
             alpha=1,
             label=l_bkg,
             stacked=True,
             linewidth=3,
             linestyle='--')

    ax.hist(X_test_sig.squeeze(),
             bins=bins,
             density=False,
             weights=ew_test_sig,
             histtype='step',
             color=c_red,
             alpha=1,
             label=signal_label,
             stacked=False,
             linewidth=3,
             linestyle='--')

    handles, labels = plt.gca().get_legend_handles_labels()
    leg_train = mpatches.Patch(color='black', linestyle='solid', fill=True, alpha=0.75, label='train')
    leg_test = mpatches.Patch(color='black', linestyle='dashed', fill=False, alpha=0.75, label='test')

    leg1 = ax.legend(
                     handles=[handles[0], handles[1], handles[2], handles[3], handles[4], handles[5], handles[6], handles[7], handles[8], handles[9]], 
                     labels=[labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6], labels[7], labels[8], labels[9]], 
                     ncol=1,
                     loc='upper left',
                     frameon=False,
                     #title='Train vs. test',
                     bbox_to_anchor=(1.05, 0.8),
                     borderaxespad=0.
                     )

    leg2 = ax.legend(
                     handles=[leg_train, leg_test], 
                     labels=["train", "test"],
                     ncol=1,
                     loc='upper left',
                     frameon=False,
                     #title='Train vs. test',
                     bbox_to_anchor=(1.05, 1),
                     borderaxespad=0.
                     )

    ax.add_artist(leg1)

    #plt.rc('text', usetex=True)
    #props = dict('')
    ax.text(0.2, 0.85, '13 TeV, 70/fb\nTraining vs. test scores', transform=ax.transAxes)#, bbox=props)

  return



def plotFinalTestOutput(d_X_test_bkg, d_ew_test_bkg, X_test_sig, ew_test_sig, signal_region, figure_text=''):

  n_classes = d_X_test_bkg['Zjets'].shape[1]

  if n_classes is 1:
    l_X_test_bkg_arrs = [d_X_test_bkg[i_bkg].output for i_bkg in l_bkg]
  else:
    l_X_test_bkg_arrs = [[d_X_test_bkg[i_bkg]['output'+str(i_class)] for i_bkg in l_bkg] for i_class in range(n_classes)]

  if d_ew_test_bkg is not None:
    l_ew_test_bkg_arrs = [d_ew_test_bkg[i_bkg] for i_bkg in l_bkg]
  else:
    l_ew_test_bkg_arrs = None

  signal_group_label = 'C1N2 '+ signal_region
  if 'low' in signal_region:
    signal_label = '(200, 100) GeV'
  elif 'int' in signal_region:
    signal_label = '(500, 200) GeV'
  elif 'high' in signal_region:
    signal_label = '(600, 0) GeV'

  bins = np.linspace(0., 1., 100)
  l_classes = ['Signal', 'Zjets', 'Diboson']

  fig, ax = plt.subplots(n_classes, 1, figsize=[6.4, 4.8*n_classes])
  for i_class in range(n_classes):

    if n_classes > 1:
      ax = ax[i_class]
      
      l_X_test_bkg_arrs = l_X_test_bkg_arrs[i_class]
      X_test_sig = X_test_sig['output'+str(i_class)]

      ax.set_xlabel(l_classes[i_class]+' probability')
    else:
      ax.set_xlabel('Signal probability')

    ax.set_yscale('symlog')
    ax.set_ylim([0, 1E7])
    ax.set_ylabel('Entries')

    ax.hist(l_X_test_bkg_arrs,
             bins=bins,
             weights=l_ew_test_bkg_arrs,
             histtype='stepfilled',
             #color=colpal[:len(l_bkg)][::-1],
             color=l_bkg_col,
             alpha=0.75,
             label=l_bkg,
             stacked=True,
             linewidth=2,
             linestyle='-')

    ax.hist(X_test_sig.squeeze(),
             bins=bins,
             weights=ew_test_sig,
             histtype='step',
             color=c_red,
             alpha=1,
             label=signal_label,
             stacked=False,
             linewidth=2,
             linestyle='-')

    #handles, labels = plt.gca().get_legend_handles_labels()
    #leg_train = mpatches.Patch(color='black', linestyle='solid', fill=True, alpha=0.75, label='train')
    #leg_test = mpatches.Patch(color='black', linestyle='dashed', fill=False, alpha=0.75, label='test')
    ax.legend(
              #handles=[leg_train, leg_test, handles[9], handles[8], handles[7], handles[6], handles[5], handles[4], handles[3], handles[2], handles[1], handles[0]], 
              #labels=["train", "test", labels[9], labels[8], labels[7], labels[6], labels[5], labels[4], labels[3], labels[2], labels[1], labels[0]], 
              ncol=1,
              loc='upper left',
              frameon=False,
              #title='Test set',
              bbox_to_anchor=(1.05, 1),
              borderaxespad=0.)

    #plt.rc('text', usetex=True)
    ax.text(0.2, 0.85, '13 TeV, 70/fb\nTest scores', transform=ax.transAxes)

  return
