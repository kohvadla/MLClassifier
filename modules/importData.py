import os

import h5py
import uproot
import numpy as np
import pandas as pd
from ROOT import TLorentzVector
from sklearn.utils import shuffle

from OpenData.infofile import infos
from modules.features import l_features_2L2J, l_features_2L3Jplus, l_eventweights
#from hepFunctions import invariantMass



def importDatasetFromHDF5(filepath, dataset_name):
  """Read hdf5 input file"""

  with h5py.File(filepath, 'r') as hf:
    ds = hf[dataset_name]
    df = np.DataFrame(data=ds[:])

  return df


def importDatasetFromROOTFile(filepath, treename, entry_start=None, entry_stop=None, flatten_bool=None, branches="*"):
  """Import TTree with jagged arrays from ROOT file using uproot"""

  tree = uproot.open(filepath)[treename]
  df = tree.pandas.df(branches, entrystart=entry_start, entrystop=entry_stop, flatten=flatten_bool)

  if "OpenData" in filepath:
    if "jet" in branches:
      df = df.loc[:,"jet_pt":].copy()
    elif "lep" in branches:
      df = df.loc[:,"lep_pt":].copy()

  elif "histfitter" in filepath:
    if "lep" in branches:
      df = df.loc[:,:"lepM"].copy()

  return df


def applyCuts(df, cuts):
  """Apply cuts to events/rows in the dataset"""

  df = df.query("&".join(cuts))

  return df


def getEventWeights(df, l_eventweights):
  """Return pandas Dataframe with a single combined eventweight calculated by
  multiplying the columns in the DataFrame that contain different eventweights,
  and normalizing to the integrated luminosity of the simulated samples"""

  #df['targetLumi'] = np.where(df.RandomRunNumber < 320000, 36200, np.where(df.RandomRunNumber > 320000 and df.RandomRunNumber < 348000, 44300, 59900))
  df['targetLumi'] = pd.cut(df.RandomRunNumber, [0, 320000,  348000, np.inf], labels=[36200, 44300, 59900])
  print("df.loc[:,l_eventweights]\n", df.loc[:,l_eventweights].head())

  # Multiply all eventweights except for xsec and sumw
  s_eventweight = df.loc[:,l_eventweights].agg('prod', axis='columns').copy()
  print("\n- s_eventweight.shape =\n", s_eventweight.shape)

  # Normalize to integrated luminosity: N = L*xsec =>  1/L = xsec/N
  #Lumi16_periodAtoD = 10.6  # [fb-1] int lumi in data16 period A-D
  #xsec_pb_to_fb = 1e3  # convert xsec from pb to fb
  #s_eventweight *= lumi * xsec_pb_to_fb
  s_eventweight = s_eventweight[s_eventweight > 0]  # remove events with negative eventweights
  print("\nNumber of zeros or NaNs:", df.isna().sum().sum())
  print("\n- After dropping events/rows where eventweight is 0:\ns_eventweight.shape", s_eventweight.shape)
  print("\nNumber of zeros or NaNs:", df.isna().sum().sum())

  return s_eventweight


def selectFeatures(df, l_column_names):
  """Select subset of variables/features in the df"""

  df_features = df.loc[:,l_column_names].copy()
  print("\n- After selecting features:\ndf_features.shape", df_features.shape)

  return df_features


def dropVariables(df, l_drop_columns):
  """Drop subset of variables in the df"""

  df_new = df.drop(columns=l_drop_columns, inplace=True)

  return df_new


def importDataset(sample_type, sample_name, mc16_campaign, selection, entrystart=None, entrystop=None):
  """Import ntuples"""

  path = '/eos/user/k/kvadla/ntuples/histfitter/skim_slim_v1.6_atLeast2L2J_ext/'
  #data_path = path + "Data/data17.periodA-D.root"

  if sample_type is 'sig':
    filepath = path + 'SUSY2_Signal_' + mc16_campaign + '/' + sample_name + '_merged_processed.root'
  elif sample_type is 'bkg':
    filepath = path + 'SUSY2_Bkgs_' + mc16_campaign + '/' + sample_name + '_merged_processed.root'
  elif sample_type is 'data':
    filepath = data_path

  if sample_type is not 'data':  # if sig or bkg
    treename = sample_name + '_NoSys'
  else:
    treename = sample_name


  print("\n============================================")
  print("\nSample: {0} {1}".format(sample_name, mc16_campaign))
  print("\n============================================")
  print("\nSample path:",filepath)
  df_flat = importDatasetFromROOTFile(filepath, treename, entrystart, entrystop)
  global n_events_read
  n_events_read += len(df_flat)
  print("\n- After importing flat DF:\ndf_flat.shape", df_flat.shape)
  global n_events_chunk
  n_events_chunk = len(df_flat)
  print("\nIn importData(): n_events_chunk =", n_events_chunk)


  # Preselection
  l_cuts_presel = ['trigMatch_2LTrigOR', 'nLep_base == 2', 'nLep_signal == 2', 'nBJet20_MV2c10_FixedCutBEff_77 == 0', '81 < mll < 101']
  if '2L2J' in selection:
    l_cuts_presel += ['nJet30 == 2', '60 < mjj < 100']
  elif '2L3J+' in selection:
    l_cuts_presel += ['nJet30 >= 3', '60 < mjj_minDPhiZMET < 100']
  df_flat = applyCuts(df_flat, l_cuts_presel)

  print("\n- Preselection cuts applied:")
  for i_cut in l_cuts_presel:
    print(i_cut)
  print("\n- After cuts:\ndf_flat.shape", df_flat.shape)

  print("\n- After importing flat DF:\ndf_flat.shape", df_flat.shape)

  print("\n---------------------------------------")
  df_jet = importDatasetFromROOTFile(filepath, treename, entrystart, entrystop, flatten_bool=True, branches="jet*")  # fetching all branches with names starting with jet*
  df_jet = df_jet.unstack()
  print("\n- After importing jet DF:\ndf_flat.shape", df_jet.shape)
  df_jet = df_jet.stack()

  l_cuts_jet = ['jetPt > 30']
  df_jet = df_jet.astype({"jetEta":float})
  df_jet = applyCuts(df_jet, l_cuts_jet)
  
  n_jet_sig = df_jet.groupby(level=0).size().values
  df_jet = df_jet.unstack()
  if '2L2J' in selection:
    df_jet = df_jet[n_jet_sig == 2].copy()
    df_jet = df_jet.stack().unstack()
  elif '2L3J+' in selection:
    df_jet = df_jet[n_jet_sig >= 3].copy()
    df_jet.dropna(axis='columns', inplace=True)

  #df_jet = df_jet.stack()
  #df_jet["subindex"] = df_jet.groupby(level=0).apply(lambda x : x.sort_values("jetPt", ascending=False)).groupby(level=0).cumcount()
  #df_jet = df_jet.set_index("subindex", append=True)
  #df_jet = df_jet.reset_index(level="subentry").drop("subentry", axis=1)
  #df_jet = df_jet.unstack(level="subindex")

  df_jet.columns = [col[0] + str(col[1]+1) for col in df_jet.columns.values]

  #df_jet['mjj'] = df_jet.apply(lambda x: invariantMass([x.jet_pt1, x.jet_eta1, x.jet_phi1, x.jet_E1], [x.jet_pt2, x.jet_eta2, x.jet_phi2, x.jet_E2]), axis=1)

  #df_jet = applyCuts(df_jet, l_cuts_WonShell)

  print("\n- Jet cuts applied:")
  for i_cut in l_cuts_jet:
    print(i_cut)
  print("\n- After cuts:\ndf_flat.shape", df_jet.shape)
  
  df_flat = pd.concat([df_flat, df_jet], axis=1, sort=False)
  print("\n- After concatenating jet DF with flat DF:\ndf_flat.shape", df_flat.shape)

  print("\n---------------------------------------")
  df_lep = importDatasetFromROOTFile(filepath, treename, entrystart, entrystop, flatten_bool=True, branches='lep*')  # fetch all branches with names starting with 'lep*'
  df_lep = df_lep.loc[:,:'lepM']  # keep only the branches up until 'lepM'
  df_lep = df_lep.unstack()
  print("\n- After importing lep DF:\ndf_flat.shape", df_lep.shape)
  df_lep = df_lep.stack()

  # 2 leading leptons: el/mu pT > 25 GeV
  l_cuts_lep = ['lepPt > 25']
  df_lep = df_lep.astype({"lepEta":float})
  df_lep = applyCuts(df_lep, l_cuts_lep)

  n_lep_sig = df_lep.groupby(level=0).size().values
  df_lep = df_lep.unstack()
  df_lep = df_lep[n_lep_sig == 2].copy()
  df_lep = df_lep.stack().unstack()  # remove all-NaN columns

  #df_lep = df_lep.stack()
  #df_lep_sorted = df_lep.groupby(level=0).apply(lambda x : x.sort_values("lepPt", ascending=False))
  #df_lep_sorted = df_lep_sorted.reset_index(level=0).drop("entry", axis=1)
  #df_lep["subindex"] = df_lep_sorted.groupby(level=0).cumcount()
  #df_lep = df_lep.set_index("subindex", append=True)
  #df_lep = df_lep.reset_index(level="subentry").drop("subentry", axis=1)
  #df_lep = df_lep.unstack(level="subindex")
  
  df_lep.columns = [col[0] + str(col[1]+1) for col in df_lep.columns.values]

  #df_lep['mll'] = df_lep.apply(lambda x: invariantMass([x.lep_pt1, x.lep_eta1, x.lep_phi1, x.lep_E1], [x.lep_pt2, x.lep_eta2, x.lep_phi2, x.lep_E2]), axis=1)

  l_cuts_SFOS = ["lepFlavor1 == lepFlavor2", "lepCharge1 != lepCharge2"]
  df_lep = applyCuts(df_lep, l_cuts_SFOS)

  print("\n- Lepton cuts applied:")
  for i_cut in l_cuts_lep + l_cuts_SFOS:
    print(i_cut)
  print("\n- After cuts:\ndf_flat.shape", df_lep.shape)
  
  df_flat = pd.concat([df_flat, df_lep], axis=1, sort=False)
  print("\n- After concatenating lep DF with flat DF:\ndf_flat.shape", df_flat.shape)

  df_flat.dropna(axis='index', inplace=True)
  print("\n- After dropping events/rows with NaNs:\ndf_flat.shape", df_flat.shape)

  return df_flat


def prepareInput(store, sample_name, mc16_campaign, sample_type, selection, chunk_size=1e4, n_chunks=100, entrystart=0):
  """Read in dataset in chunks, preprocess for ML and store to HDF5"""

  print("\nPrepare input")
  
  #for i_chunk in range(1, n_chunks+1):

  global n_events_chunk
  n_events_chunk = chunk_size
  i_chunk = 1
  global n_events_read
  n_events_read = 0
  n_events_kept = 0
  global h5_group
  h5_group = sample_type + "/" + sample_name

  while n_events_chunk == chunk_size:

    print("\nReading chunk #{:d}".format(i_chunk))
    entrystop = i_chunk*chunk_size  # entrystop exclusive

    df = importDataset(sample_type, sample_name, mc16_campaign, selection, entrystart, entrystop)
    #df = shuffle(df)  # shuffle the rows/events
    
    if '2L2J' in selection:
      df_feat = selectFeatures(df, l_features_2L2J)
    elif '2L3J+' in selection:
      df_feat = selectFeatures(df, l_features_2L3Jplus)
    df_feat = df_feat*1  # multiplying by 1 to convert booleans to integers
    
    if sample_type is not 'data':
      df_feat["eventweight"] = getEventWeights(df, l_eventweights)
      print("\n- After adding calculating and adding eventweight column:\ndf_feat.shape", df_feat.shape)

      df_feat = df_feat[df_feat.eventweight > 0]
      print("\n- After removing events/rows with eventweight <= 0:\ndf_feat.shape", df_feat.shape)

    print("\ndf.head()\n", df_feat.head())

    n_events_kept += len(df)

    print("\ni_chunk", i_chunk)
    print("h5_group", h5_group)
    if i_chunk is 1 and len(store.keys()) is 0:
      store.put(h5_group, df_feat, format='table')
      #print("\nStored initial chunk in HDF5 file to", h5_group)
      print("\nAppended chunk #{0:d} in HDF5 file to {1}".format(i_chunk, h5_group))
    else:
      store.append(h5_group, df_feat, format='table')
      print("\nAppended chunk #{0:d} in HDF5 file to {1}".format(i_chunk, h5_group))

    print("\nIn prepareInput(): n_events_chunk =", n_events_chunk)

    #if n_events_chunk < chunk_size:
    #  print("\nReached end of dataset --> do not try to read in another chunk")
    #  break

    # If n_chunks is set to None, the following if test will never evaluate to True,
    # and all events will be read in
    if i_chunk == n_chunks:
      print("\nReached the number of requested chunks: {0:d}, for sample type:".format(i_chunk, sample_type))
      break

    entrystart = entrystop  # entrystart inclusive
    i_chunk +=1

  print("\n************************************************")
  print("*** {0} {1} ***".format(sample_name, mc16_campaign))
  print("# events read:", n_events_read)
  print("# events kept:", n_events_kept)
  print("************************************************")

  # Count number of events read and kept per sample across all mc16 campaigns
  global n_events_read_sample, n_events_kept_sample
  n_events_read_sample += n_events_read
  n_events_kept_sample += n_events_kept

  # Count number of events read and kept per sample type, i.e. per sig, bkg, data
  global n_events_read_sample_type, n_events_kept_sample_type
  n_events_read_sample_type += n_events_read
  n_events_kept_sample_type += n_events_kept

  return store


def prepareHDF5(filename, l_samples, sample_type=None, selection=None, chunk_size=1e5, n_chunks=None, entrystart=0):
  """Read input dataset in chunks, select features and perform cuts,
  before storing DataFrame in HDF5 file"""

  exists = os.path.isfile(filename)
  if exists:
    os.remove(filename)
    print("\nRemoved existing file", filename)
  store = pd.HDFStore(filename)
  print("\nCreated new store with name", filename)

  global n_events_read_sample_type, n_events_kept_sample_type
  n_events_read_sample_type = 0
  n_events_kept_sample_type = 0

  for sample in l_samples:
    if sample_type is 'data':
      store = prepareInput(store, sample, None, sample_type, selection, chunk_size, n_chunks, entrystart)
    else:
      global n_events_read_sample, n_events_kept_sample
      n_events_read_sample = 0
      n_events_kept_sample = 0

      for mc16_campaign in ['mc16a', 'mc16cd', 'mc16e']:
        store = prepareInput(store, sample, mc16_campaign, sample_type, selection, chunk_size, n_chunks, entrystart)
        print("\n################################################")
        print("*** {0} ***".format(sample))
        print("# events read across mc16a+d+e:", n_events_read_sample)
        print("# events kept across mc16a+d+e:", n_events_kept_sample)
        print("################################################")

  print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
  print("*** {0} ***".format(sample_type))
  print("# events read in total:", n_events_read_sample_type)
  print("# events kept in total:", n_events_kept_sample_type)
  print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

  print("\nReturned from prepareInput()")
  print("\nstore:\n", store)
  print("\nstore.keys()", store.keys())

  store.close()
  print("\nClosed store")

  return


def invariantMass(l_4vector1, l_4vector2):
  """Calculate the invariant mass of two TLorentzVectors"""

  tlv1 = TLorentzVector()
  tlv2 = TLorentzVector()
  tlv1.SetPtEtaPhiE(l_4vector1[0], l_4vector1[1], l_4vector1[2], l_4vector1[3])
  tlv2.SetPtEtaPhiE(l_4vector2[0], l_4vector2[1], l_4vector2[2], l_4vector2[3])

  return (tlv1 + tlv2).M()


def importOpenDataInfo(df):
  """Extract info about cross section and sum of weights from infofile"""

  # Add three new, empty columns to the input DataFrame
  df["index"] = df.index
  df = df.set_index(["channelNumber"])
  df = pd.concat([df, pd.DataFrame(columns=["xsec_fb", "sumw", "1_over_sumw", "events"])], axis=1)
  #print("df.shape", df.shape)

  for key in infos.keys(): 
    if infos[key]["DSID"] in df.index:
      print("\n- Fillng info for channelNumber", infos[key]["DSID"])
      print("xsec: {0:f}, sumw: {1:f}, events: {2:f}".format(infos[key]["xsec"], infos[key]['sumw'], infos[key]['events'])), 
      df.loc[infos[key]["DSID"], ["xsec_fb", "sumw", "1_over_sumw", "events"]] = [infos[key]['xsec']*1e3, infos[key]['sumw'], 1/(infos[key]['sumw']), infos[key]['events']]

  df.insert(2, "channelNumber", df.index)
  df = df.set_index("index")
  print("\nPrinting the new info columns for the five first events:\n", df[["xsec_fb", "sumw", "1_over_sumw", "events"]].head())

  return df


def importOpenData(sample_type, entrystart=None, entrystop=None):
  """Import OpenData ntuples"""

  path = "/eos/user/k/kvadla/OpenData/2lep/"
  sig_path = path + "MC/BSM_Signal_Samples/mc15.MGPy8EG.C1N2_WZ_2L2J.root"
  #bkg_path = path + "MC/SM_Backgrounds/mc15.allBackgrounds.root"
  bkg_path = path + "MC/SM_Backgrounds/mc15.PwPy8EG.diboson.root"
  #bkg_path = path + "MC/SM_Backgrounds/mc15.PwPy8EG.WZqqll.root"
  data_path = path + "Data/data16.periodA-D.root"

  if sample_type is "sig":
    filepath = sig_path
  elif sample_type is "bkg":
    filepath = bkg_path
  elif sample_type is "data":
    filepath = data_path

  treename = "mini"

  print("\n=======================================")
  print("\n sample_type =", sample_type)
  print("\n=======================================")
  print("\nSample path:",filepath)
  df_flat = importDatasetFromROOTFile(filepath, treename, entrystart, entrystop)
  print("\n- After importing flat DF:\ndf_flat.shape", df_flat.shape)
  global n_events_chunk
  n_events_chunk = len(df_flat)
  print("\nIn importData(): n_events_chunk =", n_events_chunk)


  # Preselection: Trigger + >= 2 lep + >= 2 jets
  l_cuts_presel = ["trigE | trigM", "lep_n >= 2", "jet_n >= 2"]
  df_flat = applyCuts(df_flat, l_cuts_presel)

  print("\n- Preselection cuts applied:")
  for i_cut in l_cuts_presel:
    print(i_cut)
  print("\n- After cuts:\ndf_flat.shape", df_flat.shape)

  print("\n- After importing flat DF:\ndf_flat.shape", df_flat.shape)

  print("\n---------------------------------------")
  df_jet = importDatasetFromROOTFile(filepath, treename, entrystart, entrystop, flatten_bool=True, branches="/jet_(?!tru).*/i")  # using regex to match all brachnames starting with 'jet_', except those that continue with 'tru', since these branches are not properly filled for the data ntuples
  df_jet = df_jet.unstack()
  print("\n- After importing jet DF:\ndf_flat.shape", df_jet.shape)
  df_jet = df_jet.stack()

  l_cuts_jet = ["jet_pt > 30e3", "abs(jet_eta) < 2.8"]
  l_cuts_bjet = ["jet_MV2c10 < 0.6459"] # 77% WP: 0.6459; 85% WP: 0.1758
  df_jet = df_jet.astype({"jet_eta":float})
  df_jet = applyCuts(df_jet, l_cuts_jet + l_cuts_bjet)
  
  n_jet_sig = df_jet.groupby(level=0).size().values
  df_jet = df_jet.unstack()
  df_jet = df_jet[n_jet_sig == 2].copy()

  df_jet = df_jet.stack()
  df_jet["subindex"] = df_jet.groupby(level=0).apply(lambda x : x.sort_values("jet_pt", ascending=False)).groupby(level=0).cumcount()
  df_jet = df_jet.set_index("subindex", append=True)
  df_jet = df_jet.reset_index(level="subentry").drop("subentry", axis=1)
  df_jet = df_jet.unstack(level="subindex")

  df_jet.columns = [ col[0] + str(col[1]+1) for col in df_jet.columns.values]

  df_jet['mjj'] = df_jet.apply(lambda x: invariantMass([x.jet_pt1, x.jet_eta1, x.jet_phi1, x.jet_E1], [x.jet_pt2, x.jet_eta2, x.jet_phi2, x.jet_E2]), axis=1)

  l_cuts_WonShell = ['60e3 < mjj < 100e3']
  df_jet = applyCuts(df_jet, l_cuts_WonShell)

  print("\n- Jet cuts applied:")
  for i_cut in l_cuts_jet + l_cuts_bjet + ["n_jet_sig == 2"] + l_cuts_WonShell:
    print(i_cut)
  print("\n- After cuts:\ndf_flat.shape", df_jet.shape)
  
  df_flat = pd.concat([df_flat, df_jet], axis=1, sort=False)
  print("\n- After concatenating jet DF with flat DF:\ndf_flat.shape", df_flat.shape)

  print("\n---------------------------------------")
  df_lep = importDatasetFromROOTFile(filepath, treename, entrystart, entrystop, flatten_bool=True, branches="/lep_(?!tru).*/i")  # using regex to match all brachnames starting with 'lep_', except those that continue with 'tru', since these branches are not properly filled for the data ntuples
  df_lep = df_lep.unstack()
  print("\n- After importing lep DF:\ndf_flat.shape", df_lep.shape)
  df_lep = df_lep.stack()

  # 2 leading leptons: el/mu pT > 25 GeV, el |eta| < 2.47, mu |eta| < 2.7 
  l_cuts_lep = ["lep_pt > 25e3",
                "((lep_type == 11) & (abs(lep_eta) < 2.47)) | ((lep_type == 13) & (abs(lep_eta) < 2.4))"]
  df_lep = df_lep.astype({"lep_eta":float})
  df_lep = applyCuts(df_lep, l_cuts_lep)

  n_lep_sig = df_lep.groupby(level=0).size().values
  df_lep = df_lep.unstack()
  df_lep = df_lep[n_lep_sig == 2].copy()

  df_lep = df_lep.stack()
  df_lep_sorted = df_lep.groupby(level=0).apply(lambda x : x.sort_values("lep_pt", ascending=False))
  df_lep_sorted = df_lep_sorted.reset_index(level=0).drop("entry", axis=1)
  df_lep["subindex"] = df_lep_sorted.groupby(level=0).cumcount()
  df_lep = df_lep.set_index("subindex", append=True)
  df_lep = df_lep.reset_index(level="subentry").drop("subentry", axis=1)
  df_lep = df_lep.unstack(level="subindex")
  
  df_lep.columns = [col[0]+str(col[1]+1) for col in df_lep.columns.values]

  df_lep['mll'] = df_lep.apply(lambda x: invariantMass([x.lep_pt1, x.lep_eta1, x.lep_phi1, x.lep_E1], [x.lep_pt2, x.lep_eta2, x.lep_phi2, x.lep_E2]), axis=1)

  l_cuts_SFOS = ["lep_type1 == lep_type2", "lep_charge1 != lep_charge2"]
  l_cuts_ZonShell = ['81e3 < mll < 101e3']
  df_lep = applyCuts(df_lep, l_cuts_SFOS + l_cuts_ZonShell)

  print("\n- Lepton cuts applied:")
  for i_cut in l_cuts_lep + ["n_lep_sig == 2"] + l_cuts_SFOS + l_cuts_ZonShell:
    print(i_cut)
  print("\n- After cuts:\ndf_flat.shape", df_lep.shape)
  
  df_flat = pd.concat([df_flat, df_lep], axis=1, sort=False)
  print("\n- After concatenating lep DF with flat DF:\ndf_flat.shape", df_flat.shape)

  df_flat.dropna(axis='index', inplace=True)
  print("\n- After dropping events/rows with NaNs:\ndf_flat.shape", df_flat.shape)

  if sample_type is not 'data':
    df_flat = importOpenDataInfo(df_flat)
    print("\n- After adding xsec, sumw and events info:\ndf_flat.shape", df_flat.shape)
  
  return df_flat


