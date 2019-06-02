import h5py
import uproot
import numpy as np
import pandas as pd
from ROOT import TLorentzVector
from sklearn.utils import shuffle

from infofile import infos
#from hepFunctions import invariantMass


# Features/variables to use for classification
l_features = ["channelNumber",
              "lep_pt1", "lep_eta1", "lep_phi1", "lep_E1",
              "lep_pt2", "lep_eta2", "lep_phi2", "lep_E2",
              "jet_pt1", "jet_eta1", "jet_phi1", "jet_E1",
              "jet_pt2", "jet_eta2", "jet_phi2", "jet_E2",
              "met_et", "met_phi",
              "mll", "mjj"]

# Event weights to apply, except for xsec and sumw,
# which are being handled in getEventWeights
l_eventweights = ['mcWeight',
                'scaleFactor_PILEUP',
                'scaleFactor_ELE',
                'scaleFactor_MUON',
                'scaleFactor_BTAG',
                'scaleFactor_LepTRIGGER',
                'xsec_fb',
                '1_over_sumw']


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


def getEventWeights(df, l_eventweight_column_names):
  """Return pandas Dataframe with a single combined eventweight calculated by
  multiplying the columns in the DataFrame that contain different eventweights,
  and normalizing to the integrated luminosity of the simulated samples"""

  # Multiply all eventweights except for xsec and sumw
  s_eventweight = df.loc[:,l_eventweight_column_names].agg('prod', axis='columns').copy()
  print("\n- s_eventweight.shape =\n", s_eventweight.shape)

  # Normalize to integrated luminosity: N = L*xsec =>  1/L = xsec/N
  Lumi16_periodAtoD = 10.6  # [fb-1] int lumi in data16 period A-D
  #xsec_pb_to_fb = 1e3  # convert xsec from pb to fb
  s_eventweight *= Lumi16_periodAtoD #* xsec_pb_to_fb
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


def prepareInput(store, sample_type, filename, chunk_size=1e4, n_chunks=100, entrystart=0):
  """Read in dataset in chunks, preprocess for ML and store to HDF5"""

  print("\nPrepare input")
  
  #for i_chunk in range(1, n_chunks+1):

  global n_events_chunk
  n_events_chunk = chunk_size
  i_chunk = 1

  while n_events_chunk == chunk_size:

    print("\nReading chunk #{:d}".format(i_chunk))
    entrystop = i_chunk*chunk_size  # entrystop exclusive

    df = importOpenData(sample_type, entrystart, entrystop)
    df = shuffle(df)  # shuffle the rows/events
    
    df_feat = selectFeatures(df, l_features)
    df_feat = df_feat*1  # multiplying by 1 to convert booleans to integers
    
    if sample_type is not 'data':
      df_feat["eventweight"] = getEventWeights(df, l_eventweights)
      print("\n- After adding calculating and adding eventweight column:\ndf_feat.shape", df_feat.shape)

      df_feat = df_feat[df_feat.eventweight > 0]
      print("\n- After removing events/rows with eventweight <= 0:\ndf_feat.shape", df_feat.shape)

    print("\ndf.head()\n", df_feat.head())
    
    if i_chunk is 1:
      store.append(sample_type, df_feat)
      print("\nStored initial chunk in HDF5 file")
    else:
      store.append(sample_type, df_feat)
      print("\nAppended chunk #{:d} in HDF5 file".format(i_chunk))

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

  return store



def invariantMass(l_4vector1, l_4vector2):
  """Calculate the invariant mass of two TLorentzVectors"""

  tlv1 = TLorentzVector()
  tlv2 = TLorentzVector()
  tlv1.SetPtEtaPhiE(l_4vector1[0], l_4vector1[1], l_4vector1[2], l_4vector1[3])
  tlv2.SetPtEtaPhiE(l_4vector2[0], l_4vector2[1], l_4vector2[2], l_4vector2[3])

  return (tlv1 + tlv2).M()


