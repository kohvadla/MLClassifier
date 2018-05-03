import sys
import h5py
import numpy as np
from root_numpy import tree2array, root2array
import ROOT
from sklearn.preprocessing import Imputer
from featureList import features

# User-defined
fileName = "/mn/felt/u1/kovadla/cern/AnalysisChallenge2018/ewk/flat_ntuples/2L_sig_flat.root"
treeName = "FlatTree"
#treeName_eventweight = "FlatTree_eventweight"
selection = ""

# ROOT files
print "Opening ROOT file..."
outputName = fileName[:len(fileName)-5]+".h5"
print "outputName = "+outputName
inputFile = ROOT.TFile(fileName)
inputTree = inputFile.Get(treeName)

# Chunks
totalEvents = inputTree.GetEntries()
print "totalEvents",totalEvents
chunkSize = 1000
nChunks = (totalEvents // chunkSize) + 1
print "nChunks",nChunks
remainderSize = totalEvents-(nChunks-1)*chunkSize
print "remainderSize",remainderSize

# Output file
with h5py.File(outputName, "w") as f:

   # First chunk (numpy array)
   print "Processing events 0 to",chunkSize
   firstChunk = tree2array(inputTree,start=0,stop=chunkSize,step=1,selection=selection,branches=trackingFeatures)
   # Break into columns
   firstChunk = np.array(firstChunk[:].tolist())

   # Output data structure
   dtype = firstChunk.dtype
   print "firstChunk[0].dtype = ",firstChunk[0].dtype
   print "dtype = "+str(dtype)
   row_count = firstChunk.shape[0]
   maxshape = (None,) + firstChunk.shape[1:]
   dset = f.create_dataset(treeName, shape=firstChunk.shape, maxshape=maxshape,chunks=firstChunk.shape, dtype=dtype)
   # Write the first chunk 
   dset[:] = firstChunk
   print "dset.shape",dset.shape

   ## First chunk event of weight dataset
   #firstChunk_eventweight = tree2array(inputTree,start=0,stop=chunkSize,step=1,selection=selection,branches=["event_weight"])
   ## Break into columns
   #firstChunk_eventweight = np.array(firstChunk_eventweight[:].tolist())

   ## Output data structure
   #dtype_eventweight = firstChunk_eventweight.dtype
   #print "firstChunk_eventweight[0].dtype = ",firstChunk_eventweight[0].dtype
   #print "dtype_eventweight = "+str(dtype_eventweight)
   #row_count_eventweight = firstChunk_eventweight.shape[0]
   #maxshape_eventweight = (None,) + firstChunk_eventweight.shape[1:]
   #dset_eventweight = f.create_dataset(treeName_eventweight, shape=firstChunk_eventweight.shape, maxshape=maxshape_eventweight,chunks=firstChunk_eventweight.shape, dtype=dtype_eventweight)
   ## Write the first chunk 
   #dset_eventweight[:] = firstChunk_eventweight
   #print "dset_eventweight.shape",dset_eventweight.shape

   # Loop over chunks
   for chunk in range(chunkSize,nChunks*chunkSize,chunkSize):
      start = chunk
      stop = start+chunkSize 
      if (stop>totalEvents): stop = totalEvents 
      print "Processing events ",start," to ",stop      
      # Extraction of data in numpy form, chunk by chunk
      chunk = tree2array(inputTree,
                         start=start,
                         stop=stop,
                         step=1,
                         selection = selection,
                         branches=trackingFeatures)
      #chunk_eventweight = tree2array(inputTree,
      #                   start=start,
      #                   stop=stop,
      #                   step=1,
      #                   selection = selection,
      #                   branches=["event_weight"])
      
      # Break up into columns
      chunk = np.array(chunk[:].tolist())
      #chunk_eventweight = np.array(chunk_eventweight[:].tolist())
      
      # Replane -999
      imp = Imputer(missing_values=-999, strategy='mean', axis=0)
      imp.fit(chunk)
      chunk = imp.transform(chunk)

      # Resize the dataset to accommodate the next chunk of rows
      dset.resize(row_count + chunk.shape[0], axis=0)
      #dset_eventweight.resize(row_count + chunk_eventweight.shape[0], axis=0)

      # Write the next chunk
      dset[row_count:] = chunk
      #dset_eventweight[row_count:] = chunk_eventweight

      # Increment the row count
      row_count += chunk.shape[0]
      print "chunk.shape",chunk.shape
      print "dset.shape",dset.shape
      #print "chunk_eventweight.shape",chunk_eventweight.shape
      #print "dset_eventweight.shape",dset_eventweight.shape


print "Number of events    = ",totalEvents
print "Number of features  = ",len(trackingFeatures)
print "Output file name = ",outputName
