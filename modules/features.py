
# Features/variables to use for classification
l_features = ["DatasetNumber", "RandomRunNumber",
              "lepPt1", "lepEta1", "lepPhi1", "lepM1",
              "lepPt2", "lepEta2", "lepPhi2", "lepM2",
              "jetPt1", "jetEta1", "jetPhi1", "jetM1",
              "jetPt2", "jetEta2", "jetPhi2", "jetM2",
              "met_Et", "met_Sign",
              'mt2leplsp_0',
              'mll', 'Rll', 'Ptll', 'Etall', 'dPhiPllMet',
              'nJet30', 'Ht30',
              'dPhiVectorSumJetsMET',
              'DPhiJ1Met', 'minDPhi2JetsMet',
              'dPhiMetJet1',
              ]

l_features_2L2J = l_features + [
                  'mjj', 'Rjj',
                  'METOverPtZ', 'METOverPtW',
                  'dPhiPjjMet',
                  'H2PP', 'H5PP', 'RPT_HT5PP', 'R_minH2P_minH3P',
                  'dphiVP', 'minDphi',
                  ]

l_features_2L3Jplus = l_features + [
                  "jetPt3", "jetEta3", "jetPhi3", "jetM3",
                  'mjj_minDPhiZMET', 'dPhiMetISR',
                  'PTISR', 'RISR', 'PTI', 'dphiISRI',
                  'PTCM', 'NjS', 'NjISR', 
                  'MZ', 'MJ'
                  ]

# Event weights to apply
l_eventweights = ['pileupWeight',
                  'leptonWeight',
                  'eventWeight',
                  'genWeight',
                  'bTagWeight',
                  'jvtWeight',
                  'globalDiLepTrigSF',
                  'targetLumi']

