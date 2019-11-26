from ROOT import TLorentzVector

def invariantMass(l_4vector1, l_4vector2):
  """Calculate the invariant mass of two TLorentzVectors"""

  tlv1 = TLorentzVector()
  tlv2 = TLorentzVector()
  tlv1.SetPtEtaPhiE(l_4vector1[0], l_4vector1[1], l_4vector1[2], l_4vector1[3])
  tlv2.SetPtEtaPhiE(l_4vector2[0], l_4vector2[1], l_4vector2[2], l_4vector2[3])

  return (tlv1 + tlv2).M()
