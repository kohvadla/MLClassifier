# -- Background samples
l_bkg = [
    'Zjets',
    'diboson',
    'higgs',
    'ttbar',
    'singleTop',
    'topOther',
    'lowMassDY',
    'Wjets',
    'triboson',
    ]
l_bkg.reverse()

# -- Signal samples
# delta_m < 200 GeV
l_sig_low = [
    'C1N2_WZ_350p0_250p0_2L2J',
    'C1N2_WZ_300p0_250p0_2L2J',
    'C1N2_WZ_350p0_300p0_2L2J',
    'C1N2_WZ_150p0_100p0_2L2J',
    'C1N2_WZ_300p0_200p0_2L2J',
    'C1N2_WZ_350p0_200p0_2L2J',
    'C1N2_WZ_150p0_50p0_2L2J',
    'C1N2_WZ_250p0_100p0_2L2J',
    'C1N2_WZ_250p0_150p0_2L2J',
    'C1N2_WZ_500p0_400p0_2L2J',
    'C1N2_WZ_450p0_400p0_2L2J',
    'C1N2_WZ_200p0_150p0_2L2J',
    'C1N2_WZ_400p0_300p0_2L2J',
    'C1N2_WZ_100p0_0p0_2L2J',
    'C1N2_WZ_300p0_150p0_2L2J',
    'C1N2_WZ_450p0_350p0_2L2J',
    'C1N2_WZ_100p0_50p0_2L2J',
    'C1N2_WZ_200p0_100p0_2L2J',
    'C1N2_WZ_250p0_200p0_2L2J',
    'C1N2_WZ_500_350_2L2J',
    'C1N2_WZ_550_400_2L2J',
    'C1N2_WZ_150p0_1p0_2L2J',
    'C1N2_WZ_200p0_1p0_2L2J',
    'C1N2_WZ_200p0_50p0_2L2J',
    ]

# 200 <= delta_m < 450 GeV
l_sig_int = [
    'C1N2_WZ_500p0_100p0_2L2J',
    'C1N2_WZ_450p0_150p0_2L2J',
    'C1N2_WZ_300p0_100p0_2L2J',
    'C1N2_WZ_350p0_100p0_2L2J',
    'C1N2_WZ_400p0_100p0_2L2J',
    'C1N2_WZ_350p0_50p0_2L2J',
    'C1N2_WZ_450p0_250p0_2L2J',
    'C1N2_WZ_450p0_50p0_2L2J',
    'C1N2_WZ_400p0_0p0_2L2J',
    'C1N2_WZ_350p0_0p0_2L2J',
    'C1N2_WZ_400p0_200p0_2L2J',
    'C1N2_WZ_500p0_200p0_2L2J',
    'C1N2_WZ_350p0_150p0_2L2J',
    'C1N2_WZ_500p0_300p0_2L2J',
    'C1N2_WZ_250_50_2L2J_2L7',
    'C1N2_WZ_500_250_2L2J',
    'C1N2_WZ_500_150_2L2J',
    'C1N2_WZ_550_350_2L2J',
    'C1N2_WZ_550_300_2L2J',
    'C1N2_WZ_550_250_2L2J',
    'C1N2_WZ_550_200_2L2J',
    'C1N2_WZ_550_150_2L2J',
    'C1N2_WZ_600_400_2L2J',
    'C1N2_WZ_600_350_2L2J',
    'C1N2_WZ_600_300_2L2J',
    'C1N2_WZ_600_250_2L2J',
    'C1N2_WZ_600_200_2L2J',
    'C1N2_WZ_650_350_2L2J',
    'C1N2_WZ_700_400_2L2J',
    'C1N2_WZ_700_300_2L2J',
    'C1N2_WZ_650_250_2L2J',
    'C1N2_WZ_250p0_1p0_2L2J',
    ]

# delta_m >= 450 GeV
l_sig_high = [
    'C1N2_WZ_500p0_0p0_2L2J',
    'C1N2_WZ_500_50_2L2J',
    'C1N2_WZ_550_100_2L2J',
    'C1N2_WZ_550_50_2L2J',
    'C1N2_WZ_550_0_2L2J',
    'C1N2_WZ_600_150_2L2J',
    'C1N2_WZ_600_100_2L2J',
    'C1N2_WZ_600_50_2L2J',
    'C1N2_WZ_600_0_2L2J',
    'C1N2_WZ_650_150_2L2J',
    'C1N2_WZ_650_50_2L2J',
    'C1N2_WZ_700_200_2L2J',
    'C1N2_WZ_700_100_2L2J',
    'C1N2_WZ_700_0_2L2J',
    ]

d_sig = {
    'low': l_sig_low,
    'int': l_sig_int,
    'high': l_sig_high
    }

