
#########################################################################
'''Program to calculate the forced eccentricity and forced longitude of
pericentre on a test particle in a two-planet system. Method based on
Chapter 7 of Murray & Dermott (1999). The default parameters below are
for Jupiter and Saturn in the Solar System, and reproduce Fig. 7.5 in 
Murray & Dermott (1999).

To use the program, simply change the values in the 'User Inputs' section
just below. You should not need to change anything outside of that 
section.

Feel free to use this code, and if the results go into a publication,
then please cite Pearce et al. (2025) and Murray & Dermott (1999). Also,
let me know if you find any bugs or have any requests!'''

############################### Libraries ###############################
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

############################## User inputs ##############################
# Star mass (Solar masses). Default to reproduce Murray & Durmott: 1.
mStar_mSun = 1.

# Planet 1 parameters. These are:
# 	mP1_mJup: mass in Jupiter masses (default 1.)
# 	a1_au: semimajor axis  in au (default 5.202545)
#	e1: eccentricity (default 0.0474622)
#	varpi1_deg: longitude of ascending node in degrees (default: 13.9838)
m1_mJup = 1.
a1_au = 5.202545
e1 = 0.0474622
varpi1_deg = 13.9838

# Planet 2 parameters, following the same notation as Planet 1 above.
# Default values: mP2_mJup = 0.2994, a2_au = 9.554841, e2 = 0.0575481,
# varpi2_deg = 88.719425
m2_mJup = 0.2994
a2_au = 9.554841
e2 = 0.0575481
varpi2_deg = 88.719425

# Test particle semimajor axis (or axes), in au. Can either take a list,
# in which case graphs are produced, or a single value, in which case the
# results are printed. Default: 1000 points between 0 and 30 au
as_au = np.linspace(0, 30, 1000)

#################### Constants (don't change these!) ####################
# Jupiter mass in Solar masses
mJup_mSun = 9.55e-4

# Mass ratios of planets to star
m1_mStar = m1_mJup * mJup_mSun / mStar_mSun
m2_mStar = m2_mJup * mJup_mSun / mStar_mSun

# Angles in radians
varpi1_rad = varpi1_deg / 180. * np.pi
varpi2_rad = varpi2_deg  / 180. * np.pi

############################### Functions ###############################
def GetForcedEAndVarpiOfTestParticle_deg(a_au, t_yr=0):

	print('Calculating forced e and varpi for a = %.2e au...' % a_au, end='\r')
	
	# First, get the planet parameters
	
	# Mean motion
	n1_radPerYr = GetMeanMotion_radPerYr(a1_au, mStar_mSun)
	n2_radPerYr = GetMeanMotion_radPerYr(a2_au, mStar_mSun)	
	
	# Alpha and alphaBar values
	alpha1, alphaBar1 = GetAlphaAndAlphaBar(a1_au, a2_au)
	alpha2, alphaBar2 = GetAlphaAndAlphaBar(a2_au, a1_au)
	
	# The A matrix	
	aMatrix_radPerYr = GetAMatrix_radPerYr(n1_radPerYr, n2_radPerYr, alpha1, alpha2, alphaBar1, alphaBar2)

	# The eigenvalues and scaled eigenvectors of A (Eq. 7.41 and 7.42
	# in MD99)
	gs_radPerYr, eJIs, betaJs_rad = GetGsAndEJIsAndBetaJs(aMatrix_radPerYr)
	
	g1_radPerYr = gs_radPerYr[0]
	g2_radPerYr = gs_radPerYr[1]
		
	e11 = eJIs[0][0]
	e21 = eJIs[0][1]
	e12 = eJIs[1][0]
	e22 = eJIs[1][1]

	# Now get the particle parameters

	# Mean motion		
	n_radPerYr = GetMeanMotion_radPerYr(a_au, mStar_mSun)
	
	# Alpha and alphaBar values
	alphaP1, alphaBarP1 = GetAlphaAndAlphaBar(a_au, a1_au)
	alphaP2, alphaBarP2 = GetAlphaAndAlphaBar(a_au, a2_au)
	
	# A, A1 and A2 (Eq. 7.55 and 7.56)
	A_radPerYr = GetA_radPerYr(n_radPerYr, m1_mStar, m2_mStar, alphaP1, alphaP2, alphaBarP1, alphaBarP2)
	A1_radPerYr = GetAJ_radPerYr(n_radPerYr, m1_mStar, alphaP1, alphaBarP1)
	A2_radPerYr = GetAJ_radPerYr(n_radPerYr, m2_mStar, alphaP2, alphaBarP2)

	# The vi terms (MD Eq. 7.76)
	v1_radPerYr = A1_radPerYr*e11 + A2_radPerYr*e21
	v2_radPerYr = A1_radPerYr*e12 + A2_radPerYr*e22
	
	# h0 and k0	(Eq. 7.78 and 7.79)
	const1 = v1_radPerYr / (A_radPerYr - g1_radPerYr)
	const2 = v2_radPerYr / (A_radPerYr - g2_radPerYr)

	beta1_rad = betaJs_rad[0]
	beta2_rad = betaJs_rad[1]
	
	arg1_rad = g1_radPerYr*t_yr + beta1_rad
	arg2_rad = g2_radPerYr*t_yr + beta2_rad
	
	h0 = - const1 * np.sin(arg1_rad) - const2 * np.sin(arg2_rad)
	k0 = - const1 * np.cos(arg1_rad) - const2 * np.cos(arg2_rad)

	# Get the forced eccentricity
	eF = (h0**2 + k0**2)**0.5
	
	# Get the forced longitude of pericentre
	varpiF_rad = np.arctan2(h0, k0)
	
	# Get varpi in degrees, and between 0 and 360 deg
	varpiF_deg = varpiF_rad * 180. / np.pi

	while varpiF_deg < 0: varpiF_deg += 360
	while varpiF_deg >= 360: varpiF_deg -= 360

	return eF, varpiF_deg

#------------------------------------------------------------------------
def GetAlphaAndAlphaBar(aObject_au, aPerturber_au):

	# Get min and max semimajor axes
	sortedAs_au = sorted([aObject_au, aPerturber_au])
	
	# Get alpha
	alpha = sortedAs_au[0] / sortedAs_au[1]

	# Get alphaBar if internal perturber
	if aPerturber_au < aObject_au:
		alphaBar = 1.

	# Otherwise external perturber
	else:
		alphaBar = alpha
	
	return alpha, alphaBar
	
#------------------------------------------------------------------------
def GetAMatrix_radPerYr(n1_radPerYr, n2_radPerYr, alpha1, alpha2, alphaBar1, alphaBar2):
	
	aMatrix11_radPerYr = GetAMatrixJJ(n1_radPerYr, m2_mStar, m1_mStar, alpha1, alphaBar1)
	aMatrix22_radPerYr = GetAMatrixJJ(n2_radPerYr, m1_mStar, m2_mStar, alpha2, alphaBar2)
	
	aMatrix12_radPerYr = GetAMatrixJK(n1_radPerYr, m2_mStar, m1_mStar, alpha1, alphaBar1)
	aMatrix21_radPerYr = GetAMatrixJK(n2_radPerYr, m1_mStar, m2_mStar, alpha2, alphaBar2)
	
	aMatrix_radPerYr = np.array([[aMatrix11_radPerYr, aMatrix12_radPerYr], [aMatrix21_radPerYr, aMatrix22_radPerYr]])
	
	return aMatrix_radPerYr

#------------------------------------------------------------------------
def GetAMatrixJJ(nJ_radPerYr, mK_mStar, mJ_mStar, alpha, alphaBar):
	'''Eq. 7.9 in MD99'''

	# Laplace coefficient
	b1_32 = GetLaplaceCoefficient(1.0, 1.5, alpha)
	
	# Matrix element
	aMatrixJJ_radPerYr = nJ_radPerYr / 4. * mK_mStar/(1. + mJ_mStar) * alpha * alphaBar * b1_32

	return aMatrixJJ_radPerYr
	
#------------------------------------------------------------------------
def GetAMatrixJK(nJ_radPerYr, mK_mStar, mJ_mStar, alpha, alphaBar):
	'''Eq. 7.10 in MD99'''

	# Laplace coefficient
	b2_32 = GetLaplaceCoefficient(2.0, 1.5, alpha)

	# Matrix element
	aMatrixJK_radPerYr = - nJ_radPerYr / 4. * mK_mStar/(1. + mJ_mStar) * alpha * alphaBar * b2_32

	return aMatrixJK_radPerYr

#------------------------------------------------------------------------
def GetGsAndEJIsAndBetaJs(aMatrix_radPerYr):
	'''Eq. 7.41 - 7.50 in MD99'''

	# Get the eigenvalues and unscaled eigenvectors of A (Eq. 7.41 and
	# 7.42 in MD99)
	gs_radPerYr, eJIBars = np.linalg.eig(aMatrix_radPerYr)

	g1_radPerYr = gs_radPerYr[0]
	g2_radPerYr = gs_radPerYr[1]

	e11Bar = eJIBars[0][0]
	e21Bar = eJIBars[1][0]
	e12Bar = eJIBars[0][1]
	e22Bar = eJIBars[1][1]
		
	# Get the values of h and k at t=0
	hT01 = GetH(e1, varpi1_rad)
	hT02 = GetH(e2, varpi2_rad)

	kT01 = GetK(e1, varpi1_rad)
	kT02 = GetK(e2, varpi2_rad)
	
	# Solve the simultaneous equations (Eq. 7.47 in MD99):
	# S1eJ1BarSinBeta1 + S2eJ2BarSinBeta2 = hT0J
	sJBetaJCoeffs = np.array([[e11Bar, e12Bar], [e21Bar, e22Bar]])
	
	hT0s = np.array([hT01, hT02])
	sJSinBetaJs = np.linalg.solve(sJBetaJCoeffs, hT0s)
		
	# S1eJ1BarCosBeta1 + S2eJ2BarCosBeta2 = kT0J
	kT0s = np.array([kT01, kT02])
	sJCosBetaJs = np.linalg.solve(sJBetaJCoeffs, kT0s)	

	# Solve for betas and Ss
	beta1_rad = np.arctan2(sJSinBetaJs[0], sJCosBetaJs[0])
	beta2_rad = np.arctan2(sJSinBetaJs[1], sJCosBetaJs[1])

	betaJs_rad = [beta1_rad, beta2_rad]

	s1 = (sJSinBetaJs[0]**2 + sJCosBetaJs[0]**2)**0.5
	s2 = (sJSinBetaJs[1]**2 + sJCosBetaJs[1]**2)**0.5

	# Scale the eigenvectors
	e11 = s1 * e11Bar
	e21 = s1 * e21Bar
	e12 = s2 * e12Bar
	e22 = s2 * e22Bar
	
	eJIs = np.array([[e11, e21], [e12, e22]])
	
	return gs_radPerYr, eJIs, betaJs_rad
	
#------------------------------------------------------------------------
def GetH(e, varpi_rad):
	'''Eq. 7.18 in MD99'''

	h = e * np.sin(varpi_rad)
	
	return h
	
#------------------------------------------------------------------------
def GetK(e, varpi_rad):
	'''Eq. 7.18 in MD99'''
	
	k = e * np.cos(varpi_rad)
	
	return k
	
#------------------------------------------------------------------------
def GetA_radPerYr(n_radPerYr, m1_mStar, m2_mStar, alpha1, alpha2, alphaBar1, alphaBar2):
	'''Eq. 7.55 in MD99'''

	# Laplace coefficients
	object1B1_32 = GetLaplaceCoefficient(1.0, 1.5, alpha1)
	object2B1_32 = GetLaplaceCoefficient(1.0, 1.5, alpha2)
	
	# A	
	A_radPerYr = n_radPerYr / 4. * (m1_mStar*alpha1*alphaBar1*object1B1_32 + m2_mStar*alpha2*alphaBar2*object2B1_32)

	return A_radPerYr

#------------------------------------------------------------------------
def GetAJ_radPerYr(n_radPerYr, mJ_mStar, alphaJ, alphaBarJ):
	'''Eq. 7.55 in MD99'''

	# Laplace coefficient
	b2_32 = GetLaplaceCoefficient(2.0, 1.5, alphaJ)
	
	# aJ
	AJ_radPerYr = - n_radPerYr / 4. * mJ_mStar*alphaJ*alphaBarJ*b2_32

	return AJ_radPerYr

#------------------------------------------------------------------------
def GetMeanMotion_radPerYr(a_au, mStar_mSun):
	
	# Catch if a_au not positive
	if a_au <= 0: return np.nan
	
	n_radPerYr = abs(4*np.pi**2*mStar_mSun/a_au**3)**.5

	return n_radPerYr

#--------------------------------------------------------------------------------------------------
def GetLaplaceCoefficient(j, s, alpha):
	'''Returns b_s^j(alpha)'''
	
	laplaceCoefficientIntegrand = lambda psi: np.cos(j*psi) / (1.0 - 2*alpha*np.cos(psi) + alpha**2)**s
	
	laplaceCoefficient = 1.0 / np.pi * integrate.quad(laplaceCoefficientIntegrand, 0, 2*np.pi)[0]
	
	return laplaceCoefficient

#------------------------------------------------------------------------
def MakePlot(eFs, varpiFs_rad):

	print('\nMaking plots...')

	# Define the figure and axes
	fig,axes = plt.subplots(2, figsize=(8,8))

	# Get the Pyplot default colour cycle. Differs depending on version
	# of Python and Pyplot
	try:
		colourNames = plt.rcParams['axes.prop_cycle'].by_key()['color']
	except:
		try:
			colourNames = plt.rcParams['axes.prop_cycle'].by_key()['color']
		except:
			colourNames = plt.rcParams['axes.color_cycle']
	
	# Forced eccentricity
	axes[0].plot(as_au, eFs, color=colourNames[0])
	
	axes[0].plot(a1_au, e1, marker='o', color=colourNames[3])
	axes[0].plot(a2_au, e2, marker='o', color=colourNames[3])
	
	axes[0].set_ylabel(r'Forced eccentricity')

	axes[0].set_ylim(0, 5*eFs[int(len(eFs)/2)])

	# Forced longitude of pericentre
	axes[1].plot(as_au, varpiFs_rad, color=colourNames[1])
	
	axes[1].plot(a1_au, varpi1_deg, marker='o', color=colourNames[3])
	axes[1].plot(a2_au, varpi2_deg, marker='o', color=colourNames[3])
	
	axes[1].set_ylabel(r'Forced longitude of pericentre / deg')

	axes[1].set_ylim(0, 360)	
	
	# Shared parameters
	for ax in axes:
		ax.set_xlabel(r'Test particle semimajor axis / au')
		ax.set_xlim(0, max(as_au))
	
	plt.show()

################################ Program ################################
# Determine if analysing a list of test-particle semimajor axes
if type(as_au) in [list, np.ndarray] and len(as_au) > 1:

	# Calculate the forced e and varpi for each semimajor axis
	eFs, varpiFs_deg = [], []

	for a_au in as_au:
		eF, varpiF_deg = GetForcedEAndVarpiOfTestParticle_deg(a_au)
		
		eFs.append(eF)
		varpiFs_deg.append(varpiF_deg)

	# Plot the results
	MakePlot(eFs, varpiFs_deg)

# Otherwise only a single semimajor axis is provided
else:

	# Redefine as float if necessary
	if type(as_au) is list:
		a_au = as_au[0]
	else:
		a_au = as_au

	# Calculate the forcing eccentricity and longitude of pericentre
	eF, varpiF_deg = GetForcedEAndVarpiOfTestParticle_deg(a_au)

	# Print the results
	print()
	print('Forced eccentricity: %.2e' % eF)
	print('Forced longitude of pericentre: %.2e deg' % varpiF_deg)

#########################################################################


