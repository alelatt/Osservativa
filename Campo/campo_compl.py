import numpy as np
from numpy.random import uniform, pareto, poisson
from matplotlib import pyplot as plt
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from skimage.util import random_noise
from skimage.restoration import richardson_lucy
import os
import time

"""
Field reconstruction analysis


EXECUTION EXAMPLE (">>" indicates user input):
Radius around bright pixels to which the fit is applied: >>10

Running Reconstruction
        100%

Cut at % of brightest: >>0.0015
Fictitious pixels reconstructed: 26 (0.260 %)
Stars lost in reconstruction: 12 (17.143 %)
Try other threshold? (y/n): >>n
"""

#################	GLOBAL CONSTANTS	#################
#Grid setup constants
N = 70
grid_len = 100

#Stars generator constants
exp = 2.35
M0 = 0.35
Mmax = 5

#Luminosity scaling factor
Lmin = 10

#Image modifier constants
sigma = 2
backgr = 10


#################	FUNCTIONS	#################
def GenerateSequence():
	"""
	Generates positions on the board and luminosity of stars and saves output

	Outputs:
		data : ndarray
			2D matrix such that the first column is the x coord, the second the y coord, the third is the luminosity L in solar luminosities for all stars

	Generated values of mass (M) given in solar masses
	"""

	x = uniform(0, grid_len, N)
	y = uniform(0, grid_len, N)
	M = (pareto(exp-1, N) + 1)*M0
	L = np.zeros(len(M))

	for i in range(len(M)):
		if M[i] > Mmax:
			while M[i] > Mmax:
				M[i] = (pareto(exp-1, 1) + 1)*M0

		if M[i] < 0.4:
			L[i] = 0.23*(M[i]**(2.3))
		elif M[i] > 0.4 and M[i] < 2:
			L[i] = M[i]**4
		elif M[i] > 2 and M[i] < 55:
			L[i] = 1.4*(M[i]**(1.5))
		elif M[i] > 55:
			L[i] = 32000*M[i]
		else:
			print("ERROR GENERATING M", M[i], i)
			exit()

	data = np.vstack((x, y, L)).T

	return data


def GeneratePixels():
	"""
	Generates pixel board

	Outputs:
		board_base : ndarray
			Pixel board: each cell of the 2D board is the luminosity value given as the sum of the luminosities of all the stars within the pixel renormalized and rounded

	Placement of the stars is given by "GenerateSequence()"

	The renormalization is such that the faintest star has a brightness of "Lmin" (see global constants)
	"""

	board_base = np.zeros((grid_len, grid_len))

	data = GenerateSequence()

	for i in range(0, grid_len):
		for j in range(0, grid_len):
			board_base[i,j] = np.sum(data[:,2][np.where((data[:,0] >= i) & (data[:,0] < i+1) & (data[:,1] >= j) & (data[:,1] < j+1))])

	board_base = np.round(board_base * Lmin/np.min(board_base[np.where(board_base > 0)]))

	np.savetxt("board.txt", board_base)

	return board_base


def BoardIt():
	"""
	Checks for the existence of "board.txt". If present imports the board, else generates a new board

	Outputs:
		board_out	Pixel board: each cell of the 2D board is the luminosity value given as the sum of the luminosities of all the stars within the pixel

	Board generation is done by "GeneratePixel"
	"""

	board_out = None
	if os.path.isfile('./board.txt') == False:
		board_out = GeneratePixels()

	elif os.path.isfile('./board.txt') == True:
		board_out = np.genfromtxt("board.txt")

	return board_out


def PixelPlot(board, title = "", color_scale = 'linear', color_scheme = 'Spectral'):
	"""
	Makes a colormap plot of the given board

	Inputs:
		plot_num		Identifier number for the plot (must be different for each plot, else it will be overwritten)
		board			The board to plot
		color_scale		Scale of the colormap, must be 'linear', 'log' or other specified keywords
		color_scheme	Color scheme of the colormap
	"""

	f = plt.figure(figsize = [10,10], dpi = 100, layout = 'tight')
	ax = f.add_subplot(1,1,1)
	ax.set_title(title)
	cmap = plt.get_cmap(color_scheme)
	im = ax.pcolormesh(board, cmap = cmap, norm = color_scale)
	f.colorbar(im, ax=ax)

	return


def GaussFn(x, sigma, height, offset):
	"""
	Fit funcion for a gaussian with mean = 0 and an offset

	Inputs:
		x	Data
		sigma	Sigma of the distribution
		height	Multiplicative factor
		offset	Offset
	"""

	return ((np.exp(-x**2/(2*(sigma**2)))*height) + offset)


def BoardToDistance(board_sect):
	"""
	Finds the maximum value in a section of the board and takes the minimum value on the board at each distance from the maximum

	Inputs:
		board_sect	Section of board

	Outputs:
		dist			Array with every distance from the maximum ordered from shortest distance to furthest
		vals			Array with the minimum value on the board at each distance from maximum
		center_coords	Array containing the most luminous point's coordinates on the board
	"""

	x_c, y_c = np.where(board_sect == np.max(board_sect))
	center_coords = np.array([x_c[0], y_c[0]])

	a = board_sect.shape[0]
	b = board_sect.shape[1]

	[X, Y] = np.meshgrid(np.arange(b) - center_coords[1], np.arange(a) - center_coords[0])
	R = np.sqrt(np.square(X) + np.square(Y))
	R_copy = np.copy(R)

	dist = []
	vals = []

	while np.max(R_copy)>=0:
		dist = np.append(dist, R_copy.max())
		vals = np.append(vals, np.min(board_sect[np.where(R_copy == R_copy.max())]))
		R_copy[np.where(R_copy == R_copy.max())] = -1

	dist = np.flip(dist)
	vals = np.flip(vals)

	return(dist, vals, center_coords)

def PSFFit(board_sect, sigma_min, debug = False):
	"""
	Finds sigma for gaussian PSF from a portion of the board around a given luminosity peak
	
	Inputs:
		board_sect	Section of board around a peak
		sigma_min	Minimum value for sigma, needed for the fit. Also used as first estimate for sigma in the fit
		debug		Debug option: if True the function will also output a plot of a board the same size of board_sect with the normalized PSF with center in the most luminous point

	Outputs:
		pars		Best fit estimate of the fit parameters (see "GaussFn")
		errs		Errors on the best fit estimate of the fit parameters (see "GaussFn")
		PSF_board	Board with reconstruction of the normalized discrete PSF

	The 2D board is transformed into a 1D array of the minimum values on the board at each distance from the most luminous point (see "BoardToDistance")
		The fit is done on the set of minimum values using a 1D gaussian (see "GaussFn")
	"""

	dist, vals, center_coords = BoardToDistance(board_sect)

	[pars,covm] = curve_fit(GaussFn, dist, vals, p0 = [sigma_min,np.max(vals),np.min(vals)], bounds = ([sigma_min,0,0],[10,np.inf,np.max(vals)]))
	errs = np.sqrt(covm.diagonal())

	PSFboard = np.zeros(np.shape(board_sect))
	PSFboard[center_coords[0], center_coords[1]] = pars[1]
	
	if pars[0] >= 1e-4:
		PSFboard = ((pars[0]**2)*2*np.pi)*gaussian_filter(PSFboard, sigma = pars[0], mode = 'constant', cval = 0.) + pars[2]

	if debug == True:
		PixelPlot(PSFboard, title = "Fitted PSF onto cut board")
		plt.show()
		plt.close('all')

	return (pars,errs, PSFboard)

def FindPSF(board_in, nstars = 2 , sigma_min = 1, width = 10, crowded = False, debug = False):
	"""
	Finds PSF from brightest stars

	Inputs:
		board_in	Input board
		nstars		Number of stars over which the fit is applied
		sigma_min	Necessary fit parameter (see "PSFFit")
		crowded		Background computation mode: If True the background is taken as the weighted average of the offsets in the fits, if False the background is taken as the least luminous pixel
		debug		Debug option: if True the results and starting conditions (such as the limits of the cut board over which the fit is applied) of each fit and additional info are shown

	Outputs:
		PSF_board	Board containing the PSF. The sigma is taken as the weighted average over the fits
		sig			Value of the averaged sigma
		background	Constant background value for the image taken as the minimum value found in the board (won't work if the field is too crowded)

	The user will be shown the board and will have to choose a distance around the star over which the fit is applied
		If said distance goes off the board on one or more sides the cut board will stop at the board limits
	"""

	board = np.copy(board_in)
	sigma = np.zeros(nstars)
	sigma_errs = np.zeros(nstars)
	maximum = np.zeros(nstars)
	offset = np.zeros(nstars)
	offset_errs = np.zeros(nstars)

	i = 0
	while i<nstars:
		[xp,yp] = np.where(board == board.max())
		xp = int(xp[0])
		yp = int(yp[0])

		xmin = np.max(np.array([xp-width, 0]))
		xmax = np.min(np.array([xp+width,grid_len-1]))
		ymin = np.max(np.array([yp-width, 0]))
		ymax = np.min(np.array([yp+width,grid_len-1]))

		board_sect = np.copy(board_in[xmin:xmax + 1, ymin:ymax + 1])

		pars, errs, PSFboard = PSFFit(board_sect, sigma_min, debug)
		sigma[i] = pars[0]
		sigma_errs[i] = errs[0]
		maximum[i] = pars[1]
		offset[i] = pars[2]
		offset_errs[i] = errs[2]

		board[xmin:xmax + 1, ymin:ymax + 1] = board[xmin:xmax + 1, ymin:ymax + 1] - PSFboard

		if debug == True:
			print("Fit number %i centered in (%i,%i)" %(i,xp,yp))
			print("\tCut Board Coords: x = [%i,%i], y = [%i,%i]" %(xmin, xmax, ymin, ymax))
			PixelPlot(board_sect, title = "Cut Board")
			PixelPlot(board)
			PixelPlot(board_in)
			PixelPlot(PSFboard)
			print("\tSigma = %.5f pm %.5f" %(sigma[i], errs[0]))
			print("\tMax = %.5f pm %.5f" %(maximum[i], errs[1]))
			print("\tOffset = %.5f pm %.5f" %(offset[i], errs[2]))
			plt.show()
			plt.close('all')

		i = i + 1

	sig = None
	off = None

	if (sigma_errs.any() == 0) or (offset_errs.any() == 0):
		sig = np.mean(sigma)
		off = np.mean(offset)
	else:
		sig = np.average(sigma, weights = 1./sigma_errs)
		off = np.average(offset, weights = 1./offset_errs)

	print("Fit Results:")
	print("\tAvg. Sigma = %.2f" %(sig))
	print("\tAvg. Offset = %.2f" %(off))

	sig_round = int(np.ceil(sig))
	PSF_board = np.zeros((8*sig_round + 1,8*sig_round + 1))
	PSF_board[4*sig_round,4*sig_round] = 1
	PSF_board = gaussian_filter(PSF_board, sigma = sig, mode = 'constant', cval = 0.)

	background = np.min(board_in)

	if crowded == True:
		background = off

	return PSF_board, sig, background


def Difference(a,b):
	"""
	Computes the "relative difference" between two arrays of the same dimensions

	Inputs:
		a,b	Arrays (must have same dimensions)

	Outputs:
		Relative difference computed as the sum over the absolute value of each element of the array (a-b) weighted by the sum of all elements of array a (assumed positive)
	"""

	return np.sum(np.abs(a-b))/np.sum(a)


def ChiSq(a, b):
	return np.sum((a - b)**2)


def Lucy(board, PSF_board, sigma, offset, reps = 12000, debug = False):
	"""
	Applies Lucy reconstruction

	Inputs:
		board		Board to deconvolve
		PSF_board	Board containing the estimated PSF (see "FindPSF")
		offset		Constant background found (see "FindPSF")
		reps		Number of repetitions of the reconstruction (used if use_thresh = False)
		thresh		Threshold used as stopping condition (used if use_thresh = True)
		use_thresh	Selects if the reconstruction stops after a set number of repetitions or after a certain threshold is reached
		use_chisq	Selects if the output image is the one which has the least chi squared (if use_chisq = True) or the last computed one (if use_chisq = False)
		debug		Debug option: if True will print step number and "relative difference" (see "Difference") at each step

	Outputs:
		gnext	(if use_chisq = False) Deconvoluted image from the last iteration
		saveg	(if use_chisq = True) Deconvoluted image with the least chi squared

	This recursive algorithm is assumes that the PSF was estimated correctly and doesn't attempt to recover a more precise form of it

	The reconstruction stops after a set amount of iterations or after a set threshold is reached
		The threshold is computed as the relative difference between the reconstructed image at the current and previous step (see "Difference")
	"""

	print("\nRunning Reconstruction")
	g0 = np.copy(board - offset)
	
	conv_g = convolve(g0, PSF_board, mode = 'same') + 1e-12
	gnext = g0*(convolve(g0/conv_g, np.transpose(PSF_board), mode = 'same'))
	prev_g = gnext
	chisq = [ChiSq(board, gaussian_filter(gnext + offset, sigma = sigma, mode = 'constant', cval = offset))]
	saveg = gnext

	i = 1
	while True:
		conv_g = convolve(gnext, PSF_board, mode = 'same') + 1e-12
		f_g = convolve(g0/conv_g, np.transpose(PSF_board), mode = 'same')
		gnext = gnext*f_g
		
		chisq = np.append(chisq, ChiSq(board, gaussian_filter(gnext + offset, sigma = sigma, mode = 'constant', cval = offset)))
		if ChiSq(board, gaussian_filter(gnext + offset, sigma = sigma, mode = 'constant', cval = offset)) == np.min(chisq):
			saveg = gnext

		prev_g = gnext
		i = i + 1

		if i % 100 == 0:
			print("\t%i%% "%(100*i/reps), end = "\r")
		if i == reps:
			if debug == True:
				print("Minimum Chi Squared = %.1f at step %i" %(np.min(chisq), int(np.where(chisq == np.min(chisq))[0])))

				plt.figure(figsize = [10,10], dpi = 100, layout = 'tight')
				plt.plot(chisq, '-k')
				plt.title(r'$\chi^2$ vs Number of Iteration')
				plt.xlabel("Number of Iteration")
				plt.ylabel(r'$\chi^2$')
				plt.yscale('log')
				plt.show()
				plt.close('all')

			return saveg


def Reconstruction(board, sigma, offset, reps = 10000, debug = False):
	print("\nRunning Reconstruction")

	image = np.copy(board)
	fitboard = np.zeros(np.shape(board))
	saveboard = np.zeros(np.shape(board))

	chisq = [ChiSq(board, gaussian_filter(fitboard + offset, sigma = sigma, mode = 'constant', cval = offset))]

	for i in range(0, reps):
		[xp,yp] = np.where(image == image.max())

		xp = int(xp[0])
		yp = int(yp[0])

		width = int(4*sigma)

		xmin = np.max(np.array([xp-width, 0]))
		xmax = np.min(np.array([xp+width,grid_len-1]))
		ymin = np.max(np.array([yp-width, 0]))
		ymax = np.min(np.array([yp+width,grid_len-1]))

		board_sect = np.copy(image[xmin:xmax + 1, ymin:ymax + 1])

		dist, vals, center_coords = BoardToDistance(board_sect)

		[pars,covm] = curve_fit(lambda x, height:GaussFn(x, sigma, height, offset), dist, vals, p0 = np.max([0, np.max(vals)]), bounds = (0,np.inf))
		errs = np.sqrt(covm.diagonal())

		PSFboard = np.zeros(np.shape(board_sect))
		if sigma >= 1e-4:
			PSFboard[center_coords[0], center_coords[1]] = pars * ((sigma**2)*2*np.pi)
		else:
			PSFboard[center_coords[0], center_coords[1]] = pars

		fitboard[xmin:xmax + 1, ymin:ymax + 1] = fitboard[xmin:xmax + 1, ymin:ymax + 1] + PSFboard
		chisq = np.append(chisq, ChiSq(board, gaussian_filter(fitboard + offset, sigma = sigma, mode = 'constant', cval = offset)))
		if chisq[-1] == np.min(chisq):
			saveboard = fitboard

		if sigma >= 1e-4:
			PSFboard = gaussian_filter(PSFboard, sigma = sigma, mode = 'constant', cval = 0.) + offset

		image[xmin:xmax + 1, ymin:ymax + 1] = image[xmin:xmax + 1, ymin:ymax + 1] - PSFboard

		if i % 100 == 0:
			print("\t%i%% "%(100*i/reps), end = "\r")

	if debug == True:
		print("Minimum Chi Squared = %.1f at step %i" %(np.min(chisq), int(np.where(chisq == np.min(chisq))[0][0])))

		plt.figure(figsize = [10,10], dpi = 100, layout = 'tight')
		plt.plot(chisq, '-k')
		plt.title(r'$\chi^2$ vs Number of Iteration')
		plt.xlabel("Number of Iteration")
		plt.ylabel(r'$\chi^2$')
		plt.yscale('log')
		plt.show()
		plt.close('all')

	return saveboard


def Completeness(niter = 100, treshold = 0.001, board_type = '', rec_type = 'gauss'):
	count_trys = np.zeros((grid_len, grid_len))
	count_occur = np.zeros((grid_len, grid_len))

	for i in range(0,niter):
		board = GeneratePixels()

		img = None
		psf = None
		sigma_psf = None
		offset = None

		if board_type == 'gauss':
			img = gaussian_filter(board, sigma = sigma, mode = 'constant', cval = 0)
		elif board_type == 'poisson':
			img = np.random.poisson(board)
		elif board_type == 'bakgauss':
			img = gaussian_filter(board + backgr, sigma = sigma, mode = 'constant', cval = backgr)
		else:
			img = board

		trysum = np.copy(board)
		trysum[trysum > 0] = 1
		count_trys = count_trys + trysum

		psf, sigma_psf, offset = FindPSF(img, nstars = 5, sigma_min = 1, crowded = True)

		board_out = None
		if rec_type == 'gauss':
			board_out = Reconstruction(img, sigma_psf, offset, reps = 4000)
		if rec_type == 'lucy':
			board_out = Lucy(img, psf, sigma_psf, offset, reps = 4000)

		binary_out = np.copy(board_out)
		binary_out[binary_out < treshold*np.max(binary_out)] = 0
		binary_out[binary_out >= treshold*np.max(binary_out)] = 1

		occursum = 2*trysum - binary_out
		occursum[occursum <= 0] = 0
		occursum[occursum >=2] = 0
		count_occur = count_occur + occursum
		
		print("\n",i+1,"\n")
		
		outerr = np.zeros(np.shape(board))

	return np.divide(count_occur, count_trys, out = outerr, where = count_trys != 0)

t1 = time.time()
out = Completeness(niter = 1000, treshold = 0.001, board_type = 'gauss', rec_type = 'gauss')
t2 = time.time()
print((t2-t1)/60)
PixelPlot(out)
plt.show()
choice = input("Save? ")
if choice == 'y':
	np.savetxt("out_stats.txt", out)