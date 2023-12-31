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



#################	GLOBAL CONSTANTS	#################
#Grid setup constants
N = 70
grid_len = 100

#Stars generator constants
exp = 2.35
M0 = 0.4
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

	Generated values of mass (M) given in solar masses and luminosity (L) in solar luminosities
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
		elif M[i] >= 0.4 and M[i] < 2:
			L[i] = M[i]**4
		elif M[i] >= 2 and M[i] < 55:
			L[i] = 1.4*(M[i]**(1.5))
		elif M[i] >= 55:
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

	board_base *= Lmin/(0.4**4)

	return board_base


def PixelPlot(board, title = '', color_scale = 'linear', color_scheme = 'Spectral'):
	"""
	Makes a colormap plot of the given board

	Inputs:
		board : ndarray
			The board to plot
		title : str
			Plot title
		color_scale : str
			Scale of the colormap, must be 'linear', 'log' or other specified keywords
		color_scheme : str
			Color scheme of the colormap
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
		x : ndarray
			Data
		sigma : float
			Sigma of the distribution
		height : float
			Multiplicative factor
		offset : float
			Offset
	"""

	return ((np.exp(-x**2/(2*(sigma**2)))*height) + offset)


def BoardToDistance(board_sect):
	"""
	Finds the maximum value in a section of the board and takes the minimum value at each distance from the maximum

	Inputs:
		board_sect : ndarray
			Section of board

	Outputs:
		dist : ndarray
			Array with every distance from the maximum ordered from shortest distance to furthest
		vals : ndarray
			Array with the minimum value on the board at each distance from maximum
		weights : ndarray
			Array with weights for each distance (needed for later fit). The weight corresponds to how many pixels are at the relative distance from the center
		center_coords : 
			Array containing the most luminous point's coordinates on the board
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
	weights = []

	while np.max(R_copy)>=0:
		dist = np.append(dist, R_copy.max())
		vals = np.append(vals, np.min(board_sect[np.where(R_copy == R_copy.max())]))
		weights = np.append(weights, len(np.where(R_copy == R_copy.max())[0]))
		R_copy[np.where(R_copy == R_copy.max())] = -1

	dist = np.flip(dist)
	vals = np.flip(vals)
	weights = np.flip(weights)

	return(dist, vals, weights, center_coords)


def PSFFit(board_sect, sigma_min, debug = False):
	"""
	Finds sigma for gaussian PSF from a portion of the board around a given luminosity peak
	
	Inputs:
		board_sect : ndarray
			Section of board around a peak
		sigma_min : float
			Minimum value for sigma, needed for the fit. Also used as first estimate for sigma in the fit
		debug : bool
			Debug option: if True the function will also output a plot of a board the same size of board_sect with the normalized PSF with center in the most luminous point

	Outputs:
		pars : ndarray
			Best fit estimate of the fit parameters (see "GaussFn")
		errs : ndarray
			Errors on the best fit estimate of the fit parameters (see "GaussFn")
		PSF_board : ndarray
			Board with reconstruction of the normalized discrete PSF

	The 2D board is transformed into a 1D array of the minimum values on the board at each distance from the most luminous point (see "BoardToDistance")
		The fit is done on the set of minimum values using a 1D gaussian (see "GaussFn")
	"""

	dist, vals, weights, center_coords = BoardToDistance(board_sect)
	vals[vals < 0] = 0

	[pars,covm] = curve_fit(GaussFn, dist, vals, sigma = 1/weights, p0 = [sigma_min,np.max(vals),np.min(vals)], bounds = ([sigma_min,0,0],[10,np.inf,np.max(vals)]))
	errs = np.sqrt(covm.diagonal())

	PSFboard = np.zeros(np.shape(board_sect))
	PSFboard[center_coords[0], center_coords[1]] = pars[1]
	
	if pars[0] >= 1e-4:
		PSFboard = ((pars[0]**2)*2*np.pi)*gaussian_filter(PSFboard, sigma = pars[0], mode = 'constant', cval = 0.)

	if debug == True:
		PixelPlot(PSFboard, title = "Fitted PSF onto cut board")
		plt.show()
		plt.close('all')

	return (pars, errs, PSFboard)


def FindPSF(board_in, nstars = 2 , sigma_min = 1, width = 10, crowded = False, debug = False):
	"""
	Finds PSF from brightest stars

	Inputs:
		board_in : ndarray
			Input board
		nstars : float
			Number of stars over which the fit is applied
		sigma_min : float
			Necessary fit parameter (see "PSFFit")
		width : int
			Half-width of the square over which the fit is applied
		crowded : bool
			Background computation mode: If True the background is taken as the weighted average of the offsets in the fits, if False the background is taken as the least luminous pixel
		debug : bool
			Debug option: if True the results and starting conditions (such as the limits of the cut board over which the fit is applied) of each fit and additional info are shown

	Outputs:
		PSF_board : ndarray
			Board containing the PSF. The sigma is taken as the weighted average over the fits
		sig : float
			Value of the averaged sigma
		background : float
			Constant background value for the image taken as the minimum value found in the board (won't work if the field is too crowded)

	If the square over which the fit is applied goes off the board on one or more sides the board section over which the fit is applied will stop at the board limits
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


def ChiSq(a, b):
	"""
	Computes chisq

	Inputs:
	a,b : ndarray
		Boards over which the chisq is computed
	"""

	return np.sum((a - b)**2)


def Lucy(board, PSF_board, sigma, offset, reps = 12000, debug = False):
	"""
	Applies Lucy reconstruction

	Inputs:
		board : ndarray
			Board to deconvolve
		PSF_board : ndarray
			Board containing the estimated PSF (see "FindPSF")
		sigma : float
			PSF sigma found (see "FindPSF")
		offset : float
			Constant background found (see "FindPSF")
		reps : int
			Number of repetitions of the reconstruction
		debug : bool
			Debug option: if True will print the chisq plot and its minimum value and step at which it was achieved

	Outputs:
		saveg : ndarray
			Deconvoluted image with the least chi squared

	This recursive algorithm is assumes that the PSF was estimated correctly and doesn't attempt to recover a more precise form of it

	The reconstruction stops after a set amount of iterations or if the last minimum in the chisq was found more than 1/3 of the repetitions ago
	"""

	print("\nRunning Reconstruction")
	g0 = np.copy(board - offset)
	
	conv_g = convolve(g0, PSF_board, mode = 'same') + 1e-12
	gnext = g0*(convolve(g0/conv_g, np.transpose(PSF_board), mode = 'same'))
	chisq = [ChiSq(board, gaussian_filter(gnext + offset, sigma = sigma, mode = 'constant', cval = offset))]
	saveg = gnext

	i = 1
	min_index = 1
	min_chisq = chisq
	while True:
		conv_g = convolve(gnext, PSF_board, mode = 'same') + 1e-12
		f_g = convolve(g0/conv_g, np.transpose(PSF_board), mode = 'same')
		gnext = gnext*f_g
		
		chisq = np.append(chisq, ChiSq(board, gaussian_filter(gnext + offset, sigma = sigma, mode = 'constant', cval = offset)))
		if chisq[-1] < min_chisq:
			saveg = gnext
			min_index = i
			min_chisq = chisq[-1]

		if i % 100 == 0:
			print("\t%i%% "%(100*i/reps), end = "\r")
		if (i - min_index >= reps/3) or (i == reps):
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

		i = i + 1


def Reconstruction(board, sigma, offset, reps = 12000, debug = False):
	"""
	Reconstructs image using gaussian fits

	Inputs:
		board : ndarray
			Board to deconvolve
		sigma : float
			PSF sigma found (see "FindPSF")
		offset : float
			Constant background found (see "FindPSF")
		reps : int
			Number of repetitions of the reconstruction
		debug : bool
			Debug option: if True will print the chisq plot and its minimum value and step at which it was achieved

	Outputs:
		saveboard : ndarray
			Deconvoluted image with the least chi squared

	This recursive algorithm is assumes that the PSF was estimated correctly (thus locking in sigma and offset in the gaussian fit)

	The reconstruction stops after a set amount of iterations or if the last minimum in the chisq was found more than 1/3 of the repetitions ago
	"""

	print("\nRunning Reconstruction")

	image = np.copy(board)
	fitboard = np.zeros(np.shape(board))
	saveboard = np.zeros(np.shape(board))

	chisq = [ChiSq(board, gaussian_filter(fitboard + offset, sigma = sigma, mode = 'constant', cval = offset))]

	i = 1
	min_index = 1
	min_chisq = chisq
	while True:
		[xp,yp] = np.where(image == image.max())

		xp = int(xp[0])
		yp = int(yp[0])

		width = int(np.rint(4*sigma))

		xmin = np.max(np.array([xp-width, 0]))
		xmax = np.min(np.array([xp+width,grid_len-1]))
		ymin = np.max(np.array([yp-width, 0]))
		ymax = np.min(np.array([yp+width,grid_len-1]))

		board_sect = np.copy(image[xmin:xmax + 1, ymin:ymax + 1])

		dist, vals, weights, center_coords = BoardToDistance(board_sect)
		vals[vals < offset] = offset

		[pars,covm] = curve_fit(lambda x, height:GaussFn(x, sigma, height, offset), dist, vals, sigma = 1/weights, p0 = np.max([0, np.max(vals)]), bounds = (0,np.inf))
		errs = np.sqrt(covm.diagonal())

		PSFboard = np.zeros(np.shape(board_sect))
		if sigma >= 1e-4:
			PSFboard[center_coords[0], center_coords[1]] = pars * ((sigma**2)*2*np.pi)
		else:
			PSFboard[center_coords[0], center_coords[1]] = pars

		fitboard[xmin:xmax + 1, ymin:ymax + 1] = fitboard[xmin:xmax + 1, ymin:ymax + 1] + PSFboard
		chisq = np.append(chisq, ChiSq(board, gaussian_filter(fitboard + offset, sigma = sigma, mode = 'constant', cval = offset)))
		if chisq[-1] < min_chisq:
			saveboard = fitboard
			min_index = i
			min_chisq = chisq[-1]

		if sigma >= 1e-4:
			PSFboard = gaussian_filter(PSFboard, sigma = sigma, mode = 'constant', cval = 0.)# + offset

		image[xmin:xmax + 1, ymin:ymax + 1] = image[xmin:xmax + 1, ymin:ymax + 1] - PSFboard

		if i % 100 == 0:
			print("\t%i%% "%(100*i/reps), end = "\r")

		if (i - min_index >= reps/3) or (i == reps):
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

		i = i + 1


def Completeness(niter = 100, board_type = 'gauss', lumin_treshold = [0.01, 0.25, 0.5, 1]):
	"""
	Computes completeness and luminosity distribution from large sample of images

	Inputs:
		niter : int
			Number of boards to generate and reconstruct
		board_type : str
			Type of reconstruction (use 'gauss', 'backgauss', 'poissgauss')
		lumin_treshold : list
			List of luminosity precisions to use in computing completeness

	Outputs:
		out_gauss : list
			List containing completeness 2D arrays of the reconstructed boards from gaussian fit computed using the various luminosity tresholds
		out_lucy : list
			List containing completeness 2D arrays of the reconstructed boards from RL reconstruction computed using the various luminosity tresholds
		lumin_in : list
			List of luminosity values of every reconstructed pixel of the generated boards
		lumin_g : list
			List of luminosity values of every reconstructed pixel of the reconstructed boards from gaussian fit
		lumin_l : list
			List of luminosity values of every reconstructed pixel of the reconstructed boards from RL reconstruction
	
	A folder with all the relevant data is created the first time the reconstruction is run
		Generated and reconstructed boards are saved in appropriately named folders
		Completeness boards are saved

	If the folder and the generated/reconstructed boards is already present only the completeness (if not already present for the given tresholds) and luminosity statistics are computed
	"""

	path = './' + board_type + '_' + str(N) + '_' + str(sigma) + '_' + str(niter)
	path_b = path + '/boards'
	path_g = path + '/gauss'
	path_l = path + '/lucy'

	if os.path.isdir(path) == False:
		os.makedirs(path)
		os.makedirs(path_b)
		os.makedirs(path_g)
		os.makedirs(path_l)

		tot_time = 0

		for i in range(0,niter):
			t1 = time.time()
			board = GeneratePixels()

			np.savetxt(path_b + '/board_' + str(i) + '.txt', board)

			img = None
			psf = None
			sigma_psf = None
			offset = None

			if board_type == 'gauss':
				img = gaussian_filter(board, sigma = sigma, mode = 'constant', cval = 0)
			elif board_type == 'backgauss':
				img = gaussian_filter(board + backgr, sigma = sigma, mode = 'constant', cval = backgr)
			elif board_type == 'poissgauss':
				img = np.random.poisson(gaussian_filter(board, sigma = sigma, mode = 'constant', cval = 0))

			psf, sigma_psf, offset = FindPSF(img, nstars = 5, sigma_min = 1, crowded = True)

			board_out_g = Reconstruction(img, sigma_psf, offset, reps = 4000)
			board_out_l = Lucy(img, psf, sigma_psf, offset, reps = 4000)

			np.savetxt(path_g + '/out_' + str(i) + '.txt', board_out_g)
			np.savetxt(path_l + '/out_' + str(i) + '.txt', board_out_l)

			t2 = time.time() - t1
			tot_time = tot_time + t2
			avg_time = tot_time/(i+1)
			if i % 10 == 0:
				print("\n\n------Average Time %.1f sec, ETA %i min\n" %(avg_time, avg_time*(niter - (i+1))/60))
			else:
				print('\n')


	for j in range(len(lumin_treshold)):
		if (os.path.isfile(path + '/out_stats_gauss_' + str(int(100*lumin_treshold[j])) + '.txt') == False) or (os.path.isfile(path + '/out_stats_lucy_' + str(int(100*lumin_treshold[j])) + '.txt') == False):
			count_trys = np.zeros((grid_len, grid_len))
			count_occur_g = np.zeros((grid_len, grid_len))
			count_occur_l = np.zeros((grid_len, grid_len))

			print("\nComputing Completeness:")
			for i in range(0,niter):
				board = np.genfromtxt(path_b + '/board_' + str(i) + '.txt')
				board_out_g = np.genfromtxt(path_g + '/out_' + str(i) + '.txt')
				board_out_l = np.genfromtxt(path_l + '/out_' + str(i) + '.txt')

				trysum = np.copy(board)
				trysum[trysum > 0] = 1
				count_trys = count_trys + trysum

				out_copy_g = np.copy(board_out_g)
				out_copy_g[out_copy_g > 0] = 1
				out_copy_g[out_copy_g <= 0] = 0

				lumin_temp_g = np.zeros((grid_len, grid_len))
				lumin_temp_g[np.where((board_out_g >= (1 - lumin_treshold[j])*board) & (board_out_g <= (1 + lumin_treshold[j])*board))] = 1

				temp_out_g = np.copy(trysum) + out_copy_g + lumin_temp_g

				occursum_g = np.zeros((grid_len, grid_len))
				occursum_g[temp_out_g == 3] = 1
				count_occur_g = count_occur_g + occursum_g
				
				out_copy_l = np.copy(board_out_l)
				out_copy_l[out_copy_l > 0] = 1
				out_copy_l[out_copy_l <= 0] = 0

				lumin_temp_l = np.zeros((grid_len, grid_len))
				lumin_temp_l[np.where((board_out_l >= (1 - lumin_treshold[j])*board) & (board_out_l <= (1 + lumin_treshold[j])*board))] = 1

				temp_out_l = np.copy(trysum) + out_copy_l + lumin_temp_l

				occursum_l = np.zeros((grid_len, grid_len))
				occursum_l[temp_out_l == 3] = 1
				count_occur_l = count_occur_l + occursum_l

				if i % 10 == 0:
					print("\t%i%% "%(100*i/niter), end = "\r")
				
			outerr = np.zeros(np.shape(board))

			out_g = np.divide(np.copy(count_occur_g), np.copy(count_trys), out = np.copy(outerr), where = np.copy(count_trys) != 0)
			out_l = np.divide(np.copy(count_occur_l), np.copy(count_trys), out = np.copy(outerr), where = np.copy(count_trys) != 0)
			np.savetxt(path + '/out_stats_gauss_' + str(int(100*lumin_treshold[j])) + '.txt', out_g)
			np.savetxt(path + '/out_stats_lucy_' + str(int(100*lumin_treshold[j])) + '.txt', out_l)

	out_gauss = []
	out_lucy = []

	for i in range(len(lumin_treshold)):
		gauss = np.loadtxt(path + '/out_stats_gauss_' + str(int(100*lumin_treshold[i])) + '.txt')
		lucy = np.loadtxt(path + '/out_stats_lucy_' + str(int(100*lumin_treshold[i])) + '.txt')
		out_gauss.append(gauss)
		out_lucy.append(lucy)

	lumin_in = []
	lumin_g = []
	lumin_l = []
	
	print("Computing Luminosity Statistics")
	for i in range(0,niter):
		board = np.genfromtxt(path_b + '/board_' + str(i) + '.txt')
		board_out_g = np.genfromtxt(path_g + '/out_' + str(i) + '.txt')
		board_out_l = np.genfromtxt(path_l + '/out_' + str(i) + '.txt')

		lumin_in.append(board)
		lumin_g.append(board_out_g)
		lumin_l.append(board_out_l)

		if i % 10 == 0:
			print("\t%i%% "%(100*i/niter), end = "\r")

	lumin_in = np.concatenate(lumin_in).ravel()
	lumin_g = np.concatenate(lumin_g).ravel()
	lumin_l = np.concatenate(lumin_l).ravel()
	
	return (out_gauss, out_lucy, lumin_in, lumin_g, lumin_l)


def Visualise(board):
	"""
	Visualises one single completeness result board and allows for basic analysis

	Inputs:
		board : ndarry
			Completeness statistics

	The user will see the board and its histogram and will be able to choose a cut in completeness to separate two poissible distributions.
		If a cut is chosen, the board will be shown separated in the two regions
	"""

	PixelPlot(board)
	plt.show()

	points = np.sort(board.flatten())
	plt.figure(figsize = [10,10], dpi = 100, layout = 'tight')
	plt.hist(points, bins = 25)
	plt.show(block = False)
	choice = input("\nWant to cut? (y/n) ")

	if choice == 'y':
		cut = float(input("\tCut at: "))
		plt.close('all')

		under = np.copy(board)
		over = np.copy(board)

		under[board < cut] = 1
		over[board < cut] = 0
		under[board >= cut] = 0
		over[board >= cut] = 1

		PixelPlot(over, color_scheme = 'gray', title = 'Over-Under treshold regions')
		plt.show()

		under_p = points[points < cut]
		over_p = points[points >= cut]

		mean_under = np.mean(under_p)
		sigma_under = np.std(under_p)
		mean_over = np.mean(over_p)
		sigma_over = np.std(over_p)
		mean = np.mean(points)
		sigma = np.std(points)

		plt.figure(figsize = [10,10], dpi = 100, layout = 'tight')
		plt.hist(under_p, bins = 25)
		plt.axvline(mean_under, color = 'k')
		plt.axvline(mean_under + sigma_under, color = 'r')
		plt.axvline(mean_under - sigma_under, color = 'r')

		plt.figure(figsize = [10,10], dpi = 100, layout = 'tight')
		plt.hist(over_p, bins = 25)
		plt.axvline(mean_over, color = 'k')
		plt.axvline(mean_over + sigma_over, color = 'r')
		plt.axvline(mean_over - sigma_over, color = 'r')
		plt.show()

		print("\n\tUnder treshold mean: %.3f pm %.3f" %(mean_under, sigma_under))
		print("\tOver treshold mean: %.3f pm %.3f" %(mean_over, sigma_over))
		print("\tOverall mean: %.3f pm %.3f" %(np.mean(points), np.std(points)))

	else:
		plt.close('all')

		mean = np.mean(points)
		sigma = np.std(points)

		plt.figure(figsize = [10,10], dpi = 100, layout = 'tight')
		plt.hist(points, bins = 25)
		plt.axvline(mean, color = 'k')
		plt.axvline(mean + sigma, color = 'r')
		plt.axvline(mean - sigma, color = 'r')
		plt.show()

		print("\n\tMean: %.3f pm %.3f" %(np.mean(points), np.std(points)))

	return


def Visualise_Multipl(board_list, lumin_treshold, board_type = 'gauss', rec_type = '', save_fig = False):
	"""
	Shows and saves two figures: one with the completeness boards (with luminosity tresholds given in input) and the other with the relevant histograms

	Inputs:
		board_list : list
			List contaioning all completeness boards given as output from "Completeness"
		lumin_treshold : list
			List containing all luminosity precisions relative to the boards (used in titling the plots)
		board_type : str
			Added effects type ('gauss', 'backgauss', 'poissgauss') used in saving the file
		rec_type : str
			Reconstruction type ('gauss', 'lucy') used in saving the file
		save_fig : bool
			If True the plots will be saved
	"""

	cmap = plt.get_cmap('Spectral')
	nrows = int(np.ceil(len(board_list)/2))

	fig, axes = plt.subplots(nrows = nrows, ncols = 2, figsize = [11,10])
	i = 0
	for ax in axes.flat:
		if i < len(board_list):
		    im = ax.pcolormesh(board_list[i], cmap = cmap, norm = 'linear', vmin = 0, vmax = 1)
		    ax.set_title("Luminosity Precision " + str(int(lumin_treshold[i]*100)) + "%", fontsize = 20)
		    ax.tick_params(labelsize = 15)
		    i += 1

	fig.subplots_adjust(left = 0.05, bottom = 0.05, top = 0.95, right = 0.85)
	cbar_ax = fig.add_axes([0.9, 0.1, 0.05, 0.8])
	fig.colorbar(im, cax=cbar_ax)
	cbar_ax.tick_params(labelsize = 15)
	fig.get_constrained_layout()
	if save_fig == True:
		fig.savefig(fname = board_type + str(sigma) + '_boards_' + rec_type + '.pdf', format = 'pdf')
	plt.show()

	fig, axes = plt.subplots(nrows = nrows, ncols = 2, figsize = [10,10], sharex = True, sharey = True)
	i = 0
	for ax in axes.flat:
		if i < len(board_list):
		    ax.hist(np.sort(board_list[i].flatten()), range = (0,1), bins = 35)
		    ax.set_title("Luminosity Precision " + str(int(lumin_treshold[i]*100)) + "%", fontsize = 20)
		    ax.annotate('Mean = ' + str(np.round(np.mean(board_list[i]), 2)), xy=(0.03, 0.93), xycoords = 'axes fraction', fontsize = 15)
		    ax.xaxis.set_tick_params(labelbottom=True)
		    ax.yaxis.set_tick_params(labelbottom=True)
		    ax.tick_params(labelsize = 15)
		    i += 1

	fig.subplots_adjust(left = 0.07, bottom = 0.07, top = 0.93, right = 0.93)
	if save_fig == True:
		fig.savefig(fname = board_type + str(sigma) + '_hists_' + rec_type + '.pdf', format = 'pdf')
	plt.show()

	return


def LuminDistr(list_in, list_gauss, list_lucy, board_type = 'gauss', save_fig = False):
	"""
	Plots luminosity distributions

	Inputs:
		list_in : list
			List of luminosity values of every reconstructed pixel of the generated boards
		list_gauss : list
			List of luminosity values of every reconstructed pixel of the reconstructed boards from gaussian fit
		list_lucy : list
			List of luminosity values of every reconstructed pixel of the reconstructed boards from RL reconstruction
		board_type : str
			Added effects type ('gauss', 'backgauss', 'poissgauss') used in saving the file
		save_fig : bool
			If True the plots will be saved
	"""

	plt.figure(figsize = [10,10], dpi = 100, layout = 'tight')
	bins = np.linspace(0, 25000, 2000)
	plt.hist(list_in, bins = bins, alpha = 0.5, histtype = 'stepfilled', linewidth = 1.5, label = "Starting Distribution")
	plt.hist(list_gauss, bins = bins, alpha = 0.5, histtype = 'step', linewidth = 1.5, label = "Gaussian Reconstruction")
	plt.hist(list_lucy, bins = bins, alpha = 0.5, histtype = 'step', linewidth = 1.5, label = "Lucy Reconstruction")
	plt.title("Number of pixels per luminosity interval", fontsize = 20)
	plt.xscale('log')
	plt.yscale('log')
	plt.legend(loc = 'best', fontsize = 15)
	plt.tick_params(labelsize = 15)
	if save_fig == True:
		plt.savefig(fname = board_type + str(sigma) + '_lumin.pdf', format = 'pdf')
	print(np.sum(list_in), np.sum(list_gauss), np.sum(list_lucy))
	plt.show()

	return


if __name__ == '__main__':
	tstart = time.time()
	lum_tresh = [0.1, 0.25, 0.5, 1]
	board_t = 'poissgauss'
	savef = True
	out_g, out_l, lumin_in, lumin_g, lumin_l = Completeness(niter = 5000, board_type = board_t, lumin_treshold = lum_tresh)
	tend = time.time()
	print((tend-tstart)/60)
	Visualise_Multipl(out_g, lumin_treshold = lum_tresh, board_type = board_t, rec_type = 'gauss', save_fig = savef)
	Visualise_Multipl(out_l, lumin_treshold = lum_tresh, board_type = board_t, rec_type = 'lucy', save_fig = savef)
	LuminDistr(lumin_in, lumin_g, lumin_l, board_type = board_t, save_fig = savef)