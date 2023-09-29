from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage
import os
from skimage import io
from tracefns import *

os.chdir('./28_marzo_2023')

image_names = ["sirius.fit", "sirius2.fit"]
dark_names = ["dark_1.fit", "dark_2.fit"]
flat_names = ["flat_1.fit", "flat_2.fit", "flat_3.fit", "flat_4.fit", "flat_5.fit"]

calibrator = np.flip(io.imread("Spectrum_calibration.png", as_gray = True), axis = 0)


################	GENERAL OBSERVATION DATA	################
ra_ICRS = 101.28715533
dec_ICRS = -16.71611586
phi = 43.722094
L = 10.4079
height = 4


def ImagePlot(fig_num, image, title = "", color_scale = 'linear', color_scheme = 'gray'):
	'''
	Plots a specified image

	Inputs:
		fig_num			Number handle of the figure
		image			Image to plot
		title			Title of the plot
		color_scale		Scale of the colormap
		color_scheme	Scheme of the colormap
	'''
	
	plt.figure(fig_num, dpi = 170, layout = 'tight')
	plt.imshow(image, cmap = color_scheme, origin = 'lower')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title(title)
	
	return


def SpectrumPlot(fig_num, x, y, xunit = "", title = ""):
	'''
	Plots a series of points

	Inputs:
		fig_num	Number handle of the figure
		x		Array of x
		y		Array of y
		title	Title of the plot
	'''

	plt.figure(fig_num, dpi = 170, layout = 'tight')
	plt.plot(x, y, '-k')
	plt.xlabel("$\lambda$ ["+xunit+"]")
	plt.ylabel("Intensity")
	plt.title(title)

	return


def AvgFields(file_names):
	'''
	Averages given images

	Inputs:
		file_names	File names of the images to average

	Outputs:
		data	Averaged image
	'''
	hdul = fits.open(file_names[0])
	data = hdul[0].data/(len(file_names)*float(hdul[0].header['EXPOSURE']))
	hdul.close()
	for i in range(1, len(file_names)):
		hdul = fits.open(file_names[i])
		data += hdul[0].data/(len(file_names)*float(hdul[0].header['EXPOSURE']))
		hdul.close()

	return data


def LinFit(x, a, b):
	'''
	Linear fit function

	Inputs:
		x	Data
		a	Offset
		b	Slope
	'''

	return a + b*x


def PolyFit(x, a, b, c):
	'''
	Polynomial fit function

	Inputs:
		x	Data
		a	Order 0 coefficient
		b	Order 1 coefficient
		c	Order 2 coefficient
	'''

	return a + b*x + c*(x**2)


def ExpFit(x, a, b, c):
	return np.exp(a*(x-b)) + c


def SubdivideBoard(image):
	'''
	Subdivides image in two horizontal sections, also cleans up  gradient leftovers which aren't of interest. Needed to find rotation angle from flat field.

	Inputs:
		image	Gradient of the flat field along X axis

	Outputs:
		board1/board2	Sections of the gradient image

	First the user selects a window around the two expected horizontal lines (see "CorrectRotation"), this is needed to eliminate any spurious points left over.
		Then the user is asked to set an X value at which the two board sections can be divided (such that a linear fit can be executed on the lines singularely).
	''' 

	x_min = np.zeros(2, dtype = int)
	x_max = np.zeros(2, dtype = int)
	y_min = np.zeros(2, dtype = int)
	y_max = np.zeros(2, dtype = int)

	ImagePlot(100, image)
	plt.show(block = False)

	print("\nSelect a window around each of the two lines:")

	for i in range(0,2):
		if i == 0:
			print("\tWindow for lower line")
		else:
			print("\tWindow for upper line")

		while True:
			print("\t\tSelect window along x axis:")
			y_min[i] = int(input("\t\t\tLower bound at x = "))
			y_max[i] = int(input("\t\t\tUpper bound at x = "))

			if (y_min[i] in range(0, len(image[0,:]))) and (y_max[i] in range(0, len(image[0,:]))) and (y_min[i] < y_max[i]):
				break
			else:
				print("\t\tValues not in bounds")

		while True:
			print("\t\tSelect window along y axis:")
			x_min[i] = int(input("\t\t\tLower bound at y = "))
			x_max[i] = int(input("\t\t\tUpper bound at y = "))

			if (x_min[i] in range(0, len(image[:,0]))) and (x_max[i] in range(0, len(image[:,0]))) and (x_min[i] < x_max[i]):
				break
			else:
				print("\t\tValues not in bounds")

	plt.close(100)

	plt.figure(101, figsize = [10,10], dpi = 100, layout = 'tight')
	plt.imshow(image, cmap = 'gray', origin = 'lower')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.colorbar()
	plt.axvline(x = y_min[0], linestyle = '--', color = 'r')
	plt.axvline(x = y_max[0], linestyle = '--', color = 'r')
	plt.axhline(y = x_min[0], linestyle = '--', color = 'r')
	plt.axhline(y = x_max[0], linestyle = '--', color = 'r')
	plt.axvline(x = y_min[1], linestyle = '--', color = 'g')
	plt.axvline(x = y_max[1], linestyle = '--', color = 'g')
	plt.axhline(y = x_min[1], linestyle = '--', color = 'g')
	plt.axhline(y = x_max[1], linestyle = '--', color = 'g')
	plt.show()
	plt.close(101)

	board1 = np.copy(image)
	board1[:x_min[0], :] = 0
	board1[x_max[0]:, :] = 0
	board1[:, :y_min[0]] = 0
	board1[:, y_max[0]:] = 0

	board2 = np.copy(image)
	board2[:x_min[1], :] = 0
	board2[x_max[1]:, :] = 0
	board2[:, :y_min[1]] = 0
	board2[:, y_max[1]:] = 0

	return (board1, board2)


def LineFit(board, FitFn, debug = False):
	'''
	Fits a line

	Inputs:
		board	Section of the image containing one horizontal line from the gradient of the flat field
		FitFn	Function for the fit
		debug	Debug option: if True the fit results are shown

	Outputs:
		pars[0]	Slope of the line
		pars[1]	Offset of the line
	'''

	xarr = np.where(board == 1)[0]
	yarr = np.where(board == 1)[1]

	y = np.unique(yarr)
	x = np.zeros(len(y))

	for i in range(len(y)):
		x[i] = np.mean(xarr[np.where(yarr == y[i])])

	[pars,covm] = curve_fit(FitFn, x, y)
	errs = np.sqrt(covm.diagonal())

	if debug == True:
		print("\nFit parameters:")
		print(pars, errs)

		plt.figure(200, figsize = [10,10], dpi = 100, layout = 'tight')
		plt.plot(y, x, '.k')
		lin = np.linspace(0, len(board[:,0]), len(board[:,0]))
		plt.plot(FitFn(lin, *pars), lin, 'r')
		plt.ylim(0, len(board[:,0]))
		plt.xlim(0, len(board[0,:]))
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.show()
		plt.close(200)

	return (pars[0], pars[1])


def FindRotation(image, debug = False):
	'''
	Finds angle needed to rectify the images
	
	Inputs:
		image	Image used to find the angle. The gradient along the X axis of the flat field is needed to use this method
		debug	Debug option: If true all information is shown

	Outputs:
		rot_ang	Angle to rotate the images

	First the two lines (see "CorrectRotation") are fit. From the parameters the bisector line is found.
		With the bisector line the rotation angle and the X coordinate corresponding to Y = 1 along the bisector are computed.
	'''

	board1, board2 = SubdivideBoard(image)

	q1, m1 = LineFit(board1, LinFit, debug)
	q2, m2 = LineFit(board2, LinFit, debug)

	x_mid = int(np.rint((q2 - q1)/(m1 - m2)))
	y_mid = int(np.rint(m1*x_mid + q1))
	vec1 = -np.array([1., m1])/np.sqrt(m1**2 + 1)
	vec2 = -np.array([1., m2])/np.sqrt(m2**2 + 1)
	vec = (vec1 + vec2)/np.linalg.norm(vec1 + vec2)
	m_mid = vec[1]/vec[0]
	q_mid = y_mid - m_mid*x_mid

	rot_ang = 0.

	if vec[0] < 0:
		rot_ang = -np.arccos(np.dot(vec, np.array([0,1])))*180/np.pi
	else:
		rot_ang = np.arccos(np.dot(vec, np.array([0,1])))*180/np.pi

	if debug == True:
		print("Rotation Angle: ", rot_ang)

	plt.figure(100, figsize = [10,10], dpi = 100, layout = 'tight')
	plt.imshow(image, cmap = 'gray', origin = 'lower')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.colorbar()
	lin = np.linspace(0, len(image[:,0]), len(image[:,0]))
	plt.plot(LinFit(lin, q1, m1), lin, 'r')
	plt.plot(LinFit(lin, q2, m2), lin, 'r')
	plt.plot(LinFit(lin, q_mid, m_mid), lin, 'r')
	plt.ylim(0, len(image[:,0]))
	plt.xlim(0, len(image[0,:]))
	plt.show()
	plt.close(100)

	return rot_ang


def CorrectRotation(flat, debug = False):
	'''
	Straightens spectrum and lamp using flat field.

	Imputs:
		flat	Flat field
		debug	Debug Option

	Outputs:
		rot_ang	Angle of rotation

	This method relies on the fact that in the flat field the slit's outline projection on the chip is clearly defined.

	The gradient of the flat field along the X axis is taken. This highlights the slit's outline projection.
		A threshold is input by the user to filter out small gradients and leave just the two upper and lower outlines.
		The two (more or less straight) outlines are fitted with a line (see "FindRotation" and "LineFit") in order to compute the bisector
			(which is used as the indicator for straightening).
		The angle of the rotation is computed from the bisector of the two lines.
	
	This process, albeit not as reliable as a full 3D deprojection, is simpler and works well enough and assures that at least the bisector line is perfectly straight.
	'''

	grad = abs(np.gradient(flat)[0])

	while True:
		print("\nRegulate intensity treshold (0-1 of the brightest pixel) until you see exactly two lines:")

		thresh = 0.2
		temp = np.copy(grad)
		temp[np.where(grad < np.max(grad)*thresh)] = 0
		temp[np.where(grad >= np.max(grad)*thresh)] = 1

		ImagePlot(100, temp)
		plt.show()
		plt.close(100)

		while True:
			print("\tLast treshold = %.1f" %(thresh))

			while True:
				thresh = float(input("\tNew threshold (0-1): "))

				if (thresh >= 0) and (thresh <= 1):
					break
				else:
					print("\tOut of bounds, must be more than 0 and less than 1")
			
			temp = np.copy(grad)
			temp[np.where(grad < np.max(grad)*thresh)] = 0
			temp[np.where(grad >= np.max(grad)*thresh)] = 1

			ImagePlot(101, temp)
			plt.show()
			plt.close(101)

			choice = input("\tTry new threshold? (y/n): ")

			if choice != "y":
				break
			print("")

		temp = np.copy(grad)
		temp[np.where(grad < np.max(grad)*thresh)] = 0
		temp[np.where(grad >= np.max(grad)*thresh)] = 1

		rot_ang = FindRotation(temp, debug)

		repeat = input("\nTry new fit? (y/n) ")
		if repeat != "y":
			break
	
	return rot_ang


def ExtractSpectrum(image, lamp):
	'''
	Gets the spectrum from within chosen boundaries averaged along the Y axis
	
	Inputs:
		image	Image of the measurement
		lamp	Image of the lamp

	Outputs:
		spectrum_meas	Extracted spectrum
		spectrum_lamp	Extracted lamp
		x_min			Lower X bound
		x_max			Higher X bound
		y_c				Center line over which the spectra are extracted

	The user chooses the bounds (x_min, x_max) and the center line (y_c) over which the spectra are extracted.
		The spectra are taken as centered around y_c and averaged over a range [y_c - width, y_c + width]

	'''

	x_min = 0
	x_max = 0
	y_c = 0
	while True:
		plt.figure(100, dpi = 170, layout = 'tight')
		plt.imshow(image, cmap = 'gray', origin = 'lower', norm = 'log')
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.ylim(0, len(image[:,0]))
		plt.xlim(0, len(image[0,:]))
		plt.show(block = False)

		print("\nSelect range over which the spectrum is taken:")

		x_min = int(input("\tLower bound at x = "))
		x_max = int(input("\tUpper bound at x = "))

		plt.close(100)

		if (x_min not in range(0, len(image[0,:]))) or (x_max not in range(0, len(image[0,:]))) or (x_min >= x_max):
			print("Values not in bounds")
			continue

		plt.figure(101, dpi = 170, layout = 'tight')
		plt.imshow(image, cmap = 'gray', origin = 'lower', norm = 'log')
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.axvline(x = x_min, linestyle = '--', color = 'r')
		plt.axvline(x = x_max, linestyle = '--', color = 'r')
		plt.show(block = False)

		y_c = int(input("\tSpectrum center at y = "))

		plt.close(101)

		if y_c not in range(0, len(image[:,0])):
			print("Value not in bounds")
			continue

		plt.figure(102, dpi = 170, layout = 'tight')
		plt.imshow(image, cmap = 'gray', origin = 'lower', norm = 'log')
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.axvline(x = x_min, linestyle = '--', color = 'r')
		plt.axvline(x = x_max, linestyle = '--', color = 'r')
		plt.axhline(y = y_c, linestyle = '--', color = 'g')
		plt.xlim(x_min - 100, x_max + 100)
		plt.ylim(y_c - 100, y_c + 100)
		plt.show(block = False)

		width = int(input("\tAverage over a half width dy = "))

		plt.close(102)

		if ((y_c + width) not in range(0, len(image[:,0]))) or ((y_c - width) not in range(0, len(image[:,0]))):
			print("Too wide")
			continue

		plt.figure(103, dpi = 170, layout = 'tight')
		plt.imshow(image, cmap = 'gray', origin = 'lower', norm = 'log')
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.colorbar()
		plt.axvline(x = x_min, linestyle = '--', color = 'r')
		plt.axvline(x = x_max, linestyle = '--', color = 'r')
		plt.axhline(y = y_c + width, linestyle = '--', color = 'g')
		plt.axhline(y = y_c - width, linestyle = '--', color = 'g')
		plt.xlim(x_min - 100, x_max + 100)
		plt.ylim(y_c - 100, y_c + 100)
		plt.show()
		plt.close(103)

		choice = input("\n\tChoose new bounds? (y/n) ")

		if choice != "y":
			break

	spectrum_meas = np.mean(image[(y_c - width):(y_c + width + 1), x_min:(x_max + 1)], axis=0)
	spectrum_lamp = np.mean(lamp[(y_c - width):(y_c + width + 1), x_min:(x_max + 1)], axis=0)

	return (spectrum_meas, spectrum_lamp, x_min, x_max, y_c, width)


def CalibrateSpectrum(spectrum_lamp, calibrator_img, pixels_array, FitFn, debug = False):
	'''
	Converts px to nm using the lamp spectrum

	Inputs:
		spectrum_lamp	Section of the lamp spectrum corresponding to the extracted spectrum
		calibrator_img	Lamp calibration image
		pixels_array	Array of the pixel position of the section of the spectrum
		FitFn			Fit function for px to wavelength conversion

	Outputs:
		pars	Conversion fit parameters

	The user is shown the extracted lamp spectrum toghether with the lamp calibration file and is asked to enter the x coord. in px.
		and the corresponding x coord in angstrom (as specified in cakibration image) together with an error for as many lines as the user wants.
		A fit is then applied to find the conversion between px and nm
	'''

	SpectrumPlot(101, pixels_array, spectrum_lamp, xunit = "arb. un.")
	ImagePlot(102, calibrator_img)
	plt.show(block = False)

	print("\nEnter values of peaks:")
	x = []
	y = []
	err = []
	while True:
		x.append(float(input("\tEnter line on extracted lamp spectrum: x = ")))
		y.append(float(input("\tEnter corresponding line on calibrator plot: x[angstrom] = ")))
		err.append(float(input("\t\tWith error: dx[angstrom] = ")))

		choice = ""
		while True:
			choice = input("\n\tEnter new line (y/n) or delete last line set (d)? (y/n/d) ")
			if choice == "y" or choice == "n" or choice == "d":
				break

		print("\n")

		if choice == "n":
			break
		if choice == "d":
			x.pop()
			y.pop()
			err.pop()

	plt.close('all')

	x = np.array(x)
	y = np.array(y)
	err = np.array(err)

	points = np.array([x[x.argsort()], y[x.argsort()], err[x.argsort()]])

	pars, covm = curve_fit(FitFn, points[0], points[1]/10., sigma = points[2])
	errs = np.sqrt(covm.diagonal())

	if debug == True:
		print("Fit results:")
		for i in range(len(pars)):
			print("\tParameter %i: %f pm %f" %(i, pars[i], errs[i]))

		plt.figure(1001, dpi = 170, layout = 'tight')
		plt.errorbar(x, y/10., yerr = err, marker = '.', linestyle = '', color = 'k')
		lin = np.linspace(np.min(x), np.max(x), 1000)
		plt.plot(lin, FitFn(lin, pars[0], pars[1], pars[2]))
		plt.xlabel("Px.")
		plt.ylabel("$\lambda$ [nm]")
		plt.show()
		plt.close(1001)

	choice = input("\nSave wavelength calibration? (y/n) ")
	if choice == "y":
		fname = input("Enter wavelength calibration file name: ")
		np.savetxt(fname+".txt", points.T, delimiter = '\t', fmt = '%.2f')

	return pars


def UseCalibrator(debug = False):
	'''
	Uses calibrator file to convert from px to nm

	Inputs:
		debug	Debug option

	Outputs:
		pars	Conversion fit parameters
	'''
	fname = input("\tEnter wavelength calibration file name: ")
	x, y, err = np.genfromtxt(fname+".txt", unpack = True)

	pars, covm = curve_fit(PolyFit, x, y/10., sigma = err)
	errs = np.sqrt(covm.diagonal())

	if debug == True:
		print("Fit results:")
		for i in range(len(pars)):
			print("\tParameter %i: %f pm %f" %(i, pars[i], errs[i]))

		plt.figure(1001, dpi = 170, layout = 'tight')
		plt.errorbar(x, y/10., yerr = err, marker = '.', linestyle = '', color = 'k')
		lin = np.linspace(np.min(x), np.max(x), 1000)
		plt.plot(lin, PolyFit(lin, pars[0], pars[1], pars[2]))
		plt.xlabel("Px.")
		plt.ylabel("$\lambda$ [nm]")
		plt.show()
		plt.close(1001)

	return pars


def GetSpectrum(image, lamp, calibrator_img, FitFn, debug = False):
	'''
	Extracts the calibrated spectrum

	Inputs:
		image			Image of the measurement
		lamp			Image of the lamp
		calibrator_img	Lamp calibration image
		FitFn			Fit function for px to wavelength conversion
		debug			Debug option

	Outputs:
		lam				Array of converted px to wavelength (nm)
		spectrum_meas	Measured spectrum (see "ExtractSpectrum")
		x_min			Output to be saved in calibration file (see "ExtractSpectrum")
		x_max			Output to be saved in calibration file (see "ExtractSpectrum")
		y_c				Output to be saved in calibration file (see "ExtractSpectrum")
		width			Output to be saved in calibration file (see "ExtractSpectrum")

	The spectrum and measured lamp are first extracted (see "ExtractSpectrum")
	Then the calibration is done to convert from px to nm (see "CalibrateSpectrum" or "UseCalibrator")
	'''

	spectrum_meas, spectrum_lamp, x_min, x_max, y_c, width = ExtractSpectrum(image, lamp)

	pixels = np.arange(x_min, x_max + 1, 1)

	pars = None
	choice = input("\nUse wavelength calibration file? (y/n) ")
	if choice == "y":
		pars = UseCalibrator(debug)
	else:
		pars = CalibrateSpectrum(spectrum_lamp, calibrator_img, pixels, FitFn, debug)

	lam = PolyFit(pixels, *pars)

	return (lam, spectrum_meas, x_min, x_max, y_c, width)


def Interactivity(image_names, lamp, flat, dark, FitFn, debug = False):
	'''
	Sets different procedures if a calibrator file is used or not

	Inputs:
		image_names	Image files' names
		lamp_names	Lamp file
		flat_names	Flat file
		dark_names	Dark file
		FitFn		Fit function for px to wavelength conversion
		Debug		Debug option

	Outputs:
		spectrum_meas1	Measured spectrum intensities for first spectrum (see "ExtractSpectrum")
		lam1			Measured spectrum wavelength array for first spectrum (see "GetSpectrum")
		spectrum_meas2	Measured spectrum intensities for second spectrum (see "ExtractSpectrum")
		lam2			Measured spectrum wavelength array for second spectrum (see "GetSpectrum")

	If no calibration file is used one can be created at the end of the process
	'''

	gain = np.mean(flat)/flat

	spectrum_meas1 = None
	lam1 = None
	spectrum_meas2 = None
	lam2 = None

	choice = input("Use spectrum extraction file? (y/n) ")
	if choice == "y":
		fname = input("\tEnter spectrum extraction file name: ")

		rot_ang1, x_min1, x_max1, y_c1, width1 = np.genfromtxt(fname+"1_config.txt", unpack = True, dtype = (float, int, int, int, int))
		rot_ang2, x_min2, x_max2, y_c2, width2 = np.genfromtxt(fname+"2_config.txt", unpack = True, dtype = (float, int, int, int, int))

		hdul1 = fits.open(image_names[0])
		hdul2 = fits.open(image_names[1])

		image1_raw = ndimage.rotate(hdul1[0].data/float(hdul1[0].header['EXPOSURE']), -rot_ang1, reshape = False)
		image1 = (image1_raw - dark)*gain
		image2 = ((hdul2[0].data/float(hdul2[0].header['EXPOSURE'])) - dark)*gain

		hdul1.close()
		hdul2.close()

		rot_temp = ndimage.rotate((image1_raw - dark), rot_ang1, reshape = False)
		rot_gain = ndimage.rotate(gain, rot_ang1, reshape = False)

		rot_img1 = ndimage.rotate(image1, rot_ang1, reshape = False)
		'''
		rot_dark = ndimage.rotate(dark, rot_ang1, reshape = False)
		rot_flat = ndimage.rotate(flat, rot_ang1, reshape = False)
		plt.figure(554, dpi = 170, layout = 'tight')
		plt.imshow(rot_gain, cmap = 'gray', origin = 'lower', norm = 'log')
		plt.title("mean(Flat)/Flat")
		plt.colorbar()
		
		plt.figure(555, dpi = 170, layout = 'tight')
		plt.subplot(2, 1, 1)
		plt.imshow(rot_img1[1055:1066, 850:1000], cmap = 'gray', origin = 'lower', norm = 'log')
		plt.colorbar()
		plt.subplot(2, 1, 2)
		plt.plot(np.arange(850, 1000, 1), np.mean(rot_img1[1055:1066, 850:1000], axis = 0))
		
		plt.figure(556, figsize = [10,10], dpi = 50, layout = 'tight')
		for i in range(850,2500):
			if i % 5 == 0:
				plt.plot(np.arange(1000, 1100, 1), fits.getdata(image_names[0], ext=0)[1000:1100, i])
				plt.ylabel("Immagine1 raw (già ruotata)")

		plt.figure(557, figsize = [10,10], dpi = 50, layout = 'tight')
		for i in range(850,2500):
			if i % 5 == 0:
				plt.plot(np.arange(1000, 1100, 1), rot_temp[1000:1100, i])
				plt.ylabel("Immagine1 - Dark (già ruotata)")
		
		plt.figure(558, figsize = [10,10], dpi = 50, layout = 'tight')
		for i in range(850,2500):
			if i % 5 == 0:
				plt.plot(np.arange(1000, 1150, 1), rot_img1[1000:1150, i])
				plt.ylabel("(Immagine1 - Dark)*(mean(Flat)/Flat) (già ruotata)")
		'''
		rot_img2 = ndimage.rotate(image2, rot_ang1, reshape = False)
		'''
		plt.figure(560, figsize = [10,10], dpi = 50, layout = 'tight')
		for i in range(850,2500):
			if i % 5 == 0:
				plt.plot(np.arange(1150, 1250, 1), rot_img2[1150:1250, i])
				plt.ylabel("(Immagine2 - Dark)*(mean(Flat)/Flat) (già ruotata)")
		plt.show()
		'''

		rot_lamp = ndimage.rotate(lamp, rot_ang1, reshape = False)

		spectrum_meas1 = np.mean(rot_img1[(y_c1 - width1):(y_c1 + width1 + 1), x_min1:(x_max1 + 1)], axis=0)
		spectrum_meas2 = np.mean(rot_img2[(y_c2 - width2):(y_c2 + width2 + 1), x_min2:(x_max2 + 1)], axis=0)

		spectrum_lamp1 = np.mean(rot_lamp[(y_c1 - width1):(y_c1 + width1 + 1), x_min1:(x_max1 + 1)], axis=0)

		pixels1 = np.arange(x_min1, x_max1 + 1, 1)
		pixels2 = np.arange(x_min2, x_max2 + 1, 1)

		pars = None
		choice = input("\nUse wavelength calibration file? (y/n) ")
		if choice == "y":
			pars = UseCalibrator(debug)
		else:
			pars = CalibrateSpectrum(spectrum_lamp1, calibrator, pixels1, FitFn, debug)

		lam1 = PolyFit(pixels1, *pars)
		lam2 = PolyFit(pixels2, *pars)
		
	else:
		rot_ang = CorrectRotation(flat, debug)

		hdul1 = fits.open(image_names[0])
		hdul2 = fits.open(image_names[1])

		image1_raw = ndimage.rotate(hdul1[0].data/float(hdul1[0].header['EXPOSURE']), -rot_ang, reshape = False)
		image1 = (image1_raw - dark)*gain
		image2 = ((hdul2[0].data/float(hdul2[0].header['EXPOSURE'])) - dark)*gain

		hdul1.close()
		hdul2.close()

		rot_img1 = ndimage.rotate(image1, rot_ang, reshape = False)
		rot_img2 = ndimage.rotate(image2, rot_ang, reshape = False)
		rot_lamp = ndimage.rotate(lamp, rot_ang, reshape = False)

		lam1, spectrum_meas1, x_min1, x_max1, y_c1, width1 = GetSpectrum(rot_img1, rot_lamp, calibrator, FitFn, debug)
		lam2, spectrum_meas2, x_min2, x_max2, y_c2, width2 = GetSpectrum(rot_img2, rot_lamp, calibrator, FitFn, debug)

		save = input("\nSave spectrum extraction file? (y/n) ")
		if save == "y":
			name = input("\tEnter spectrum extraction file name: ")
			np.savetxt(name+"1_config.txt", [np.array([rot_ang, x_min1, x_max1, y_c1, width1])], delimiter = '\t', fmt = ['%.4f', '%d', '%d', '%d', '%d'])
			np.savetxt(name+"2_config.txt", [np.array([rot_ang, x_min2, x_max2, y_c2, width2])], delimiter = '\t', fmt = ['%.4f', '%d', '%d', '%d', '%d'])

	return (spectrum_meas1, lam1, spectrum_meas2, lam2)


def InterpolateSpectrum(spectrum_meas, lam, bin_size):
	start = 0
	end = 0

	if (lam[0] % bin_size) > bin_size/2:
		start = ((lam[0] // bin_size) + 1)*bin_size
	else:
		start = (lam[0] // bin_size)*bin_size

	if (lam[-1] % bin_size) > bin_size/2:
		end = ((lam[-1] // bin_size) + 1)*bin_size
	else:
		end = (lam[-1] // bin_size)*bin_size

	x = np.arange(start, end + bin_size, bin_size)
	y = []

	for i in range(0,len(x)):
		y = np.append(y, np.mean(spectrum_meas[np.where((lam >= start + (i - 0.5)*bin_size) & (lam < start + (i + 0.5)*bin_size))]))
	y = np.array(y)

	return (x, y)


def ExtrapolateAirmass(wl1, sp1, wl2, sp2, a1, a2, bin_size, debug = False):
	low = np.max([np.min(wl1), np.min(wl2)])
	high = np.min([np.max(wl1), np.max(wl2)])

	ratio = (sp1[np.where((wl1 >= low) & (wl1 <= high))])/(sp2[np.where((wl2 >= low) & (wl2 <= high))])

	m1 = -2.5*np.log10(sp1[np.where((wl1 >= low) & (wl1 <= high))])
	m2 = -2.5*np.log10(sp2[np.where((wl2 >= low) & (wl2 <= high))])

	allpoints = np.sort(np.array(list(set(np.append(wl1, wl2)))))
	points = allpoints[np.where((allpoints >= low) & (allpoints <= high))]

	tau_fit = np.log(ratio)/(a2 - a1)

	pars, covm = curve_fit(ExpFit, points[np.where(points >= 415)], tau_fit[np.where(points >= 415)], p0 = (-0.01, 455, 0), bounds = ((-np.inf, 0, -np.inf), (0, np.inf, np.inf)))
	print(pars)

	tau = ExpFit(points, pars[0], pars[1], 0)
	T = np.exp(-tau)
	T2 = np.exp(-tau_fit)
	A = 1-T
	A2 = 1-T2

	intensity = (sp1[np.where((wl1 >= low) & (wl1 <= high))])*np.exp(tau*a1)
	intensity2 = (sp1[np.where((wl1 >= low) & (wl1 <= high))])*(ratio**(a1/(a2 - a1)))

	m0 = m1 - (2.5/np.log(10))*tau*a1

	K_B = (2.5*np.log10(np.sum(sp1[np.where((points >= 400) & (points <= 500))])/np.sum(intensity[np.where((points >= 400) & (points <= 500))])))/a1
	K_B2 = (2.5*np.log10(np.sum(sp2[np.where((points >= 400) & (points <= 500))])/np.sum(intensity[np.where((points >= 400) & (points <= 500))])))/a2
	K_V = (2.5*np.log10(np.sum(sp1[np.where((points >= 500) & (points <= 700))])/np.sum(intensity[np.where((points >= 500) & (points <= 700))])))/a1
	K_V2 = (2.5*np.log10(np.sum(sp2[np.where((points >= 500) & (points <= 700))])/np.sum(intensity[np.where((points >= 500) & (points <= 700))])))/a2
	K_R = (2.5*np.log10(np.sum(sp1[np.where(points >= 550)])/np.sum(intensity[np.where(points >= 550)])))/a1
	K_R2 = (2.5*np.log10(np.sum(sp2[np.where(points >= 550)])/np.sum(intensity[np.where(points >= 550)])))/a2
	print("Transmission Coefficient:")
	print("\tB: ",K_B, K_B2)
	print("\tV: ",K_V, K_V2)
	print("\tR: ",K_R, K_R2)

	mB = -2.5*np.log10(np.sum(intensity[np.where((points >= 400) & (points <= 500))]*bin_size/100))
	mV = -2.5*np.log10(np.sum(intensity[np.where((points >= 500) & (points <= 700))]*bin_size/200))
	mR = -2.5*np.log10(np.sum(intensity[np.where(points >= 550)]*bin_size/(np.max(points)-550)))
	print("Filter Magnitude:")
	print("\tB: ", mB)
	print("\tV: ", mV)
	print("\tR: ", mR)
	print("Color Indices:")
	print("\tB-V: ", mB-mV)
	print("\tB-R: ", mB-mR)
	print("\tV-R: ", mV-mR)

	if debug == True:
		plt.figure(102, dpi = 170, layout = 'tight')
		plt.plot(points, ratio, '.k')

		plt.figure(103, dpi = 170, layout = 'tight')
		plt.plot(points, tau_fit, '.k')
		lin = np.linspace(np.min(points), np.max(points), 1000)
		plt.plot(lin, ExpFit(lin, *pars))

		plt.figure(104, dpi = 170, layout = 'tight')
		plt.plot(points, intensity, 'k')
		plt.plot(points, intensity2, 'r')

		plt.figure(105, dpi = 170, layout = 'tight')
		plt.plot(points, A, 'k')
		plt.plot(points, A2, 'r')

		plt.figure(106, dpi = 170, layout = 'tight')
		plt.subplot(3, 1, 1)
		plt.plot(points, m1)
		plt.subplot(3, 1, 2)
		plt.plot(points, m2)
		plt.subplot(3, 1, 3)
		plt.plot(points, m0)

		plt.show()

	return


def FindAirmass(image_name):
	hdul = fits.open(image_name)
	date = hdul[0].header['DATE-OBS']
	print(date)

	yy = int(date[:4])
	mm = int(date[5:7])
	dd = int(date[8:10])
	hh = int(date[11:13])
	mn = int(date[14:16])
	ss = int(date[17:19])

	dayfrac = (hh + mn/60 + ss/3600)/24

	return ComputeAirmass(dayfrac, dd, mm, yy, ra_ICRS, dec_ICRS, phi, L, height)


dark = AvgFields(dark_names)
flat = AvgFields(flat_names) - dark
hdulamp = fits.open("sirius_lamp.fit")
lamp = (((hdulamp[0].data/float(hdulamp[0].header['EXPOSURE'])) - dark)*np.mean(flat))/flat


spectrum1, wavelength1, spectrum2, wavelength2 = Interactivity(image_names, lamp, flat, dark, PolyFit, debug = True)

wav1, spav1 = InterpolateSpectrum(spectrum1, wavelength1, 5)
wav2, spav2 = InterpolateSpectrum(spectrum2, wavelength2, 5)
'''
SpectrumPlot(1, wavelength1, spectrum1, xunit = "nm")
SpectrumPlot(2, wav1, spav1, xunit = 'nm')
SpectrumPlot(3, wavelength2, spectrum2, xunit = "nm")
SpectrumPlot(4, wav2, spav2, xunit = 'nm')
plt.show()
plt.close('all')
'''

airmass1 = FindAirmass(image_names[0])
airmass2 = FindAirmass(image_names[1])

print(airmass1, airmass2)

ExtrapolateAirmass(wav1, spav1, wav2, spav2, airmass1, airmass2, 5, debug = True)

