from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage
from scipy.interpolate import CubicSpline, UnivariateSpline
import os
from skimage import io
from tracefns import *
from astropy.nddata import CCDData
from ccdproc import Combiner, ccd_process, ImageFileCollection
from astropy import units as u


os.chdir('./28_marzo_2023')

image_names = ["sirius.fit", "sirius2.fit"]
dark_names = ["dark_1.fit", "dark_2.fit"]
flat_names = ["flat_1.fit", "flat_2.fit", "flat_3.fit", "flat_4.fit", "flat_5.fit"]

calibrator = np.flip(io.imread("Spectrum_calibration.png", as_gray = True), axis = 0)


################	GENERAL OBSERVATION DATA	################
phi = 43.722094
L = 10.4079
height = 4


def SpectrumPlot(x, y, xunit = "", ylabel = "Intensity [arb. un.]", title = ""):
	"""
	Plots a series of points

	Inputs:
		x : ndarray
			Array of x
		y : ndarray
			Array of y
		xunit : str
			Unit for wavelength
		ylabel : str
			Label for y axis
		title : str
			Title of the plot
	"""

	plt.figure(dpi = 150, layout = 'tight')
	plt.plot(x, y, '-k')
	plt.xlabel("$\lambda$ ["+xunit+"]")
	plt.ylabel(ylabel)
	plt.title(title)

	return


def LinFit(x, a, b):
	"""
	Linear fit function

	Inputs:
		x : ndarray
			Data
		a : float
			Offset
		b : float
			Slope
	"""

	return a + b*x


def PolyFit(x, a, b, c):
	"""
	Polynomial fit function

	Inputs:
		x : ndarray
			Data
		a : float
			Order 0 coefficient
		b : float
			Order 1 coefficient
		c : float
			Order 2 coefficient
	"""

	return a + b*x + c*(x**2)


def ExpFit(x, a, b, c):
	"""
	Exponential fit function

	Inputs:
		x : ndarray
			Data
		a : float
			Order 0 coefficient
		b : float
			Order 1 coefficient
		c : float
			Order 2 coefficient
	"""

	return np.exp(a*(x-b)) + c


def AtmosphericTau(x):
	"""
	Optical depth from https://www.aanda.org/articles/aa/full_html/2012/07/aa19040-12/aa19040-12.html

	Inputs:
	x : ndarray
		Array of wavelength in nm
	"""

	return 0.00864*((x/1000)**(-(3.916 + (0.074*(x/1000)) + (50/x))))


def FitCenter(image, x_min, x_max, debug = False):
	"""
	Finds rotation angle by fitting a line through brightest pixels in cross-dispersion and rectifying

	Inputs:
		image : ndarray
			Cleaned Science Frame (see "FineRotate()")
		x_min : int
			Lower bound along x axis for the fit
		x_max : int
			Upper bound along x axis for the fit
		debug : bool
			Debug Option: shows additional info

	Outputs:
		rot_ang : float
			Angle to straighten spectrum
	"""

	x = np.linspace(x_min, x_max, 100, dtype = 'int')
	y = []
	yerrs = []

	for i in range(len(x)):
		y = np.append(y, np.where(image[:, x[i]] == np.max(image[:, x[i]]))[0])
		max_index = int(y[-1])
		low_err = abs(max_index - np.where(image[:max_index, x[i]] <= 0.5*np.max(image[:, x[i]]))[0])[-1]
		high_err = np.where(image[max_index:, x[i]] <= 0.5*np.max(image[:, x[i]]))[0][0]
		yerrs = np.append(yerrs, (low_err + high_err)/2)

	y = np.array(y)
	yerrs = np.array(yerrs)

	pars, covm = curve_fit(LinFit, x, y, sigma = yerrs)
	errs = np.sqrt(covm.diagonal())

	vec = np.array([1., pars[1]])/np.sqrt(pars[1]**2 + 1)

	rot_ang = None

	if vec[1] < 0:
		rot_ang = -np.arccos(np.dot(vec, np.array([1,0])))*180/np.pi
	else:
		rot_ang = np.arccos(np.dot(vec, np.array([1,0])))*180/np.pi

	if debug == True:
		plt.figure(dpi = 150, layout = 'tight')
		plt.title("100 cross-dispersion profiles from inside the spectrum")
		for i in range(len(x)):
			plt.plot(image[:,x[i]])

		print("\n\tFit Results: q = %f pm %f, m = %f pm %f" %(pars[0], errs[0], pars[1], errs[1]))
		print("\tRotation Angle = %f" %(rot_ang))

		plt.figure(dpi = 150, layout = 'tight')
		plt.title("Fitted line from spectrum")
		plt.errorbar(x, y, yerrs, marker = '.', color = 'b', label = "Brightest pixel and FWHM errorbar")
		plt.imshow(image, cmap = 'gray', origin = 'lower', norm = 'log')
		plt.xlabel("X")
		plt.ylabel("Y")
		lin = np.arange(0, len(image[0,:]), 1)
		liny = LinFit(lin, *pars)
		plt.plot(lin, liny, 'r', label = "Fitted Line")
		plt.ylim(0, len(image[:,0]))
		plt.xlim(0, len(image[0,:]))
		plt.legend(loc = 'best')
		plt.show()
		plt.close('all')

	return rot_ang


def FineRotate(image, debug = False):
	"""
	Rotates Science Frame so that the spectrum is straight

	Inputs:
		image : ndarray
			Science Frame
		debug : bool
			Debug Option: shows additional info
	"""

	y_min = None
	y_max = None
	x_min = None
	x_max = None

	plt.figure(dpi = 150, layout = 'tight')
	plt.title("Science Frame")
	plt.imshow(image, cmap = 'gray', origin = 'lower', norm = 'log')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.ylim(0, len(image[:,0]))
	plt.xlim(0, len(image[0,:]))
	plt.show(block = False)

	rot_ang = None

	while True:
		print("\nSelect window inside which the spectrum lies:")

		while True:
			print("\t\tSelect window along x axis:")
			y_min = int(input("\t\t\tLower bound at x = "))
			y_max = int(input("\t\t\tUpper bound at x = "))

			if (y_min in range(0, len(image[0,:]))) and (y_max in range(0, len(image[0,:]))) and (y_min < y_max):
				break
			else:
				print("\t\tValues not in bounds")

		while True:
			print("\t\tSelect window along y axis:")
			x_min = int(input("\t\t\tLower bound at y = "))
			x_max = int(input("\t\t\tUpper bound at y = "))

			if (x_min in range(0, len(image[:,0]))) and (x_max in range(0, len(image[:,0]))) and (x_min < x_max):
				break
			else:
				print("\t\tValues not in bounds")

		plt.close('all')

		img_cut = np.zeros(np.shape(image))
		img_cut[x_min:x_max + 1, y_min:y_max + 1] = image[x_min:x_max + 1, y_min:y_max + 1]

		rot_ang = FitCenter(img_cut, y_min, y_max, debug)

		choice = input("\n\tTry fit with different parameters? (y/n) ")
		if choice == 'n':
			break

	return rot_ang


def ExtractSpectrum(image):
	"""
	Extracts the spectrum from within chosen boundaries (averaged along the Y axis)
	
	Inputs:
		image : ndarray
			Science Frame

	Outputs:
		x_min : int
			Lower bound along x axis from which the spectra are extracted
		x_max : int
			Higher bound along x axis to which the spectra are extracted
		y_c : int
			Center line along y axis over which the spectra are extracted
		width : int
			Half-width along y axis over which the spectra are averaged

	The spectra will be taken from x_min to x_max, centered around y_c and averaged over a range [y_c - width, y_c + width]

	"""

	x_min = None
	x_max = None
	y_c = None
	width = None

	while True:
		plt.figure(dpi = 150, layout = 'tight')
		plt.title("Science Frame")
		plt.imshow(image, cmap = 'gray', origin = 'lower', norm = 'log')
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.ylim(0, len(image[:,0]))
		plt.xlim(0, len(image[0,:]))
		plt.show(block = False)

		print("\nSelect range over which the spectrum is taken:")

		x_min = int(input("\tLower bound at x = "))
		x_max = int(input("\tUpper bound at x = "))

		plt.close('all')

		if (x_min not in range(0, len(image[0,:]))) or (x_max not in range(0, len(image[0,:]))) or (x_min >= x_max):
			print("Values not in bounds")
			continue

		plt.figure(dpi = 150, layout = 'tight')
		plt.title("Science Frame")
		plt.imshow(image, cmap = 'gray', origin = 'lower', norm = 'log')
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.axvline(x = x_min, linestyle = '--', color = 'r')
		plt.axvline(x = x_max, linestyle = '--', color = 'r')
		plt.show(block = False)

		y_c = int(input("\tSpectrum center at y = "))

		plt.close('all')

		if y_c not in range(0, len(image[:,0])):
			print("Value not in bounds")
			continue

		plt.figure(dpi = 150, layout = 'tight')
		plt.title("Science Frame")
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

		plt.close('all')

		if ((y_c + width) not in range(0, len(image[:,0]))) or ((y_c - width) not in range(0, len(image[:,0]))):
			print("Too wide")
			continue

		plt.figure(dpi = 150, layout = 'tight')
		plt.title("Science Frame with final selected region")
		plt.imshow(image, cmap = 'gray', origin = 'lower', norm = 'log')
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.axvline(x = x_min, linestyle = '--', color = 'r')
		plt.axvline(x = x_max, linestyle = '--', color = 'r')
		plt.axhline(y = y_c + width, linestyle = '--', color = 'g')
		plt.axhline(y = y_c - width, linestyle = '--', color = 'g')
		plt.xlim(x_min - 100, x_max + 100)
		plt.ylim(y_c - 100, y_c + 100)
		plt.show()
		plt.close('all')

		choice = input("\n\tChoose new bounds? (y/n) ")

		if choice != "y":
			break

	return (x_min, x_max, y_c, width)


def CalibrateSpectrum(spectrum_lamp, calibrator_img, pixels_array, FitFn, debug = False):
	"""
	Converts px to nm using the lamp spectrum

	Inputs:
		spectrum_lamp : ndarray
			Section of the lamp spectrum corresponding to the extracted spectrum
		calibrator_img : image
			Lamp calibration image
		pixels_array : ndarray
			Array of the pixel position of the section of the spectrum
		FitFn : function
			Fit function for px to wavelength conversion
		debug : bool
			Debug Option: shows additional info

	Outputs:
		pars : ndarray
			Conversion fit parameters
		points : ndarray
			Calibration points set

	The user is shown the extracted lamp spectrum toghether with the lamp calibration image and is asked to enter the x coord. in px.
		and the corresponding x coord in angstrom (as specified in calibration image) together with an error for as many lines as the user wants.
		A fit is then applied to find the conversion between px and nm.
	"""

	SpectrumPlot(pixels_array, spectrum_lamp, xunit = "px")

	plt.figure(dpi = 150, layout = 'tight')
	plt.imshow(calibrator_img, cmap = 'gray', origin = 'lower')
	plt.xlabel("X")
	plt.ylabel("Y")
	
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
			print("\tParameter %i: %f pm %f" %(i+1, pars[i], errs[i]))

		plt.figure(dpi = 150, layout = 'tight')
		plt.title("Calibration points with fit")
		plt.errorbar(x, y/10., yerr = err, marker = '.', linestyle = '', color = 'k', label = "Calibration Points")
		lin = np.linspace(np.min(x), np.max(x), 1000)
		plt.plot(lin, FitFn(lin, *pars), label = "Fitted Polynomial")
		plt.xlabel("$\lambda$ [px]")
		plt.ylabel("$\lambda$ [nm]")
		plt.legend(loc = 'best')
		plt.show()
		plt.close('all')

	return (pars, points)


def UseCalibrator(fname, FitFunc, debug = False):
	"""
	Uses calibrator file to convert from px to nm

	Inputs:
		fname : str
			Name of the calibration file
		FitFunc : function
			Polynomial function used for px to nm conversion
		debug : bool
			Debug Option: shows additional info

	Outputs:
		pars : ndarray
			Conversion fit parameters
	"""
	x, y, err = np.genfromtxt(fname, unpack = True)

	pars, covm = curve_fit(PolyFit, x, y/10., sigma = err)
	errs = np.sqrt(covm.diagonal())

	if debug == True:
		print("\nFit results:")
		for i in range(len(pars)):
			print("\tParameter %i: %f pm %f" %(i+1, pars[i], errs[i]))

		plt.figure(dpi = 150, layout = 'tight')
		plt.title("Calibration points with fit")
		plt.errorbar(x, y/10., yerr = err, marker = '.', linestyle = '', color = 'k', label = "Calibration Points")
		lin = np.linspace(np.min(x), np.max(x), 1000)
		plt.plot(lin, FitFunc(lin, *pars), label = "Fitted Polynomial")
		plt.xlabel("$\lambda$ [px]")
		plt.ylabel("$\lambda$ [nm]")
		plt.legend(loc = 'best')
		plt.show()
		plt.close('all')

	return pars


def GetSpectrum(image, lamp, calibrator_img, FitFn, debug = False):
	"""
	Extracts the calibrated spectrum

	Inputs:
		image : ndarray
			Science Frame
		lamp : ndarray
			Lamp Frame
		calibrator_img : image
			Calibration image
		FitFn : function
			Polynomial function used for px to nm conversion
		debug : bool
			Debug Option: shows additional info

	Outputs:
		lam : ndarray
			Array of converted px to wavelength (nm)
		spectrum_meas : ndarray
			Extracted spectrum
		rot_ang : float
			Angle to straighten spectrum (see "FineRotate()")
		x_min : int
			Output to be saved in calibration file (see "ExtractSpectrum()")
		x_max : int
			Output to be saved in calibration file (see "ExtractSpectrum()")
		y_c : int
			Output to be saved in calibration file (see "ExtractSpectrum()")
		width : int
			Output to be saved in calibration file (see "ExtractSpectrum()")
		points : ndarray
			Calibration points set (see "CalibrateSpectrum()")
	
	The angle that straightens the spectrum is first found (see "FineRotate()").
	Then the spectrum and lamp are rotated.
	The extraction bounds are found (see "ExtractSpectrum()") and the spectra extracted.
	The calibration is done to convert from px to nm (see "CalibrateSpectrum()")
	"""

	rot_ang = FineRotate(image, debug)

	image_rot = ndimage.rotate(image, rot_ang, reshape = False)
	lamp_rot = ndimage.rotate(lamp, rot_ang, reshape = False)

	if debug == True:
		plt.figure(dpi = 150, layout = 'tight')
		plt.title("Rotated Science Frame")
		plt.imshow(image_rot, cmap = 'gray', origin = 'lower')
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.ylim(0, len(image[:,0]))
		plt.xlim(0, len(image[0,:]))
		plt.show()
		plt.close('all')

	x_min, x_max, y_c, width = ExtractSpectrum(image_rot)

	spectrum_meas = np.mean(image_rot[(y_c - width):(y_c + width + 1), x_min:(x_max + 1)], axis=0)
	spectrum_lamp = np.mean(lamp_rot[(y_c - width):(y_c + width + 1), x_min:(x_max + 1)], axis=0)

	pixels = np.arange(x_min, x_max + 1, 1)

	pars, points = CalibrateSpectrum(spectrum_lamp, calibrator_img, pixels, FitFn, debug)

	lam = PolyFit(pixels, *pars)

	return (lam, spectrum_meas, rot_ang, x_min, x_max, y_c, width, points)


def Interactivity(image_names, lamp, flat, dark, FitFn, debug = False):
	"""
	Sets different procedures if a calibrator file is used or not and extracts spectra

	Inputs:
		image_names : list
			Image files' names
		lamp_names : ndarray
			Lamp Frame
		flat : ndarray
			Master Flat Frame
		dark : ndarray
			Master Dark Frame
		FitFn : function
			Fit function for px to wavelength conversion
		debug : bool
			Debug Option: shows additional info

	Outputs:
		spectrum_meas1 : ndarray
			Measured spectrum intensities for first spectrum
		lam1 : ndarray
			Measured spectrum wavelength array for first spectrum
		spectrum_meas2 : ndarray
			Measured spectrum intensities for second spectrum
		lam2 : ndarray
			Measured spectrum wavelength array for second spectrum
		ra_ICRS : float
			Object ra coordinate in degrees in ICRS frame
		dec_ICRS : float
			Object dec coordinate in degrees in ICRS frame

	If no calibration file is used one can be created at the end of the process
	"""

	hdul_dark = fits.open(dark_names[0])
	texp_dark = float(hdul_dark[0].header['EXPOSURE'])

	spectrum_meas1 = None
	lam1 = None
	spectrum_meas2 = None
	lam2 = None

	choice = input("Use spectrum extraction file? (y/n) ")
	if choice == "y":
		fname = input("\tEnter spectrum extraction file name: ")

		ra_ICRS, dec_ICRS, rot_ang1, x_min1, x_max1, y_c1, width1 = np.genfromtxt(fname+"1_config.txt", unpack = True, dtype = (float, float, float, int, int, int, int))
		ra_ICRS, dec_ICRS, rot_ang2, x_min2, x_max2, y_c2, width2 = np.genfromtxt(fname+"2_config.txt", unpack = True, dtype = (float, float, float, int, int, int, int))

		hdul1 = fits.open(image_names[0])
		rotated_ang = float(hdul1[0].header['ROT_ANG'])
		texp_img1 = float(hdul1[0].header['EXPOSURE'])
		img1_raw = np.asarray(ccd_process(CCDData(ndimage.rotate(fits.getdata(image_names[0], ext=0), -rotated_ang, reshape = False), unit = "adu"), dark_frame = dark, dark_exposure = texp_dark*u.second, data_exposure = texp_img1*u.second,  gain_corrected = False))

		hdul2 = fits.open(image_names[1])
		texp_img2 = float(hdul2[0].header['EXPOSURE'])
		img2_raw = np.asarray(ccd_process(CCDData(fits.getdata(image_names[1], ext=0), unit = "adu"), dark_frame = dark, dark_exposure = texp_dark*u.second, data_exposure = texp_img2*u.second, gain_corrected = False))

		rot_img1 = ndimage.rotate(img1_raw, rot_ang1, reshape = False)
		rot_img2 = ndimage.rotate(img2_raw, rot_ang2, reshape = False)

		change = input("\nChange spectrum extraction? (y/n) ")
		if change == "y":
			print("Old settings for spectrum 1:")
			print("\tx_min = %i \n\tx_max = %i \n\ty_center = %i \n\thalf-width = %i" %(x_min1, x_max1, y_c1, width1))
			x_min1, x_max1, y_c1, width1 = ExtractSpectrum(rot_img1)
			print("\nOld settings for spectrum 2:")
			print("\tx_min = %i \n\tx_max = %i \n\ty_center = %i \n\thalf-width = %i" %(x_min2, x_max2, y_c2, width2))
			x_min2, x_max2, y_c2, width2 = ExtractSpectrum(rot_img2)

		spectrum_meas1 = np.mean(rot_img1[(y_c1 - width1):(y_c1 + width1 + 1), x_min1:(x_max1 + 1)], axis=0)
		spectrum_meas2 = np.mean(rot_img2[(y_c2 - width2):(y_c2 + width2 + 1), x_min2:(x_max2 + 1)], axis=0)

		pixels1 = np.arange(x_min1, x_max1 + 1, 1)
		pixels2 = np.arange(x_min2, x_max2 + 1, 1)

		pars1 = UseCalibrator(fname+"1_calibr.txt", FitFn, debug)
		pars2 = UseCalibrator(fname+"2_calibr.txt", FitFn, debug)

		lam1 = PolyFit(pixels1, *pars1)
		lam2 = PolyFit(pixels2, *pars2)
		
	else:
		hdul1 = fits.open(image_names[0])
		rotated_ang = float(hdul1[0].header['ROT_ANG'])
		img1_raw = np.asarray(ccd_process(CCDData(ndimage.rotate(fits.getdata(image_names[0], ext=0), -rotated_ang, reshape = False), unit = "adu"), dark_frame = dark, master_flat = flat, dark_exposure = 20*u.second, data_exposure = 10*u.second, exposure_unit = u.second, gain_corrected = False))

		hdul2 = fits.open(image_names[1])
		img2_raw = np.asarray(ccd_process(CCDData(fits.getdata(image_names[1], ext=0), unit = "adu"), dark_frame = dark, master_flat = flat, dark_exposure = 20*u.second, data_exposure = 5*u.second, exposure_unit = u.second, gain_corrected = False))

		lam1, spectrum_meas1, rot_ang1, x_min1, x_max1, y_c1, width1, points1 = GetSpectrum(img1_raw, lamp, calibrator, FitFn, debug)
		lam2, spectrum_meas2, rot_ang2, x_min2, x_max2, y_c2, width2, points2 = GetSpectrum(img2_raw, lamp, calibrator, FitFn, debug)

		ra_ICRS = None
		dec_ICRS = None
		print("Enter object coordinates in ICRS frame:")
		while True:
			while True:
				ra_ICRS = float(input("\tEnter Object RA (deg): "))
				if (ra_ICRS >= 0) and (ra_ICRS < 360):
					break
				else:
					print("\tValue not in bounds [0,360)")

			while True:
				dec_ICRS = float(input("\n\tEnter Object Dec (deg): "))
				if (dec_ICRS >= -90) and (dec_ICRS <= 90):
					break
				else:
					print("\tValue not in bounds [-90,90]")

			choice = input("Change coordinates? (y/n) ")
			if input == 'n':
				break

		save = input("\nSave spectrum extraction file? (y/n) ")
		if save == "y":
			name = input("\tEnter spectrum extraction file name: ")
			np.savetxt(name+"1_config.txt", [np.array([ra_ICRS, dec_ICRS, rot_ang1, x_min1, x_max1, y_c1, width1])], delimiter = '\t', fmt = ['%.6f', '%.6f', '%.4f', '%d', '%d', '%d', '%d'])
			np.savetxt(name+"1_calibr.txt", points1.T, delimiter = '\t', fmt = '%.2f')
			np.savetxt(name+"2_config.txt", [np.array([ra_ICRS, dec_ICRS, rot_ang2, x_min2, x_max2, y_c2, width2])], delimiter = '\t', fmt = ['%.6f', '%.6f', '%.4f', '%d', '%d', '%d', '%d'])
			np.savetxt(name+"2_calibr.txt", points2.T, delimiter = '\t', fmt = '%.2f')

	return (spectrum_meas1, lam1, spectrum_meas2, lam2, ra_ICRS, dec_ICRS)


def BinSpectrum(wavelength, spectrum, bin_size):
	"""
	Bins spectrum

	Inputs:
		wavelength : ndarray
			Spectrum wavelength array
		spectrum : ndarray
			Spectrum intensity array
		bin_size : int
			Bin size for the binning

	Outputs:
		x : ndarray
			Binned wavelength array
		y : ndarray
			Binned intensity array
	"""

	start = 0
	end = 0

	if (wavelength[0] % bin_size) > bin_size/2:
		start = ((wavelength[0] // bin_size) + 1)*bin_size
	else:
		start = (wavelength[0] // bin_size)*bin_size

	if (wavelength[-1] % bin_size) > bin_size/2:
		end = ((wavelength[-1] // bin_size) + 1)*bin_size
	else:
		end = (wavelength[-1] // bin_size)*bin_size

	x = np.arange(start, end + bin_size, bin_size)
	y = []

	for i in range(0,len(x)):
		y = np.append(y, np.mean(spectrum[np.where((wavelength >= start + (i - 0.5)*bin_size) & (wavelength < start + (i + 0.5)*bin_size))]))
	y = np.array(y)

	return (x, y)


def InterpolateSpectra(wavelength1, spectrum1, wavelength2, spectrum2, bin_size):
	"""
	Interpolates both sirius spectra at the same time

	Inputs:
		wavelength1 : ndarray
			Spectrum 1 wavelength array
		spectrum1 : ndarray
			Spectrum 1 intensity array
		wavelength2 : ndarray
			Spectrum 2 wavelength array
		spectrum2 : ndarray
			Spectrum 2 intensity array
		bin_size : int
			Bin size for the binning

	Outputs:
		out_wave : ndarray
			Interpolated wavelength array for both spectra
		out_spectrum1 : ndarray
			Interpolated spectrum 1 intensity array
		out_spectrum2 : ndarray
			Interpolated spectrum 2 intensity array
	"""

	wav1, spav1 = BinSpectrum(wavelength1, spectrum1, bin_size)
	wav2, spav2 = BinSpectrum(wavelength2, spectrum2, bin_size)

	low = np.max([np.min(wavelength1), np.min(wavelength2)])
	high = np.min([np.max(wavelength1), np.max(wavelength2)])

	out_wave = wav1[np.where((wav1 >= low) & (wav1 <= high))]
	out_spectrum1 = spav1[np.where((wav1 >= low) & (wav1 <= high))]
	out_spectrum2 = spav2[np.where((wav2 >= low) & (wav2 <= high))]

	return (out_wave, out_spectrum1, out_spectrum2)


def FindAirmass(image_name, ra_ICRS, dec_ICRS):
	"""
	Finds airmass for object at observation time

	Inputs:
		image_name : str
			Science Frame file name
		ra_ICRS : float
			Object ra coordinate in degrees in ICRS frame
		dec_ICRS : float
			Object dec coordinate in degrees in ICRS frame

	Outputs:
		airmass : float
			Airmass value
	"""

	hdul = fits.open(image_name)
	date = hdul[0].header['DATE-OBS']

	yy = int(date[:4])
	mm = int(date[5:7])
	dd = int(date[8:10])
	hh = int(date[11:13])
	mn = int(date[14:16])
	ss = int(date[17:19])

	dayfrac = (hh + mn/60 + ss/3600)/24

	airmass = ComputeAirmass(dayfrac, dd, mm, yy, ra_ICRS, dec_ICRS, phi, L, height)

	return airmass


def Response(wavelength, spectrum1, spectrum2, spectrum3, airmass1, airmass2, texp1, texp2, debug = False):
	"""
	Computes atmospheric transmission at 1 airmass and instrument response at 1s exposure time

	Inputs:
		wavelength : ndarray
			Interpolated wavelength array for all spectra
		spectrum1 : ndarray
			Interpolated spectrum 1 intensity array
		spectrum2 : ndarray
			Interpolated spectrum 2 intensity array
		spectrum3 : ndarray
			Interpolated spectrum 3 intensity array
		airmass1 : float
			Airmass for spectrum 1
		airmass2 : float
			Airmass for spectrum 2
		texp1 : float
			Exposure time for spectrum 1
		texp2 : float
			Exposure time for spectrum 2
		debug : bool
			Debug Option: shows additional info

	Outputs:
		trans_0 : ndarray
			Atmospheric transmission coefficient at 1 airmass
		instr_0 : ndarray
			Instrument response coefficient at 1s exposure time
	"""

	ratio = spectrum1/spectrum2

	tau_prime = np.log(ratio)/(airmass2 - airmass1)

	pars, covm = curve_fit(ExpFit, wavelength, tau_prime, p0 = (-0.01, 450, -0.2), bounds = ((-np.inf, 0, -np.inf),(0, np.inf, np.inf)))
	errs = np.sqrt(covm.diagonal())

	tau = ExpFit(wavelength, pars[0], pars[1], 0)

	trans_0 = np.exp(-tau)

	A1 = 1
	A2 = (texp1/texp2)*np.exp(pars[2]*(airmass1 - airmass2))

	instr_0_1 = spectrum1/(spectrum3*(trans_0**airmass1)*A1*texp1)
	instr_0_2 = spectrum2/(spectrum3*(trans_0**airmass2)*A2*texp2)
	spl0 = UnivariateSpline(wavelength, np.mean(np.vstack((instr_0_1, instr_0_2)), axis = 0))
	spl0.set_smoothing_factor(0.00001)
	instr_0 = spl0(wavelength)

	if debug == True:
		plt.figure(dpi = 150, layout = 'tight')
		plt.title("Intensity Ratio")
		plt.plot(wavelength, ratio, '.k', label = "$I_1 / I_2$")
		plt.legend(loc = 'best')

		plt.figure(dpi = 150, layout = 'tight')
		plt.title("Optical Depth")
		plt.plot(wavelength, tau, 'r', label = r"Corrected $\tau$")
		plt.plot(wavelength, tau_prime, '.k', label = r"$\tau'$")
		lin = np.linspace(wavelength[0], wavelength[-1], 1000)
		plt.plot(lin, ExpFit(lin, *pars), 'b', label = r"Fit over $\tau'$")
		plt.plot(lin, AtmosphericTau(lin), '--g', label = r"Reference $\tau$")
		plt.legend(loc = 'best')

		plt.figure(dpi = 150, layout = 'tight')
		plt.title("Atmospheric Transmittance")
		plt.plot(wavelength, trans_0, 'r', label = "Computed Transmittance")
		plt.plot(lin, np.exp(-AtmosphericTau(lin)), '--g', label = "Reference Transmittance")
		plt.legend(loc = 'best')

		plt.figure(dpi = 150, layout = 'tight')
		plt.title("Instrumental Response")
		plt.plot(wavelength, instr_0_1, 'r', label = "Instrumental Response from spectrum 1")
		plt.plot(wavelength, instr_0_2, 'b', label = "Instrumental Response from spectrum 2")
		plt.plot(wavelength, instr_0, '--k', label = "Averaged and smoothed Instrumental Response")
		plt.legend(loc = 'best')

		plt.figure(dpi = 150, layout = 'tight')
		plt.title("Total Response Function at 1 airmass and 1s exposure")
		plt.plot(wavelength, trans_0*instr_0, label = "Total Response Function")
		plt.legend(loc = 'best')

		plt.show()
		plt.close('all')

	resp_array = np.vstack((wavelength, trans_0, instr_0))

	choice = input("\nSave response functions? (y/n) ")
	if choice == 'y':
		np.savetxt("response.txt", resp_array.T, delimiter = '\t', fmt = ['%i', '%.8f', '%.8f'], header = "Colums are [wavelength, atmospheric transmission (airmass = 1), instrument response (exposure time = 0)]")

	return trans_0, instr_0


def CheckBack(wavelength, spectrum, vega, airmass, texp):
	"""
	Checks correct computation of response functions by comparing corrected spectrum and reference Vega spectrum

	Inputs:
		wavelength : ndarray
			Interpolated wavelength array for both spectra
		spectrum : ndarray
			Interpolated spectrum intensity array
		vega : ndarray
			Interpolated Vega spectrum intensity array
		airmass : float
			Spectrum airmass
		texp : float
			Spectrum exposure time
	"""

	resp_wave, trans_0, instr_0 = np.genfromtxt("response.txt", unpack = True, dtype = (int, float, float))

	low = np.max([np.min(wavelength), np.min(resp_wave)])
	high = np.min([np.max(wavelength), np.max(resp_wave)])

	mask_img = (wavelength >= low) & (wavelength <= high)
	mask_resp = (resp_wave >= low) & (resp_wave <= high)

	spectrum_corr = spectrum[mask_img]/(instr_0[mask_resp]*texp*(trans_0[mask_resp]**airmass))

	plt.figure(dpi = 150, layout = 'tight')
	plt.title("Corrected Spectrum and Vega (used as reference)")
	plt.plot(wavelength[mask_img], spectrum_corr, 'k', label = "Corrected Spectrum")
	plt.plot(wavelength[mask_img], vega[mask_img], 'r', label = "Vega")
	plt.legend(loc = 'best')
	plt.xlabel(r"$\lambda$ [nm]")
	plt.ylabel(r"Flux [erg $s^{-1}$ $cm^{-2}$ A $10^{16}$]")
	plt.show()
	return



if __name__ == '__main__':
	hdul_dark = fits.open(dark_names[0])
	texp_dark = float(hdul_dark[0].header['EXPOSURE'])

	dark_list = []
	for i in range(len(dark_names)):
		dark_list.append(CCDData.read(dark_names[i], unit = "adu"))
	combiner_dark = Combiner(dark_list)

	scaling_func = lambda arr: 1/np.ma.average(arr)
	combiner_dark.scaling = scaling_func  
	master_dark = combiner_dark.average_combine()

	hdul_flat = fits.open(flat_names[0])
	texp_flat = float(hdul_flat[0].header['EXPOSURE'])

	flat_list = []
	for i in range(len(flat_names)):
		flat_list.append(ccd_process(CCDData.read(flat_names[i], unit = "adu"), dark_frame = master_dark, dark_exposure = texp_dark*u.second, data_exposure = texp_flat*u.second, exposure_unit = u.second, gain_corrected = False))
	combiner_flat = Combiner(flat_list)

	combiner_flat.scaling = scaling_func  
	master_flat = combiner_flat.average_combine()

	hdulamp = fits.open("sirius_lamp.fit")
	texp_lamp = float(hdulamp[0].header['EXPOSURE'])
	lamp = np.asarray(ccd_process(CCDData(fits.getdata("sirius_lamp.fit", ext=0), unit = "adu"), dark_frame = master_dark, dark_exposure = texp_dark*u.second, data_exposure = texp_lamp*u.second, gain_corrected = False))

	spectrum1, wavelength1, spectrum2, wavelength2, ra_ICRS, dec_ICRS = Interactivity(image_names, lamp, master_flat, master_dark, PolyFit, debug = False)
	'''
	SpectrumPlot(wavelength1, spectrum1, xunit = "nm", title = "Extracted Spectrum from "sirius.fit")
	SpectrumPlot(wavelength2, spectrum2, xunit = "nm", title = "Extracted Spectrum from "sirius2.fit")
	plt.show()
	plt.close('all')
	'''
	bin_size = 5

	wav, spav1, spav2 = InterpolateSpectra(wavelength1, spectrum1, wavelength2, spectrum2, bin_size)

	vega_wave, vega_flux = np.genfromtxt("vega_std.txt", usecols = (0,1), unpack = True, dtype = (float, float))
	spl = CubicSpline(vega_wave/10, vega_flux)
	spav3 = spl(wav)
	'''
	SpectrumPlot(wav, spav1, xunit = "nm", title = "Interpolated Spectrum from "sirius.fit")
	SpectrumPlot(wav, spav2, xunit = "nm", title = "Interpolated Spectrum from "sirius2.fit")
	SpectrumPlot(wav, spav3, xunit = "nm", title = "Interpolated Vega Spectrum")
	plt.show()
	plt.close('all')
	'''
	air1 = FindAirmass(image_names[0], ra_ICRS, dec_ICRS)
	air2 = FindAirmass(image_names[1], ra_ICRS, dec_ICRS)

	texp1 = float(fits.open(image_names[0])[0].header['EXPOSURE'])
	texp2 = float(fits.open(image_names[1])[0].header['EXPOSURE'])

	mask = wav >= 450

	transmittance_0, instrumental_0 = Response(wav[mask], spav1[mask], spav2[mask], spav3[mask], air1, air2, texp1, texp2, debug = False)

	CheckBack(wav, spav1, spav3, air1, texp1)