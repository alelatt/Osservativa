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

dark_names = ["dark_1.fit", "dark_2.fit"]
flat_names = ["flat_1.fit", "flat_2.fit", "flat_3.fit", "flat_4.fit", "flat_5.fit"]

calibrator = np.flip(io.imread("Spectrum_calibration.png", as_gray = True), axis = 0)
response_fname = "response.txt"


################	GENERAL OBSERVATION DATA	################
phi = 43.722094
L = 10.4079
height = 4


def SpectrumPlot(x, y, xunit = "nm", ylabel = "Intensity", title = ""):
	"""
	Plots a series of points

	Inputs:
		fig_num	Number handle of the figure
		x		Array of x
		y		Array of y
		title	Title of the plot
	"""

	plt.figure(dpi = 150, layout = 'tight')
	plt.plot(x, y, '-k')
	plt.xlabel("$\lambda$ ["+xunit+"]")
	plt.ylabel(ylabel)
	plt.title(title)

	return


def MakeMasterFlat(file_names, master_dark):
	"""
	Averages given images

	Inputs:
		file_names	File names of the images to average

	Outputs:
		data	Averaged image
	"""

	flat_temp = fits.getdata(file_names[0], ext=0) - master_dark
	data = (flat_temp/np.mean(flat_temp))/len(file_names)

	for i in range(1, len(file_names)):
		flat_temp = fits.getdata(file_names[i], ext=0) - master_dark
		data += (flat_temp/np.mean(flat_temp))/len(file_names)

	return data


def LinFit(x, a, b):
	"""
	Linear fit function

	Inputs:
		x	Data
		a	Offset
		b	Slope
	"""

	return a + b*x


def PolyFit(x, a, b, c):
	"""
	Polynomial fit function

	Inputs:
		x	Data
		a	Order 0 coefficient
		b	Order 1 coefficient
		c	Order 2 coefficient
	"""

	return a + b*x + c*(x**2)


def ExpFit(x, a, b, c):
	return np.exp(a*(x-b)) + c


def AtmosphericTau(x):
	"""
	https://www.aanda.org/articles/aa/full_html/2012/07/aa19040-12/aa19040-12.html
	"""
	return 0.00864*(x**(-(3.916 + (0.074*x) + (0.05/x))))


def FitCenter(image, x_min, x_max, debug = False):
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
		for i in range(len(x)):
			plt.plot(image[:,x[i]])

		print("\n\tFit Results: q = %f pm %f, m = %f pm %f" %(pars[0], errs[0], pars[1], errs[1]))
		print("\tRotation Angle = %f" %(rot_ang))

		plt.figure(dpi = 150, layout = 'tight')
		plt.errorbar(x, y, yerrs, marker = '.', color = 'r')
		plt.imshow(image, cmap = 'gray', origin = 'lower', norm = 'log')
		plt.xlabel("X")
		plt.ylabel("Y")
		lin = np.arange(0, len(image[0,:]), 1)
		liny = LinFit(lin, *pars)
		plt.plot(lin, liny, 'r')
		plt.ylim(0, len(image[:,0]))
		plt.xlim(0, len(image[0,:]))
		plt.show()
		plt.close('all')

	return rot_ang


def FineRotate(image, debug = False):
	y_min = None
	y_max = None
	x_min = None
	x_max = None

	plt.figure(dpi = 150, layout = 'tight')
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

	"""

	x_min = None
	x_max = None
	y_c = None
	width = None

	while True:
		plt.figure(dpi = 150, layout = 'tight')
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
		spectrum_lamp	Section of the lamp spectrum corresponding to the extracted spectrum
		calibrator_img	Lamp calibration image
		pixels_array	Array of the pixel position of the section of the spectrum
		FitFn			Fit function for px to wavelength conversion

	Outputs:
		pars	Conversion fit parameters

	The user is shown the extracted lamp spectrum toghether with the lamp calibration file and is asked to enter the x coord. in px.
		and the corresponding x coord in angstrom (as specified in cakibration image) together with an error for as many lines as the user wants.
		A fit is then applied to find the conversion between px and nm
	"""

	SpectrumPlot(pixels_array, spectrum_lamp, xunit = "arb. un.")
	ImagePlot(calibrator_img)
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
		print("\nFit results:")
		for i in range(len(pars)):
			print("\tParameter %i: %f pm %f" %(i+1, pars[i], errs[i]))

		plt.figure(dpi = 150, layout = 'tight')
		plt.errorbar(x, y/10., yerr = err, marker = '.', linestyle = '', color = 'k')
		lin = np.linspace(np.min(x), np.max(x), 1000)
		plt.plot(lin, FitFn(lin, pars[0], pars[1], pars[2]))
		plt.xlabel("Px.")
		plt.ylabel("$\lambda$ [nm]")
		plt.show()
		plt.close('all')

	return (pars, points)


def UseCalibrator(fname, debug = False):
	"""
	Uses calibrator file to convert from px to nm

	Inputs:
		debug	Debug option

	Outputs:
		pars	Conversion fit parameters
	"""
	x, y, err = np.genfromtxt(fname, unpack = True)

	pars, covm = curve_fit(PolyFit, x, y/10., sigma = err)
	errs = np.sqrt(covm.diagonal())

	if debug == True:
		print("\nFit results:")
		for i in range(len(pars)):
			print("\tParameter %i: %f pm %f" %(i+1, pars[i], errs[i]))

		plt.figure(dpi = 150, layout = 'tight')
		plt.errorbar(x, y/10., yerr = err, marker = '.', linestyle = '', color = 'k')
		lin = np.linspace(np.min(x), np.max(x), 1000)
		plt.plot(lin, PolyFit(lin, pars[0], pars[1], pars[2]))
		plt.xlabel("Px.")
		plt.ylabel("$\lambda$ [nm]")
		plt.show()
		plt.close('all')

	return pars


def GetSpectrum(image, lamp, calibrator_img, FitFn, debug = False):
	"""
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
	"""

	rot_ang = FineRotate(image, debug)

	image_rot = ndimage.rotate(image, rot_ang, reshape = False)
	lamp_rot = ndimage.rotate(lamp, rot_ang, reshape = False)

	if debug == True:
		plt.figure(dpi = 150, layout = 'tight')
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


def Interactivity(flat, dark, FitFn, debug = False):
	"""
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
	"""

	#gain = np.mean(flat)/flat #, master_flat = flat

	hdul_dark = fits.open(dark_names[0])
	texp_dark = float(hdul_dark[0].header['EXPOSURE'])

	fname = input("Enter file name (without extension): ")
	hdul = fits.open(fname+".fit")
	texp_img = float(hdul[0].header['EXPOSURE'])
	img_raw = np.asarray(ccd_process(CCDData(fits.getdata(fname+".fit", ext=0), unit = "adu"), dark_frame = dark, dark_exposure = texp_dark*u.second, data_exposure = texp_img*u.second, gain_corrected = False))

	hdulamp = fits.open(fname+"_lamp.fit")
	texp_lamp = float(hdulamp[0].header['EXPOSURE'])
	lamp = np.asarray(ccd_process(CCDData(fits.getdata(fname+"_lamp.fit", ext=0), unit = "adu"), dark_frame = dark, dark_exposure = texp_dark*u.second, data_exposure = texp_lamp*u.second, gain_corrected = False))

	spectrum_meas1 = None
	lam1 = None
	spectrum_meas2 = None
	lam2 = None

	choice = input("\nUse spectrum extraction file? (y/n) ")
	if choice == "y":
		ra_ICRS, dec_ICRS, rot_ang, x_min, x_max, y_c, width = np.genfromtxt(fname+"_config.txt", unpack = True, dtype = (float, float, float, int, int, int, int))

		rot_img = ndimage.rotate(img_raw, rot_ang, reshape = False)

		spectrum_meas = np.mean(rot_img[(y_c - width):(y_c + width + 1), x_min:(x_max + 1)], axis=0)

		pixels = np.arange(x_min, x_max + 1, 1)

		pars = UseCalibrator(fname+"_calibr.txt", debug)

		lam = PolyFit(pixels, *pars)

		air = FindAirmass(fname+".fit", ra_ICRS, dec_ICRS)
		
	else:
		lam, spectrum_meas, rot_ang, x_min, x_max, y_c, width, points = GetSpectrum(img_raw, lamp, calibrator, FitFn, debug)

		ra_ICRS = None
		dec_ICRS = None
		print("\nEnter object coordinates in ICRS frame:")
		while True:
			while True:
				ra_ICRS = float(input("\tEnter Object RA (deg): "))
				if (ra_ICRS >= 0) and (ra_ICRS < 360):
					break
				else:
					print("\t\tValue not in bounds [0,360)")

			while True:
				dec_ICRS = float(input("\tEnter Object Dec (deg): "))
				if (dec_ICRS >= -90) and (dec_ICRS <= 90):
					break
				else:
					print("\t\tValue not in bounds [-90,90]")

			choice = input("Change coordinates? (y/n) ")
			if choice == 'n':
				break

		save = input("\nSave spectrum extraction file? (y/n) ")
		if save == "y":
			np.savetxt(fname+"_config.txt", [np.array([ra_ICRS, dec_ICRS, rot_ang, x_min, x_max, y_c, width])], delimiter = '\t', fmt = ['%.6f', '%.6f', '%.4f', '%d', '%d', '%d', '%d'])
			np.savetxt(fname+"_calibr.txt", points.T, delimiter = '\t', fmt = '%.2f')

		air = FindAirmass(fname+".fit", ra_ICRS, dec_ICRS)

	return (spectrum_meas, lam, air, texp_img, ra_ICRS, dec_ICRS)


def BinSpectrum(spectrum_meas, lam, bin_size):
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


def FindAirmass(image_name, ra_ICRS, dec_ICRS):
	hdul = fits.open(image_name)
	date = hdul[0].header['DATE-OBS']

	yy = int(date[:4])
	mm = int(date[5:7])
	dd = int(date[8:10])
	hh = int(date[11:13])
	mn = int(date[14:16])
	ss = int(date[17:19])

	dayfrac = (hh + mn/60 + ss/3600)/24

	return ComputeAirmass(dayfrac, dd, mm, yy, ra_ICRS, dec_ICRS, phi, L, height)


def CorrectSpectrum(wavelength, spectrum, airmass, texp, resp_fname):
	resp_wave, trans_0, instr_0 = np.genfromtxt(response_fname, unpack = True, dtype = (int, float, float))

	low = np.max([np.min(wavelength), np.min(resp_wave)])
	high = np.min([np.max(wavelength), np.max(resp_wave)])

	mask_img = (wavelength >= low) & (wavelength <= high)
	mask_resp = (resp_wave >= low) & (resp_wave <= high)

	spectrum_corr = spectrum[mask_img]/(instr_0[mask_resp]*texp*(trans_0[mask_resp]**airmass))

	return wavelength[mask_img], spectrum_corr
	


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

	spectrum, wavelength, airmass, texp, ra_ICRS, dec_ICRS = Interactivity(master_flat, master_dark, PolyFit, debug = True)
	
	SpectrumPlot(wavelength, spectrum, xunit = "nm")
	plt.show()
	plt.close('all')
	
	bin_size = 5

	wav, spav = BinSpectrum(spectrum, wavelength, bin_size)
	
	SpectrumPlot(wav, spav, xunit = "nm")
	plt.show()
	plt.close('all')

	wav_corr, spec_corr = CorrectSpectrum(wav, spav, airmass, texp, response_fname)

	SpectrumPlot(wav_corr, spec_corr, xunit = "nm", ylabel = r"Flux [erg $s^{-1}$ $cm^{-2}$ A $10^{16}$]")
	plt.show()
	plt.close('all')	