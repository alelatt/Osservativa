import numpy as np
import os

from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.optimize import curve_fit
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from skimage import io
from ccdproc import Combiner, ccd_process
from tracefns import *
from spectrfns import *


os.chdir('./28_marzo_2023')

dark_names = ["dark_1.fit", "dark_2.fit"]
flat_names = ["flat_1.fit", "flat_2.fit", "flat_3.fit", "flat_4.fit", "flat_5.fit"]

calibrator = np.flip(io.imread("Spectrum_calibration.png", as_gray = True), axis = 0)
response_fname = "response.txt"


################	GENERAL OBSERVATION DATA	################
phi = 43.722094
L = 10.4079
height = 4



def Interactivity(flat, dark, FitFn, debug = False):
	"""
	Sets different procedures if a calibrator file is used or not

	Inputs:
		flat : ndarray
			Master flat frame
		dark : ndarray
			Master dark frame
		FitFn : function
			Polynomial fit function for px to nm conversion
		debug : bool
			If True shows debug options in all code

	Outputs:
		spectrum_meas : ndarray
			Measured spectrum intensities
		lam : ndarray
			Measured spectrum wavelength array
		air : float
			Airmass at observation time
		texp_img : float
			Exposure time of image
		ra_ICRS : float
			Object ra coordinate in degrees in ICRS frame
		dec_ICRS : float
			Object dec coordinate in degrees in ICRS frame

	If no calibration file is used one can be created at the end of the process
	"""

	hdul_dark = fits.open(dark_names[0])
	texp_dark = float(hdul_dark[0].header['EXPOSURE'])

	fname = input("Enter file name (without extension): ")
	hdul = fits.open(fname+".fit")
	texp_img = float(hdul[0].header['EXPOSURE'])
	img_raw = np.asarray(ccd_process(CCDData(fits.getdata(fname+".fit", ext=0), unit = "adu"), dark_frame = dark, dark_exposure = texp_dark*u.second, data_exposure = texp_img*u.second, gain_corrected = False))

	hdulamp = fits.open(fname+"_lamp.fit")
	texp_lamp = float(hdulamp[0].header['EXPOSURE'])
	lamp = np.asarray(ccd_process(CCDData(fits.getdata(fname+"_lamp.fit", ext=0), unit = "adu"), dark_frame = dark, dark_exposure = texp_dark*u.second, data_exposure = texp_lamp*u.second, gain_corrected = False))

	spectrum_meas = None
	lam = None
	ra_ICRS = None
	dec_ICRS = None

	choice = input("\nUse spectrum extraction file? (y/n) ")
	if choice == "y":
		ra_ICRS, dec_ICRS, rot_ang, x_min, x_max, y_c, width = np.genfromtxt(fname+"_config.txt", unpack = True, dtype = (float, float, float, int, int, int, int))

		rot_img = ndimage.rotate(img_raw, rot_ang, reshape = False)

		spectrum_meas = np.mean(rot_img[(y_c - width):(y_c + width + 1), x_min:(x_max + 1)], axis=0)

		pixels = np.arange(x_min, x_max + 1, 1)

		pars = UseCalibrator(fname+"_calibr.txt", FitFn, debug)

		lam = PolyFit(pixels, *pars)
		
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


def CorrectSpectrum(wavelength, spectrum, airmass, texp, resp_fname):
	"""
	Corrects the spectrum using already found response functions

	Inputs:
		wavelength : ndarray
			Spectrum wavelength array
		spectrum : ndarray
			Spectrum intensity array
		airmass : float
			Airmass at observation time
		texp : float
			Exposure time
		resp_fname : str
			File name of response functions

	Outputs:
		wavelength[mask_img] : ndarray
			Wavelength array masked to match intensities array
		spectrum_corr : ndarray
			Corrected intensities array
	"""

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
	
	SpectrumPlot(wavelength, spectrum, xunit = "nm", title = "Raw Spectrum")
	plt.show()
	plt.close('all')
	
	bin_size = 5

	wav, spav = BinSpectrum(wavelength, spectrum, bin_size)
	
	SpectrumPlot(wav, spav, xunit = "nm", title = "Binned Spectrum")
	plt.show()
	plt.close('all')

	wav_corr, spec_corr = CorrectSpectrum(wav, spav, airmass, texp, response_fname)

	SpectrumPlot(wav_corr, spec_corr, xunit = "nm", ylabel = r"Flux [erg $s^{-1}$ $cm^{-2}$ $nm^{-1}$ $10^{17}$]", title = "Corrected Spectrum")
	plt.show()
	plt.close('all')	