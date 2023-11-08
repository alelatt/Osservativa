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

image_names = ["sirius.fit", "sirius2.fit"]
dark_names = ["dark_1.fit", "dark_2.fit"]
flat_names = ["flat_1.fit", "flat_2.fit", "flat_3.fit", "flat_4.fit", "flat_5.fit"]

calibrator = np.flip(io.imread("Spectrum_calibration.png", as_gray = True), axis = 0)


################	GENERAL OBSERVATION DATA	################
phi = 43.722094
L = 10.4079
height = 4



def AtmosphericTau(x):
	"""
	Optical depth from https://www.aanda.org/articles/aa/full_html/2012/07/aa19040-12/aa19040-12.html

	Inputs:
	x : ndarray
		Array of wavelength in nm
	"""

	x_um = x/1000

	return 0.00864*(x_um**(-(3.916 + (0.074*x_um) + (0.05/x_um))))


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
		img1_raw = np.asarray(ccd_process(CCDData(ndimage.rotate(fits.getdata(image_names[0], ext=0), -rotated_ang, reshape = False), unit = "adu"), master_flat = flat, dark_frame = dark, dark_exposure = texp_dark*u.second, data_exposure = texp_img1*u.second,  gain_corrected = False))

		hdul2 = fits.open(image_names[1])
		texp_img2 = float(hdul2[0].header['EXPOSURE'])
		img2_raw = np.asarray(ccd_process(CCDData(fits.getdata(image_names[1], ext=0), unit = "adu"), master_flat = flat, dark_frame = dark, dark_exposure = texp_dark*u.second, data_exposure = texp_img2*u.second, gain_corrected = False))

		rot_img1 = ndimage.rotate(img1_raw, rot_ang1, reshape = False)
		rot_img2 = ndimage.rotate(img2_raw, rot_ang2, reshape = False)

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
		img1_raw = np.asarray(ccd_process(CCDData(ndimage.rotate(fits.getdata(image_names[0], ext=0), -rotated_ang, reshape = False), unit = "adu"), master_flat = flat, dark_frame = dark, dark_exposure = 20*u.second, data_exposure = 10*u.second, exposure_unit = u.second, gain_corrected = False))

		hdul2 = fits.open(image_names[1])
		img2_raw = np.asarray(ccd_process(CCDData(fits.getdata(image_names[1], ext=0), unit = "adu"), master_flat = flat, dark_frame = dark, dark_exposure = 20*u.second, data_exposure = 5*u.second, exposure_unit = u.second, gain_corrected = False))

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
		A1 : float
			Corrective coefficient for spectrum 1
		A2 : float
			Corrective coefficient for spectrum 2
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
	mean_instr = np.mean(np.vstack((instr_0_1, instr_0_2)), axis = 0)
	atmos_mask = (wavelength > 680) & (wavelength < 700)
	spl_instr_0 = UnivariateSpline(wavelength[~atmos_mask], mean_instr[~atmos_mask], k = 4)
	instr_0 = spl_instr_0(wavelength)

	tot_0 = trans_0*instr_0

	if debug == True:
		fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = [10, 7], dpi = 100, layout = 'constrained')
		fig.suptitle("Intensity Ratio", fontsize = 20)
		ax.plot(wavelength, ratio, '.k', label = "$I_1 / I_2$")
		ax.legend(loc = 'best', fontsize = 15)
		ax.set_xlabel("$\lambda$ [nm]", fontsize = 15)
		ax.set_ylabel("$I_1 / I_2$", fontsize = 15)
		ax.tick_params(labelsize = 15)
		ax.grid(True)

		fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = [10, 7], dpi = 100, layout = 'constrained')
		fig.suptitle("Optical Depth", fontsize = 20)
		ax.plot(wavelength, tau_prime, '.k', label = r"$\tau'$")
		lin = np.linspace(wavelength[0], wavelength[-1], 1000)
		ax.plot(lin, ExpFit(lin, *pars), 'b', label = r"Fit over $\tau'$")
		ax.plot(wavelength, tau, 'r', label = r"$\tau$")
		ax.plot(lin, AtmosphericTau(lin), '--g', label = r"Reference $\tau$")
		ax.legend(loc = 'best', fontsize = 15)
		ax.set_xlabel("$\lambda$ [nm]", fontsize = 15)
		ax.set_ylabel("Optical Depth", fontsize = 15)
		ax.tick_params(labelsize = 15)
		ax.grid(True)

		fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = [10, 7], dpi = 100, layout = 'constrained')
		fig.suptitle("Atmospheric Transmittance", fontsize = 20)
		ax.plot(wavelength, trans_0, 'r', label = "Computed Transmittance")
		ax.plot(lin, np.exp(-AtmosphericTau(lin)), '--g', label = "Reference Transmittance")
		ax.legend(loc = 'best', fontsize = 15)
		ax.set_xlabel("$\lambda$ [nm]", fontsize = 15)
		ax.set_ylabel("Atmospheric Transmittance", fontsize = 15)
		ax.tick_params(labelsize = 15)
		ax.grid(True)

		fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = [10, 7], dpi = 100, layout = 'constrained')
		fig.suptitle("Instrumental Response", fontsize = 20)
		ax.plot(wavelength, instr_0_1, 'r', label = "Instrumental Response from spectrum 1")
		ax.plot(wavelength, instr_0_2, 'b', label = "Instrumental Response from spectrum 2")
		ax.plot(wavelength, instr_0, '--k', label = "Averaged and smoothed Instrumental Response")
		ax.legend(loc = 'best', fontsize = 15)
		ax.set_xlabel("$\lambda$ [nm]", fontsize = 15)
		ax.set_ylabel("Instr. Resp. [counts/(erg $s^{-1}$ $cm^{-2}$ $nm^{-1}$ $10^{17}$)]", fontsize = 15)
		ax.yaxis.offsetText.set_fontsize(15)
		ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
		ax.tick_params(labelsize = 15)
		ax.grid(True)
		
		fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = [10, 7], dpi = 100, layout = 'constrained')
		fig.suptitle("Total Response Function at 1 airmass and 1s exposure", fontsize = 20)
		ax.plot(wavelength, tot_0, label = "Total Response Function")
		ax.legend(loc = 'best', fontsize = 15)
		ax.set_xlabel("$\lambda$ [nm]", fontsize = 15)
		ax.set_ylabel("Total Resp. [counts/(erg $s^{-1}$ $cm^{-2}$ $nm^{-1}$ $10^{17}$)]", fontsize = 15)
		ax.yaxis.offsetText.set_fontsize(15)
		ax.tick_params(labelsize = 15)
		ax.grid(True)

		plt.show()
		plt.close('all')

	resp_array = np.vstack((wavelength, trans_0, instr_0))

	choice = input("\nSave response functions? (y/n) ")
	if choice == 'y':
		np.savetxt("response.txt", resp_array.T, delimiter = '\t', fmt = ['%i', '%.8f', '%.8f'], header = "Colums are [wavelength, atmospheric transmission (airmass = 1), instrument response (exposure time = 0)]")

	return trans_0, instr_0, A1, A2


def CheckBack(wavelength, spectrum, vega, airmass, texp, A):
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

	spectrum_corr = spectrum[mask_img]/(instr_0[mask_resp]*texp*(trans_0[mask_resp]**airmass)*A)

	plt.figure(figsize = [10, 7], dpi = 100, layout = 'constrained')
	plt.title("Corrected Spectrum and Vega (used as reference)", fontsize = 20)
	plt.plot(wavelength[mask_img], spectrum_corr, 'k', label = "Corrected Spectrum")
	plt.plot(wavelength[mask_img], vega[mask_img], 'r', label = "Vega")
	plt.legend(loc = 'best', fontsize = 15)
	plt.xlabel(r"$\lambda$ [nm]", fontsize = 15)
	plt.ylabel(r"Flux [erg $s^{-1}$ $cm^{-2}$ $nm^{-1}$ $10^{17}$]", fontsize = 15)
	plt.xticks(fontsize = 15)
	plt.yticks(fontsize = 15)
	plt.grid(True)
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
	SpectrumPlot(wavelength1, spectrum1, xunit = "nm", title = "Extracted Spectrum from sirius.fit")
	SpectrumPlot(wavelength2, spectrum2, xunit = "nm", title = "Extracted Spectrum from sirius2.fit")
	plt.show()
	plt.close('all')
	'''
	bin_size = 5

	wav, spav1, spav2 = InterpolateSpectra(wavelength1, spectrum1, wavelength2, spectrum2, bin_size)

	vega_wave, vega_flux = np.genfromtxt("vega_std.txt", usecols = (0,1), unpack = True, dtype = (float, float))
	wav_v, spav_v = BinSpectrum(vega_wave/10, vega_flux, bin_size)
	spav3 = spav_v[(wav_v >= np.min(wav)) & (wav_v <= np.max(wav))]
	'''
	SpectrumPlot(wav, spav1, xunit = "nm", title = "Interpolated Spectrum from sirius.fit")
	SpectrumPlot(wav, spav2, xunit = "nm", title = "Interpolated Spectrum from sirius2.fit")
	SpectrumPlot(wav, spav3, xunit = "nm", title = "Interpolated Vega Spectrum")
	plt.show()
	plt.close('all')
	'''
	air1 = FindAirmass(image_names[0], ra_ICRS, dec_ICRS)
	air2 = FindAirmass(image_names[1], ra_ICRS, dec_ICRS)
	print(air1, air2)

	texp1 = float(fits.open(image_names[0])[0].header['EXPOSURE'])
	texp2 = float(fits.open(image_names[1])[0].header['EXPOSURE'])

	mask = (wav >= 450)

	transmittance_0, instrumental_0, A1, A2 = Response(wav[mask], spav1[mask], spav2[mask], spav3[mask], air1, air2, texp1, texp2, debug = True)

	CheckBack(wav, spav1, spav3, air1, texp1, A1)
	CheckBack(wav, spav2, spav3, air2, texp2, A2)