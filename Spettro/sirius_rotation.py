#Counter-Rotates "sirius.fit", which was auto-rotated


from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage
import os
from skimage import io
from tracefns import *
from astropy.utils.data import get_pkg_data_filename

os.chdir('./28_marzo_2023')

image_names = ["sirius.fit", "sirius2.fit"]
dark_names = ["dark_1.fit", "dark_2.fit"]
flat_names = ["flat_1.fit", "flat_2.fit", "flat_3.fit", "flat_4.fit", "flat_5.fit"]

calibrator = np.flip(io.imread("Spectrum_calibration.png", as_gray = True), axis = 0)

def Line(x, a, b):
	return a + b*x



if __name__ == '__main__':
	img = abs(np.gradient(fits.getdata(image_names[0], ext=0))[0])[:60,1500:]

	img[np.where(img <= 0.7*np.max(img))] = 0
	img[np.where(img > 0.7*np.max(img))] = 1

	plt.figure(dpi = 150, layout = 'tight')
	plt.imshow(img, origin = 'lower')

	x = np.where(img == 1)[1]
	y = np.where(img == 1)[0]

	pars, covm = curve_fit(Line, x, y)
	errs = np.sqrt(covm.diagonal())

	vec = np.array([1., pars[1]])/np.sqrt(pars[1]**2 + 1)

	rot_ang = None

	if vec[1] < 0:
		rot_ang = -np.arccos(np.dot(vec, np.array([1,0])))*180/np.pi
	else:
		rot_ang = np.arccos(np.dot(vec, np.array([1,0])))*180/np.pi

	plt.figure()
	plt.plot(x,y,'.')
	lin = np.arange(0, len(img[0,:]), 1)
	plt.plot(lin, Line(lin, *pars))
	#plt.show()

	print(-rot_ang)

	plt.figure()
	plt.imshow(fits.getdata(image_names[0], ext=0), origin = 'lower', norm = 'log')
	plt.figure()
	plt.imshow(ndimage.rotate(fits.getdata(image_names[0], ext=0), rot_ang, reshape = True), origin = 'lower', norm = 'log')
	#plt.show()

	fits.setval(image_names[0], 'ROT_ANG', value = str(-rot_ang))