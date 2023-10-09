import numpy as np
from matplotlib import pyplot as plt
import os
import datetime
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, TETE, GCRS
from astropy.coordinates import solar_system_ephemeris, get_sun


def DoCalculation(dfrac, ra_set, dec_set, phi, L, T0, DT, h0):
	"""
	Calculation needed both for the iterative correction to the day fractions and for the plots
	
	Inputs:
		dfrac : float
			Day fraction for which the computation is needed
		ra_set : ndarray
			Set of ra (see "CoordArraySetup()") - Values in deg
		dec_set : ndarray
			Set of dec (see "CoordArraySetup()") - Values in deg
		phi : float
			Observer latitude - Value in deg
		L : float
			Observer longitude - Value in deg
		T0 : float
			GAST (see "GAST_deg()") - Value in deg

	Outputs:
		ra : float
			Interpolated value of ra at given day fraction - Value in deg
		dec : float
			Interpolated value of dec at given day fraction - Value in deg
		H : float
			Computed value of the local hour angle at given day fraction - Value in deg
		h : float
			Computed alt of object at given day fraction - Value in deg
		A : float
			Computed az of object at given day fraction - Value in deg
	"""

	t0 = (T0 + 360.985647*dfrac)%360
	n = dfrac + DT/86400
	a1 = ra_set[1] - ra_set[0]
	b1 = ra_set[2] - ra_set[1]
	c1 = b1-a1
	ra = ra_set[1] + n*(a1 + b1 + n*c1)/2
	a2 = dec_set[1] - dec_set[0]
	b2 = dec_set[2] - dec_set[1]
	c2 = b2-a2
	dec = dec_set[1] + n*(a2 + b2 + n*c2)/2
	H = t0+L-ra
	if H > 180:
		H = H - 360
	if H < -180:
		H = H + 360
	
	h = np.arcsin((np.sin(phi*np.pi/180)*np.sin(dec*np.pi/180)) + (np.cos(phi*np.pi/180)*np.cos(dec*np.pi/180)*np.cos(H*np.pi/180)))*180/np.pi
	A = np.arctan2(np.sin(H*np.pi/180),(np.cos(H*np.pi/180)*np.sin(phi*np.pi/180) - np.tan(dec*np.pi/180)*np.cos(phi*np.pi/180)))*180/np.pi + 180

	return (ra, dec, H, h, A)


def Seth0(instr = 'star', height_slm = 0):
	"""
	Sets h0 (in deg) for further use

	Inputs:
		instr : str
			Which h0 is needed
		height_slm : float
			Height above MSL - Value in m

	Outputs:
		h0 : float
			Value for the requested event/object corrected for height above MSL

	For the sun h0 = -50/60
	For civil twilight h0 = -6
	For astronomical twilight h0 = -18
	For other objects h0 = -34/60
	"""

	if instr == 'sun':
		return -50/60 - 0.0353*np.sqrt(height_slm)
	if instr == 'civil':
		return -6 - 0.0353*np.sqrt(height_slm)
	if instr == 'astro':
		return -18 - 0.0353*np.sqrt(height_slm)
	else:
		return -34/60 - 0.0353*np.sqrt(height_slm)


def ToJDN(dd,mm,yy):
	"""
	Calculate JDN from gregorian date

	Inputs:
		dd : int
			Day
		mm : int
			Month
		yy : int
			Year

	Outputs:
		JDN : float
			Julian Day Number
	"""

	if mm == 1 or mm == 2:
		mm = mm+12
		yy = yy-1
	A = (yy/100)//1
	B = (A/4)//1
	C = 2 - A + B
	E = (365.25*((yy+4716)//1))//1
	F = (30.6001*(mm+1))//1

	return C+dd+E+F-1524.5


def GAST_deg(JD):
	"""
	Calculate Greenwich Apparent Sideral Time at 0h for the needed day from USNO

	Inputs:
		JD : float
			Julian Day Number

	Outputs:
		GAST : float
			Greenwich Apparent Sideral Time at 0h for the needed day from USNO - Value in deg
	"""

	DUT = JD - 2451545.0
	GMST = (18.697375 + 24.065709824279*DUT)%24

	Omega = 125.04 - 0.052954*DUT
	SL = 280.47 + 0.98565*DUT
	epsilon = 23.4393 - 0.0000004*DUT

	Dphi = -0.000319*np.sin(Omega*np.pi/180) - 0.000024*np.sin(2*SL*np.pi/180)

	GAST = GMST + Dphi*np.cos(epsilon*np.pi/180)

	return GAST*360/24


def FindDT(yy):
	"""
	Calculate deltaT = TT - UT using one of the many empyrical formulae

	Inputs:
		yy : int
			Year

	Outputs:
		deltaT : float
			TT - UT in seconds
	"""

	t = (yy - 1825)/100
	return -150.315 + 31.4115*(t**2) + 284.8436*np.cos(2*np.pi*(t + 0.75)/14)


def CoordArraysSetup(dd, mm, yy, ra_ICRS, dec_ICRS, obj = 'star'):
	"""
	Finds values of "apparent" ra/dec (true equator, true equinox) for the day of the observation and the ones before and after

	Inputs:
		dd : int
			Day
		mm : int
			Month
		yy : int
			Year
		ra_ICRS : float
			ra of the object in ICRS frame - Value in deg
		dec_ICRS : float
			dec of the object in ICRS frame - Value in deg
		obj : str
			Instruction for needed body/event: 'sun', 'civil', 'astro' for the Sun, 'star' or other for an object

	Outputs:
		ra_set : ndarray
			Set of "apparent" ra - Values in deg
		dec_set : ndarray
			Set of "apparent" dec - Values in deg

	First "Time" objects are defined for the needed dates.
	With the given times "astropy" is used to get a coordinate object for the target (sun/object) which is then transformed to "TETE" and put in the output arrays

	Each output array contains three values of ra/dec:
		The first value of the array is for the day before, the second for the day of the observation and the third for the day after
	"""

	t1 = Time(datetime.datetime(yy,mm,dd-1,0,0,0), format='datetime', scale='utc')
	t2 = Time(datetime.datetime(yy,mm,dd,0,0,0), format='datetime', scale='utc')
	t3 = Time(datetime.datetime(yy,mm,dd+1,0,0,0), format='datetime', scale='utc')

	if obj == 'sun' or obj == 'civil' or obj == 'astro':
		coords1 = get_sun(time=t1)
		coords2 = get_sun(time=t2)
		coords3 = get_sun(time=t3)
		coords1 = coords1.transform_to('tete')
		coords2 = coords2.transform_to('tete')
		coords3 = coords3.transform_to('tete')

		ra_set = np.array([coords1.ra.degree, coords2.ra.degree, coords3.ra.degree])
		dec_set = np.array([coords1.dec.degree, coords2.dec.degree, coords3.dec.degree])

		return(ra_set, dec_set, t2)

	else:
		coords1 = SkyCoord(ra_ICRS, dec_ICRS, unit='deg', frame='icrs', obstime=t1)
		coords2 = SkyCoord(ra_ICRS, dec_ICRS, unit='deg', frame='icrs', obstime=t2)
		coords3 = SkyCoord(ra_ICRS, dec_ICRS, unit='deg', frame='icrs', obstime=t3)
		coords1 = coords1.transform_to('tete')
		coords2 = coords2.transform_to('tete')
		coords3 = coords3.transform_to('tete')

		ra_set = np.array([coords1.ra.degree, coords2.ra.degree, coords3.ra.degree])
		dec_set = np.array([coords1.dec.degree, coords2.dec.degree, coords3.dec.degree])

		return(ra_set, dec_set, t2)


def ComputeAirmass(dfrac, dd, mm, yy, ra_ICRS, dec_ICRS, phi, L, height):
	"""
	Uses previous functions to compute airmass at given time

	Inputs:
		dfrac : float
			Day fraction for which airmass is needed
		dd : int
			Day
		mm : int
			Month
		yy : int
			Year
		ra_ICRS : float
			ra coordinate in ICRS frame - Value in deg
		dec_ICRS : float
			dec coordinate in ICRS frame - Value in deg
		phi : float
			Observer latitude - Value in deg
		L : float
			Observer longitude - Value in deg
		height : float
			Observer height above MSL - Value in m

	Outputs:
		airmass : float
			Computed airmass value
	"""

	h0 = Seth0('star', height)

	JD = ToJDN(dd,mm,yy)
	T0 = GAST_deg(JD)

	DT = FindDT(yy)

	ra_all, dec_all, t_date = CoordArraysSetup(dd, mm, yy, ra_ICRS, dec_ICRS)

	ra, dec, H, h, A = DoCalculation(dfrac, ra_all, dec_all, phi, L, T0, DT, h0)

	return 1./np.cos((90 - h)*np.pi/180)