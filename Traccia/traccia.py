import numpy as np
from matplotlib import pyplot as plt
import os
import datetime
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, TETE, GCRS
from astropy.coordinates import solar_system_ephemeris, get_sun

solar_system_ephemeris.set('jpl') 

"""
Implementation of the algorythm by Jean Meus from "Astronomical Algorithms" (1998), Ch.15.


EXECUTION EXAMPLE (">>" indicates user input):
Interactivity (y/n): >>y
Date of Observation:
        Year: >>2023
        Month: >>9
        Day: >>19
Object ICRS Coordinates:
        Object RA [deg]: >>101.28715533
        Object Dec [deg]: >>-16.71611586
Observer Coordinates:
        For Latitude + in the Northern Hemisphere, - in the Southern
        For Longitude + westwards from Prime Meridian, - eastwards
        Observer Latitude [deg]: >>44.007947
        Observer Longitude [deg]: >>10.099098
        Observer Height MSL [m]: >>0

Sun:
        Astro Twilight End:      2023-09-19 03:24:50 [UTC]
        Civil Twilight End:      2023-09-19 04:33:55 [UTC]
        Rising:                  2023-09-19 05:02:49 [UTC]       Az. from North 86.99 [deg]
        Transit:                 2023-09-19 11:13:29 [UTC]       Az. from North 180.00 [deg]
        Setting:                 2023-09-19 17:23:23 [UTC]       Az. from North 272.73 [deg]
        Civil Twilight Start:    2023-09-19 17:52:12 [UTC]
        Astro Twilight Start:    2023-09-19 19:01:01 [UTC]

Object:
        Rising:          2023-09-19 01:19:07 [UTC]       Alt. -0.57 [deg]        Az. from North 113.01 [deg]
        Transit:         2023-09-19 06:14:12 [UTC]       Alt. 29.26 [deg]        Az. from North 180.00 [deg]     Airmass 2.046
        Setting:         2023-09-19 11:09:16 [UTC]       Alt. -0.57 [deg]        Az. from North 246.99 [deg]
"""

def Interactivity(instr = 'n'):
	"""
	Interactive input of date, observer location (latitude/longitude/height) and object ICRS coordinates

	Inputs:
		instr	Instruction for interactivity (y/n)

	Outputs:
		dd : int
			Day
		mm : int
			Month
		yy : int
			Year
		ra_ICRS : float
			ra of the object in ICRS frame  - Value in deg
		dec_ICRS : float
			dec of the object in ICRS frame  - Value in deg
		phi : float
			Observer latitude  - Value in deg
		L : float
			Observer Longitude  - Value in deg
		height : float

			Height above MSL  - Value in m

	If 'y' was chosen the user inputs each variable as prompted
	In 'n' was chosen the script defaults to current day, observer in Massa (MS) and M31 as the object
	"""

	if instr == 'y':
		while True:
			print("Date of Observation:")
			yy = int(input("\tYear: "))
			mm = int(input("\tMonth: "))
			dd = int(input("\tDay: "))

			print("Object ICRS Coordinates:")
			ra_ICRS = float(input("\tObject RA [deg]: "))
			dec_ICRS = float(input("\tObject Dec [deg]: "))

			if (ra_ICRS >= 0) and (ra_ICRS <= 360) and (dec_ICRS >= -90) and (dec_ICRS <= 90):
				break
			else:
				print("Star coordinates input error")

		while True:
			print("Observer Coordinates:")
			print("\tFor Latitude + in the Northern Emisphere, - in the Southern")
			print("\tFor Longitude + westwards from Prime Meridian, - eastwards")
			phi = float(input("\tObserver Latitude [deg]: "))
			L = float(input("\tObserver Longitude [deg]: "))
			height = float(input("\tObserver Height MSL [m]: "))

			if (phi >= -180) and (phi <= 180) and (L >= -90) and (L <= 90) and (height >= 0):
				break
			else:
				print("Observer oordinates input error")

		return (dd, mm, yy, ra_ICRS, dec_ICRS, phi, L, height)

	else:
		curr_day = datetime.datetime.now()
		yy = curr_day.year
		mm = curr_day.month
		dd = curr_day.day

		#M31
		ra_ICRS = 10.684708
		dec_ICRS = 41.268750

		#Massa(MS)
		phi = 44.007947
		L = 10.099098
		height = 0

		print("Showing results for M31 viewed today (%i/%i/%i) from Massa (MS)" %(dd,mm,yy))
		
		return (dd, mm, yy, ra_ICRS, dec_ICRS, phi, L, height)


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
	T = DUT/36525
	GMST = (6.697375 + 0.065709824279*DUT + 0.0000258*(T**2))%24

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

	t1 = Time(datetime.datetime(yy,mm,dd-1,0,0,0), format='datetime', scale='tt')
	t2 = Time(datetime.datetime(yy,mm,dd,0,0,0), format='datetime', scale='tt')
	t3 = Time(datetime.datetime(yy,mm,dd+1,0,0,0), format='datetime', scale='tt')

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


def FindH0(h0, phi, dec):
	"""
	Computes H0 for further use

	Inputs:
		h0 : float
			Computed value of h0 (see "Seth0()") - Value in deg
		phi : float
			Observer latitude - Value in deg
		dec : float
			dec of the object at 0h on observation day - Value in deg

	Outputs:
		H0 : float
			First estimate of H0 - Value in deg

	Setting H0 = 0/180 if cosH0 excedes +-1 is an integration taken from the "Explanatory Supplement to the Astronomical Almanac", 3rd Edition (2013)
		This is the case of an object which is always visible or hidden
	"""

	if (np.sin(h0*np.pi/180) - np.sin(phi*np.pi/180)*np.sin(dec*np.pi/180))/(np.cos(phi*np.pi/180)*np.cos(dec*np.pi/180)) > 1:
		return 0
	if (np.sin(h0*np.pi/180) - np.sin(phi*np.pi/180)*np.sin(dec*np.pi/180))/(np.cos(phi*np.pi/180)*np.cos(dec*np.pi/180)) < - 1:
		return 180
	else:
		return np.arccos((np.sin(h0*np.pi/180) - np.sin(phi*np.pi/180)*np.sin(dec*np.pi/180))/(np.cos(phi*np.pi/180)*np.cos(dec*np.pi/180)))*180/np.pi


def InitDayFrac(ra, L, T0, H0, debug = False):
	"""
	Sets up an array with a first estimate of the three day fractions for rising, culmination and setting (in this order)

	Inputs:
		ra : float
			ra of the object at 0h on observation day - Value in deg
		L : float
			Observer longitude - Value in deg
		T0 : float
			GAST (see "GAST_deg()") - Value in deg
		H0 : float
			First estimate of hour angle (see "FindH0()") - Value in deg
		debug : bool
			Debug Option: shows the first estimates

	Outputs:
		m : ndarray
			Array of day fractions

	Rising fraction might be < 0 if object culminates on the current day but rises the day before
	Setting fraction might be > 1 if object culminates on the current day but sets the day after
	"""

	m0 = (ra - L - T0)/360

	if m0 < 0:
		m0 = m0 + 1
	if m0 > 1:
		m0 = m0 - 1

	m1 = m0 - H0/360
	m2 = m0 + H0/360

	if debug == True:
		print(np.array([m1,m0,m2]))

	return np.array([m1,m0,m2])


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


def CorrectDayFrac(dfrac, ra_set, dec_set, phi, L, T0, DT, h0, debug = False, RiseOrSet = True):
	"""
	Corrects first estimate for day fraction estimate. Different correction between rise/set and culmination

	Inpust:
		dfrac : float
			Day fraction for which the correction is needed
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
		DT : float
			TT - UT in seconds
		h0 : float
			Computed value of h0 (see "Seth0()") - Value in deg
		debug : bool
			Debug Option: shows corrections at each step
		RiseOrSet : bool
			Sets correction procedure

	Outputs:
		m : float
			Corrected day fraction
		h : float
			Alt. - Value in deg
		A : float
			Az. - Value in deg

	Stops the procedure when correction smaller than 1e-6
	"""

	m = np.copy(dfrac)
	dm = 100.
	h = 1000.
	A = 1000.

	while True:
		ra, dec, H, h, A = DoCalculation(m, ra_set, dec_set, phi, L, T0, DT, h0)

		if RiseOrSet == False:
			m = m - H/360

			if debug == True:
				print("H = %f" %(H))
			if abs(H) <= 1e-6:
				break
		
		if RiseOrSet == True:
			dm = (h-h0)/(360*np.cos(dec*np.pi/180)*np.cos(phi*np.pi/180)*np.sin(H*np.pi/180))

			if debug == True:
				print("m = %f, dm = %f" %(m, dm))
			if abs(dm) <= 1e-6:
				break

			m = m + dm

	return (m, h, A)


def CorrectFracSet(frac_set, ra_set, dec_set, phi, L, T0, DT, h0, debug = False):
	"""
	Sets up correction of set of day fractions and outputs sets of alt/az coordinates. Order in the sets is the same as in CoordArraySetup

	Inputs:
		frac_set : ndarray
			Set of day fractions for which the correction is needed
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
		DT : float
			TT - UT in seconds
		h0 : float
			Computed value of h0 (see "Seth0()") - Value in deg
		debug : bool : 

			Debug Option: shows corrections at each step

	Outputs:
		m : ndarray
			Corrected day fraction set
		h : ndarray
			Alt. set - Values in deg
		A : ndarray
			Az. set - Values in deg		
	"""

	m = np.copy(frac_set)
	h_set = np.zeros(3)
	A_set = np.zeros(3)

	for i in range(0, len(frac_set)):
		if i == 1:
			m[1], h_set[1], A_set[1] = CorrectDayFrac(m[1], ra_set, dec_set, phi, L, T0, DT, h0, debug, RiseOrSet = False)
		else:
			m[i], h_set[i], A_set[i] = CorrectDayFrac(m[i], ra_set, dec_set, phi, L, T0, DT, h0, debug)

	return (m, h_set, A_set)


def PlotIt(dfrac_set, ra_set, dec_set, phi, L, T0, DT, h0):
	"""
	Sets up plot of alt/az evolution between rise and set of the object

	Inputs:
		dfrac_set : ndarray
			Set of corrected day fractions
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
		DT : float
			TT - UT in seconds
		h0 : float
			Computed value of h0 (see "Seth0()") - Value in deg
	"""

	m = np.linspace(dfrac_set[0], dfrac_set[2], 100)
	h = np.zeros(len(m))
	A = np.zeros(len(m))

	for i in range(0, len(m)):
		ra, dec, H, h[i], A[i] = DoCalculation(m[i], ra_set, dec_set, phi, L, T0, DT, h0)

	ra, dec, H, h_culm, A_culm = DoCalculation(dfrac_set[1], ra_set, dec_set, phi, L, T0, DT, h0)

	plt.figure(dpi = 170, layout = 'tight')
	plt.plot(m*24, h, '.-k', markersize = 2, linewidth = 1)
	plt.plot(dfrac_set[1]*24, h_culm, 'xr', label="Culmination")
	plt.xlabel("Hours since 00:00 UTC")
	plt.ylabel("Alt [deg]")
	plt.title("Altitude with time")
	plt.legend(loc='best')
	plt.grid()

	plt.figure(dpi = 170, layout = 'tight')
	plt.plot(m*24, A, '.-k', markersize = 2, linewidth = 1)
	plt.plot(dfrac_set[1]*24, A_culm, 'xr', label="Culmination")
	plt.xlabel("Hours since 00:00 UTC")
	plt.ylabel("Az [deg]")
	plt.title("Azimuth with time")
	plt.legend(loc='best')
	plt.grid()

	return


def TraceIt(dd, mm, yy, ra_ICRS, dec_ICRS, phi, L, height, obj = 'star', plot = False, debug = False):
	"""
	Manages the complete algorithm
	
	Inputs:
		dd : int
			Day
		mm : int
			Month
		yy : int
			Year
		ra_ICRS : float
			ra of the object in ICRS frame  - Value in deg
		dec_ICRS : float
			dec of the object in ICRS frame  - Value in deg
		phi : float
			Observer latitude  - Value in deg
		L : float
			Observer Longitude  - Value in deg
		height : float
			Height above MSL  - Value in m
		debug : bool
			Debug Option: shows additional info

	Outputs:
		times : ndarray
			Set of datetime objects containing events dates and times
		h : ndarray
			Set of alt. - Values in deg 
		A : ndarray
			Set of az. - Values in deg
	"""

	h0 = Seth0(obj, height)

	JD = ToJDN(dd,mm,yy)
	T0 = GAST_deg(JD)

	DT = FindDT(yy)

	ra_all, dec_all, t_date = CoordArraysSetup(dd, mm, yy, ra_ICRS, dec_ICRS, obj)

	H0 = FindH0(h0, phi, dec_all[1])

	if H0 == 0:
		print("\nObject is always under horizon")
		exit()

	if H0 == 180:
		print("\nObject is always above horizon")

		m = InitDayFrac(ra_all[1], L, T0, H0, debug)

		m_td = TimeDelta(m*24*3600, format='sec')
		times = t_date + m_td

		if plot == True:
			PlotIt(m, ra_all, dec_all, phi, L, T0, DT, h0)

		plt.show()
		exit()

	m = InitDayFrac(ra_all[1], L, T0, H0, debug)

	m, h, A = CorrectFracSet(m, ra_all, dec_all, phi, L, T0, DT, h0, debug)

	m_td = TimeDelta(m*24*3600, format='sec')
	times = t_date + m_td

	if plot == True:
		PlotIt(m, ra_all, dec_all, phi, L, T0, DT, h0)

	return(times.utc, h, A)


##################	INPUT	##################
inter = input("Interactivity (y/n): ")
dd, mm, yy, ra_ICRS, dec_ICRS, phi, L, height = Interactivity(inter)


##################	RISULTATI	##################
times_S, h_S, A_S = TraceIt(dd, mm, yy, ra_ICRS, dec_ICRS, phi, L, height, obj = 'sun', debug = False)
times_S_civ, h_S_civ, A_S_civ = TraceIt(dd, mm, yy, ra_ICRS, dec_ICRS, phi, L, height, obj = 'civil', debug = False)
times_S_astro, h_S_astro, A_S_astro = TraceIt(dd, mm, yy, ra_ICRS, dec_ICRS, phi, L, height, obj = 'astro', debug = False)

print("\nSun:")
print("\tAstro Twilight End:\t %s [UTC]" %(times_S_astro[0].strftime("%Y-%m-%d %H:%M:%S")))
print("\tCivil Twilight End:\t %s [UTC]" %(times_S_civ[0].strftime("%Y-%m-%d %H:%M:%S")))
print("\tRising:\t\t\t %s [UTC] \t Az. from North %.2f [deg]" %(times_S[0].strftime("%Y-%m-%d %H:%M:%S"), A_S[0]))
print("\tTransit:\t\t %s [UTC] \t Az. from North %.2f [deg]" %(times_S[1].strftime("%Y-%m-%d %H:%M:%S"), A_S[1]))
print("\tSetting:\t\t %s [UTC] \t Az. from North %.2f [deg]" %(times_S[2].strftime("%Y-%m-%d %H:%M:%S"), A_S[2]))
print("\tCivil Twilight Start:\t %s [UTC]" %(times_S_civ[2].strftime("%Y-%m-%d %H:%M:%S")))
print("\tAstro Twilight Start:\t %s [UTC]" %(times_S_astro[2].strftime("%Y-%m-%d %H:%M:%S")))


times, h, A = TraceIt(dd, mm, yy, ra_ICRS, dec_ICRS, phi, L, height, obj = 'star', debug = False, plot = True)

print("\nObject:")
print("\tRising:\t\t %s [UTC] \t Alt. %.2f [deg] \t Az. from North %.2f [deg]" %(times[0].strftime("%Y-%m-%d %H:%M:%S"), h[0], A[0]))
print("\tTransit:\t %s [UTC] \t Alt. %.2f [deg] \t Az. from North %.2f [deg] \t Airmass %.3f" %(times[1].strftime("%Y-%m-%d %H:%M:%S"), h[1], A[1], 1./np.cos((90 - h[1])*np.pi/180)))
print("\tSetting:\t %s [UTC] \t Alt. %.2f [deg] \t Az. from North %.2f [deg]" %(times[2].strftime("%Y-%m-%d %H:%M:%S"), h[2], A[2]))

plt.show()