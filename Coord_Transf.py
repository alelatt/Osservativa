import numpy as np

##Precession based on A&A 73 (1979), pp.282-284


def Precession(a):
	#Define precession matrix
	#a = [zi,z,theta]
	M1 = np.array([np.cos(a[0])*np.cos(a[1])*np.cos(a[2]) - np.sin(a[0])*np.sin(a[1]), -np.sin(a[0])*np.cos(a[1])*np.cos(a[2]) - np.cos(a[0])*np.sin(a[1]), -np.cos(a[1])*np.sin(a[2])])
	M2 = np.array([np.cos(a[0])*np.sin(a[1])*np.cos(a[2]) + np.sin(a[0])*np.cos(a[1]), -np.sin(a[0])*np.sin(a[1])*np.cos(a[2]) + np.cos(a[0])*np.cos(a[1]), -np.sin(a[1])*np.sin(a[2])])
	M3 = np.array([np.cos(a[0])*np.sin(a[2]), -np.sin(a[0])*np.sin(a[2]), np.cos(a[2])])

	return np.vstack((M1,M2,M3))

#Position Vector in RA,DEC 
R = np.array([83.45*(np.pi/180),-2.65*(np.pi/180),1])

#Position Vector in equatorial coordinates x,y,z
R2 = np.array([[np.cos(R[1])*np.cos(R[0]), np.cos(R[1])*np.sin(R[0]), np.sin(R[1])]]).T

#Define angles of precession and application of matrix
angles = np.array([0.640312, 0.640532, 0.556855])*(np.pi/180)
M = Precession(angles)
print(M)
R1 = M @ R2

#Inverse transformation to RA,DEC
ra = np.arctan2(R1[1,0],R1[0,0])
dec = np.arcsin(R1[2,0])

#Define constants for Galactic reference frame transformation
aG = 192.85948*(np.pi/180)
dG = 27.12825*(np.pi/180)
lNCP = 122.93192*(np.pi/180)

#Conversion to Galactic reference frame
b = np.arcsin(np.sin(dec)*np.sin(dG) + np.cos(dec)*np.cos(dG)*np.cos(ra-aG))
l = lNCP - np.arcsin((np.cos(dec)*np.sin(ra-aG))/np.cos(b))

b = b*(180/np.pi)
l = l*(180/np.pi)

print(l,b)