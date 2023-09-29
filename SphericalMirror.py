import numpy as np
from matplotlib import pyplot as plt
from numpy.random import uniform

R = 1
nsamples = int(1e5)
bound = R/4
focalz = R/2

def RejectionSampling(nsamples, bound):
	gen = uniform(-bound,bound, size = (nsamples,2))
	x = gen[:,0]
	y = gen[:,1]
	for i in range(0,len(gen)):
		if x[i]**2 + y[i]**2 > bound**2:
			x[i] = np.inf
			y[i] = np.inf
	x = x[np.isfinite(x)]
	y = y[np.isfinite(y)]
	return np.vstack((x,y)).T


def Model(x,y):
	z = np.sqrt(R**2 - x**2 - y**2)

	verin = np.array([np.sqrt(0.5),np.sqrt(0.5),1])/np.sqrt(2)

	vercen = -np.array([x,y,z])/np.sqrt(x**2 + y**2 + z**2)

	verout = verin - (2*np.dot(verin,vercen))*vercen

	k = focalz/verout[2]

	xp = k*verout[0]
	yp = k*verout[1]

	return np.array([xp,yp])

inn = RejectionSampling(nsamples,bound)
out = np.zeros(np.shape(inn))

for i in range(0,len(inn)):
	print("%.2f %%" %(i/len(inn)*100))
	out[i] = Model(inn[i,0],inn[i,1])

plt.figure("Input", figsize = [10,10])
#plt.plot(inn[:,0],inn[:,1],'.')
plt.hist2d(inn[:,0], inn[:,1], 500)

plt.figure("Output over z=R/2", figsize = [10,10])
#plt.plot(out[:,0],out[:,1],'.')
plt.hist2d(out[:,0], out[:,1], 500)

plt.show()