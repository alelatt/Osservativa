import numpy as np
from numpy.random import uniform, pareto, poisson
from matplotlib import pyplot as plt
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from skimage.util import random_noise
from skimage.restoration import richardson_lucy
import os
import time

def PixelPlot(board, title = "", color_scale = 'linear', color_scheme = 'Spectral'):
	"""
	Makes a colormap plot of the given board

	Inputs:
		plot_num		Identifier number for the plot (must be different for each plot, else it will be overwritten)
		board			The board to plot
		color_scale		Scale of the colormap, must be 'linear', 'log' or other specified keywords
		color_scheme	Color scheme of the colormap
	"""

	f = plt.figure(figsize = [10,10], dpi = 100, layout = 'tight')
	ax = f.add_subplot(1,1,1)
	ax.set_title(title)
	cmap = plt.get_cmap(color_scheme)
	im = ax.pcolormesh(board, cmap = cmap, norm = color_scale)
	f.colorbar(im, ax=ax)

	return


def Visualise(board):
	points = np.sort(board.flatten())
	plt.figure(figsize = [10,10], dpi = 100, layout = 'tight')
	plt.hist(points, bins = 25)
	plt.show()

	cut = float(input("Cut at: "))

	under = np.copy(board)
	over = np.copy(board)

	under[board < cut] = 1
	over[board < cut] = 0
	under[board >= cut] = 0
	over[board >= cut] = 1

	PixelPlot(under, color_scheme = 'gray')
	PixelPlot(over, color_scheme = 'gray')
	plt.show()

	under_p = points[points < cut]
	over_p = points[points >= cut]
	print("Under treshold: %.3f pm %.3f" %(np.mean(under_p), np.std(under_p)))
	print("Over treshold: %.3f pm %.3f" %(np.mean(over_p), np.std(over_p)))

	return

lucy = np.genfromtxt("out_stats_1000_70_l.txt")

Visualise(lucy)