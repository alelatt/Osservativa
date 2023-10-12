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

def PixelPlot(board, title = "", color_scale = 'linear', color_scheme = 'Spectral', color_bounds = [0,1]):
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
	im = ax.pcolormesh(board, cmap = cmap, norm = color_scale, vmin = color_bounds[0], vmax = color_bounds[1])
	f.colorbar(im, ax=ax)

	return


def Visualise(board):
	PixelPlot(board)
	points = np.sort(board.flatten())
	plt.figure(figsize = [10,10], dpi = 100, layout = 'tight')
	plt.hist(points, bins = 25)
	plt.show()

	choice = input("Want to cut? (y/n) ")
	if choice == 'y':
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

		mean_under = np.mean(under_p)
		sigma_under = np.std(under_p)
		mean_over = np.mean(over_p)
		sigma_over = np.std(over_p)

		plt.figure(figsize = [10,10], dpi = 100, layout = 'tight')
		plt.hist(under_p, bins = 25)
		plt.axvline(mean_under, color = 'k')
		plt.axvline(mean_under + sigma_under, color = 'r')
		plt.axvline(mean_under - sigma_under, color = 'r')

		plt.figure(figsize = [10,10], dpi = 100, layout = 'tight')
		plt.hist(over_p, bins = 25)
		plt.axvline(mean_over, color = 'k')
		plt.axvline(mean_over + sigma_over, color = 'r')
		plt.axvline(mean_over - sigma_over, color = 'r')
		plt.show()

		print("Under treshold mean: %.3f pm %.3f" %(mean_under, sigma_under))
		print("Over treshold mean: %.3f pm %.3f" %(mean_over, sigma_over))

	else:
		mean = np.mean(points)
		sigma = np.std(points)

		plt.figure(figsize = [10,10], dpi = 100, layout = 'tight')
		plt.hist(points, bins = 25)
		plt.axvline(mean, color = 'k')
		plt.axvline(mean + sigma, color = 'r')
		plt.axvline(mean - sigma, color = 'r')
		plt.show()

		print("Mean: %.3f pm %.3f" %(np.mean(points), np.std(points)))

	return

lucy = np.genfromtxt("out_stats_5000_70_l.txt")

Visualise(lucy)