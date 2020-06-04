''' catalog_process - contains a series of functions which process all of the data in the
catalog class
'''

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
from scipy.interpolate import interp1d
import scipy.stats as stats

def compute_expected_y(x, y, method='polynomial', return_plot=False, deg=1, n_neighbours=5): 
	'''
	Given a set of x and y values, this computes the expected y value (or average) as a 
	function of x. The defined method will compute this via fitting a polynomial, rolling
	average, ...
	'''

	assert method in ["polynomial", "running_mean", "window_mean"], "Undefined method!"

	if method == "polynomial":
		# fitting polynomial to x - y plane.
		z = np.polyfit(x, y, deg)
		p = np.poly1d(z)

	elif method == "running_mean":
		# sorting y values by x ordering.
		x_sorted = np.sort(x)
		y_sorted = y[np.argsort(x)]
		y_running_mean = uniform_filter1d(y_sorted, size=n_neighbours)

		# creating interpolation function to be used in future.
		p = interp1d(x_sorted, y_running_mean)

	# returning plot of fitted polynomial with points if required.
	if return_plot == True:
		plt.cla()
		plt.close()
		fig, ax = plt.subplots()
		ax.hexbin(x, y, cmap=plt.cm.Reds, gridsize=30, extent=(11.5, 15, 0, 1))
		# plotting polynomial over same range as the data.
		xrange = np.linspace(np.min(x), np.max(x))
		yrange = p(xrange)
		ax.plot(xrange, yrange, color='skyblue', linewidth=5, alpha=1)
		plt.show()
		
	return p


def compute_binned_y_three_prop(x_quantity, y_quantity, z_quantity, x_bins, z_percentiles):
	'''
	Given three properties (x, y, z), this bins in the x direction. 
	In each x bin, the population is split on percentiles in z, and the medians of x and 
	y are found. The medians for x and y in each bin (with y-errors) are returned.
	If you have no interest in splitting on z - then set z_percentiles = [0]. 

	---
	Input : 

	x : np.array()
		Array of x values (1D)

	y : np.array()
		Array of y values (1D)
	
	z : np.array()
		Array of z values (1D)

	x_bins : np.array()
		Set of bin boundaries in x direction.
	
	z_percentiles : np.array()
		Set of percentiles to split in z dimension (applied individually for each bin).
		Set to [0] if you just want the average in each x_bin for all points.
	
	---
	Output :
	
	x_medians : np.array(n_x_bins, n_quartiles)
		Array of x medians in each x bin. (2D) with values for each quartile in each row.
	
	y_medians : np.array(n_x_bins, n_quartiles)
		Array of y medians in each x bin. (2D) with values for each quartile in each row.

	y_errors : np.array(n_x_bins, n_quartiles)
		Array of y errors in each x bin. (2D) with values for each quartile in each row.

	'''

	# defining empty lists to append to.
	x_medians = []
	y_medians = []
	y_error = []

	# binning x values.
	x_digitized = np.digitize(x_quantity, x_bins)

	# in each x bin, returning values and further binning them into percentiles of z.
	for n in np.unique(x_digitized):
		# mask for values in the bin.
		x_mask = (x_digitized == n)

		# finding bin edges for percentiles.
		z_bins = np.percentile(z_quantity[x_mask], z_percentiles)
		z_digitized = np.digitize(z_quantity[x_mask], z_bins)

		# appending all median values to be plotted.
		for m in np.unique(z_digitized):
			# finding mask for bin within the bin.
			z_mask = (z_digitized == m)
			# computing medians for z percentile in x bin.
			x_medians.append(np.median(x_quantity[x_mask][z_mask]))
			y_medians.append(np.median(y_quantity[x_mask][z_mask]))
			y_error.append(stats.sem(y_quantity[x_mask][z_mask]))

	# converting to np.arrays and reshaping.
	x_medians = np.array(x_medians).reshape(np.unique(x_digitized).shape[0], np.unique(z_digitized).shape[0])
	y_medians = np.array(y_medians).reshape(np.unique(x_digitized).shape[0], np.unique(z_digitized).shape[0])
	y_error = np.array(y_error).reshape(np.unique(x_digitized).shape[0], np.unique(z_digitized).shape[0])

	return x_medians, y_medians, y_error



def compute_residuals_binned_y_three_prop(x_quantity, y_quantity, z_quantity, p, x_bins, z_percentiles):
	'''
	Given three properties (x, y, z), this bins in the x direction. 
	In each x bin, the population is split on percentiles in z, the residuals with respect 
	to a function of the form y = p(x) is then found for each point split on x and z. 
	The median residual is then returned, with standard error on the mean.
	If you have no interest in splitting on z - then set z_percentiles = [0]. 

	---
	Input : 

	x : np.array()
		Array of x values (1D)

	y : np.array()
		Array of y values (1D)
	
	z : np.array()
		Array of z values (1D)

	p : function
		Function which returns expected y as a function of x.

	x_bins : np.array()
		Set of bin boundaries in x direction.
	
	z_percentiles : np.array()
		Set of percentiles to split in z dimension (applied individually for each bin).
		Set to [0] if you just want the average in each x_bin for all points.
	
	---
	Output :
	
	x_medians : np.array(n_x_bins, n_quartiles)
		Array of x medians in each x bin. (2D) with values for each quartile in each row.
	
	y_medians : np.array(n_x_bins, n_quartiles)
		Array of y medians in each x bin. (2D) with values for each quartile in each row.

	y_errors : np.array(n_x_bins, n_quartiles)
		Array of y errors in each x bin. (2D) with values for each quartile in each row.

	'''

	# defining empty lists to append to.
	x_medians = []
	y_medians = []
	y_error = []

	# binning x values.
	x_digitized = np.digitize(x_quantity, x_bins)

	# in each x bin, returning values and further binning them into percentiles of z.
	for n in np.unique(x_digitized):
		# mask for values in the bin.
		x_mask = (x_digitized == n)

		# finding bin edges for percentiles.
		z_bins = np.percentile(z_quantity[x_mask], z_percentiles)
		z_digitized = np.digitize(z_quantity[x_mask], z_bins)

		# appending all median values to be plotted.
		for m in np.unique(z_digitized):
			# finding mask for bin within the bin.
			z_mask = (z_digitized == m)
			
			# computing medians for z percentile in x bin.
			x_medians.append(np.mean(x_quantity[x_mask][z_mask]))
			y_error.append(stats.sem(y_quantity[x_mask][z_mask]))
			
			# finding residual for every point and computing median of all points.
			y_medians.append(np.mean(y_quantity[x_mask][z_mask] - p(x_quantity[x_mask][z_mask]) ))


	# converting to np.arrays and reshaping.
	x_medians = np.array(x_medians).reshape(np.unique(x_digitized).shape[0], np.unique(z_digitized).shape[0])
	y_medians = np.array(y_medians).reshape(np.unique(x_digitized).shape[0], np.unique(z_digitized).shape[0])
	y_error = np.array(y_error).reshape(np.unique(x_digitized).shape[0], np.unique(z_digitized).shape[0])

	return x_medians, y_medians, y_error