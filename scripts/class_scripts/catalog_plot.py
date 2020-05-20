''' catalog_plot - contains a series of functions used to create plots'''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


def binned_percentiles_three_props(x_quantity, y_quantity, z_quantity, x_bins, z_percentiles, z_percentile_labels, ax, colors=['lightslategrey', 'slateblue', 'dodgerblue', 'rebeccapurple'], extrema=False):
	'''
	Given three properties (x, y, z), this bins in the x direction. In each x bin, the population is split on percentiles
	in z, and the medians of x and y are found. The plot returns these median values (x vs y). 

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

	ax : plt.axis object
		Axis to plot onto.
	
	extrema : boolean
		Optional arg. Set to True if you only want the extrema percentiles plotted.
	
	---
	Output :

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

	for m in np.unique(z_digitized):
		if extrema == True:
			if (m == np.max(np.unique(z_digitized))) or (m == 0):
				label = str(z_percentile_labels[m])
				ax.errorbar(x_medians[:,m], y_medians[:,m], yerr=y_error[:,m],
							marker='H', markersize=5, capsize=5, alpha=0.75, color=colors[m], label=label)
		else :
			label = str(z_percentile_labels[m])
			ax.errorbar(x_medians[:,m], y_medians[:,m], yerr=y_error[:,m],
						marker='H', markersize=5, capsize=5, alpha=0.75, color=colors[m], label=label)
    
	return
