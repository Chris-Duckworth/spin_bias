''' catalog_plot - contains a series of functions used to create plots'''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as stats
import catalog_process


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
	
	# calling processing method to find binned medians for each percentile.
	x_medians, y_medians, y_error = catalog_process.compute_binned_y_three_prop(x_quantity, y_quantity, z_quantity, x_bins, z_percentiles)

	# computing unique digitised.
	z_bins = np.percentile(z_quantity, z_percentiles)
	z_digitized = np.digitize(z_quantity, z_bins)

	if np.unique(z_digitized).shape[0] == 1:
		label = str(z_percentile_labels)
		ax.errorbar(x_medians, y_medians, yerr=y_error, marker='H', markersize=5, capsize=5, alpha=0.75, color=colors[0], label=label)
	
	else:
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
