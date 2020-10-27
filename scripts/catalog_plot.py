''' catalog_plot - contains a series of functions used to create plots'''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as stats
import catalog_process


def default(rcParams):
	'''
	All your favourite rcParams; here in once place!
	'''
	rcParams['axes.linewidth'] = 2.5
	rcParams['xtick.major.size'] = 5
	rcParams['ytick.major.size'] = 5
	rcParams['xtick.minor.size'] = 3.5
	rcParams['ytick.minor.size'] = 3.5
	rcParams['xtick.major.width'] = 2
	rcParams['ytick.major.width'] = 2
	rcParams['xtick.minor.width'] = 1.5
	rcParams['ytick.minor.width'] = 1.5
	rcParams['xtick.top'] = True
	rcParams['xtick.direction'] = 'in'
	rcParams['ytick.direction'] = 'in'
	rcParams['xtick.labelsize'] = 'x-large'
	rcParams['ytick.labelsize'] = 'x-large'
	rcParams['xtick.major.pad'] = 5
	rcParams['ytick.major.pad'] = 5
	rcParams['lines.linewidth'] = 2.5
	return 	


def plot_scatter(x, y, ax, **kwargs):
	'''
	Simple function which returns scatter plot (x, y) of two defined columns in the data
	frame. 
	---
	Input : 

	x : np.array()
		Array of x values (1D)

	y : np.array()
		Array of y values (1D)
	
	ax : plt.axis object
		Axis to plot onto.
	
	---
	Output :
	
	'''
	return ax.plot(x, y, **kwargs)


def plot_histogram(x, ax, **kwargs):
	'''
	Simple function which returns a histogram plot of a column in the data frame. 
	---
	Input : 

	x : np.array()
		Array of x values (1D)
	
	ax : plt.axis object
		Axis to plot onto.
	
	---
	Output :
	
	'''
	return ax.hist(x, **kwargs)


def plot_hexbin(x, y, ax, **kwargs):
	'''
	Simple function which returns a hexbin plot of a column in the data frame. 
	---
	Input : 

	x : np.array()
		Array of x values (1D)
		
	y : np.array()
		Array of y values (1D)
	
	ax : plt.axis object
		Axis to plot onto.
	
	---
	Output :
	
	'''
	return ax.hexbin(x, y, **kwargs)



def plot_binned_percentiles_three_props(x_quantity, y_quantity, z_quantity, x_bins, z_percentiles, z_percentile_labels, ax, colors=['lightslategrey', 'slateblue', 'dodgerblue', 'rebeccapurple'], extrema=False, linestyle='solid'):
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
		ax.errorbar(x_medians.flatten(), y_medians.flatten(), yerr=y_error.flatten(), marker='H', markersize=5, capsize=5, alpha=1, linewidth=5, color=colors[0], label=label, linestyle=linestyle)
	
	else:
		for m in np.unique(z_digitized):
			if extrema == True:
				if (m == np.max(np.unique(z_digitized))) or (m == 0):
					label = str(z_percentile_labels[m])
					ax.errorbar(x_medians[:,m], y_medians[:,m], yerr=y_error[:,m],
								marker='H', markersize=5, capsize=5, alpha=1, linewidth=5, color=colors[m], label=label, linestyle=linestyle)
			else :
				label = str(z_percentile_labels[m])
				ax.errorbar(x_medians[:,m], y_medians[:,m], yerr=y_error[:,m],
							marker='H', markersize=5, capsize=5, alpha=1, linewidth=5, color=colors[m], label=label, linestyle=linestyle)
    
	return


def plot_binned_percentiles_three_props_residuals(x_quantity, y_quantity, z_quantity, x_bins, z_percentiles, z_percentile_labels, p, ax, colors=['lightslategrey', 'slateblue', 'dodgerblue', 'rebeccapurple'], extrema=False):
	'''
	Given three properties (x, y, z), this bins in the x direction. In each x bin, 
	the population is split on percentiles in z, and the medians of x and y are found. 
	The plot returns the median value (x vs y) residuals with respect to the overall 
	population average (as defined by function p; where y = p(x) for total pop. )

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

	p : scipy,interpolate function
		Pre-defined function which returns y value for supplied x value(s).

	ax : plt.axis object
		Axis to plot onto.
	
	extrema : boolean
		Optional arg. Set to True if you only want the extrema percentiles plotted.
	
	---
	Output :
	
	'''
	
	# calling processing method to find binned medians for each percentile.
	x_medians, y_medians, y_error = catalog_process.compute_residuals_binned_y_three_prop(x_quantity, y_quantity, z_quantity, p, x_bins, z_percentiles)
	
	# computing unique digitised.
	z_bins = np.percentile(z_quantity, z_percentiles)
	z_digitized = np.digitize(z_quantity, z_bins)
	
	if np.unique(z_digitized).shape[0] == 1:
		label = str(z_percentile_labels)
		ax.errorbar(x_medians, y_medians, yerr=y_error, marker='H', markersize=5, capsize=5, linewidth=4, alpha=0.6, color=colors[0], label=label)
	
	else:
		for m in np.unique(z_digitized):
			if extrema == True:
				if (m == np.max(np.unique(z_digitized))) or (m == 0):
					label = str(z_percentile_labels[m])
					ax.errorbar(x_medians[:,m], y_medians[:,m], yerr=y_error[:,m],
								marker='H', markersize=5, capsize=5, alpha=0.6, linewidth=4, color=colors[m], label=label)
			else :
				label = str(z_percentile_labels[m])
				ax.errorbar(x_medians[:,m], y_medians[:,m], yerr=y_error[:,m],
							marker='H', markersize=5, capsize=5, alpha=0.6, linewidth=4, color=colors[m], label=label)
    
	return

