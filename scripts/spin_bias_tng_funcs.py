'''
spin_bias_tng_funcs : set of functions used in making comparison plots on the tng sample.
'''

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import kin_morph_funcs as knf


def plot_lambdaR_mhalo_single_population(tab, ax, color='chartreuse', lower=None, upper=None, label=None):
	'''
	Given an overall population of galaxies (with lambdaR, Mhalo, skeleton distances) this 
	separates based on halo mass, and makes a plot with the average values for the total . Returns ax.
	
	---
	Input :
	
	tab : pandas.dataframe object
		Table containing all galaxies (normally filtered on morphology prior to this). Table
		should contain CW distances, masses etc. and lambda_R measure.
	
	ax : matplotlib.axis object
	
	color : string
		Optional arg. Color for population
	
	lower : float
		Optional arg. Can set own boundaries on how to divide population into 3.
	
	upper : float
		Optional arg. Can set own boundaries on how to divide population into 3.
	
	label : string
		Optional arg. label for population
	---
	Output : 
	
	ax : 
	'''

	if (upper is None) or (lower is None):
		# If either undefined, finding percentiles (splitting at 33, 66)
		upper = np.percentile(np.log10(tab.halo_mass.values), 66)
		lower = np.percentile(np.log10(tab.halo_mass.values), 33)

	# sub-selecting percentile tabs.
	upper_tab = tab[(np.log10(tab.halo_mass.values) > upper)]
	middle_tab = tab[(np.log10(tab.halo_mass.values) <= upper) & (np.log10(tab.halo_mass.values) >= lower)]
	lower_tab = tab[np.log10(tab.halo_mass.values) < lower]
	
	# plotting filament samples.
	ax.plot(np.log10(lower_tab.halo_mass.values), lower_tab.lambda_r.values, color=color, marker='v', linestyle='None', alpha=0.3, markersize=2)
	ax.plot(np.log10(middle_tab.halo_mass.values), middle_tab.lambda_r.values, color=color, marker='v', linestyle='None', alpha=0.3, markersize=2)
	ax.plot(np.log10(upper_tab.halo_mass.values), upper_tab.lambda_r.values, color=color, marker='v', linestyle='None', alpha=0.3, markersize=2)

	# overplotting the average values in each bin.
	ax.errorbar([np.median(np.log10(lower_tab.halo_mass.values)), np.median(np.log10(middle_tab.halo_mass.values)), np.median(np.log10(upper_tab.halo_mass.values))],
				[np.median(lower_tab.lambda_r.values), np.median(middle_tab.lambda_r.values), np.median(upper_tab.lambda_r.values)], 
				yerr=[stats.sem(lower_tab.lambda_r.values), stats.sem(middle_tab.lambda_r.values), stats.sem(upper_tab.lambda_r.values)], 
				 color=color, marker='H', markersize=5, capsize=5, label=label)

	# adding percentile vertical lines to denote boundaries.
	ax.axvline(upper, color=color, alpha=0.3, linestyle='dashed')
	ax.axvline(lower, color=color, alpha=0.3, linestyle='dashed')

	# formatting final plot.
	#ax.set_xscale('log')
	ax.set_ylabel(r'$\mathrm{\lambda_{R}(< 1.5R_{e})}$', fontsize=16)
	ax.set_xlabel(r'$\mathrm{M_{halo}}$', fontsize=16)
	ax.set_ylim((0, 1))
	return ax


def plot_sJ_mhalo_single_population(tab, ax, color='chartreuse', lower=None, upper=None, label=None):
	'''
	Given an overall population of galaxies (with sJ (stellar), Mhalo, skeleton distances) this 
	separates based on halo mass, and makes a plot with the average values for the total . Returns ax.
	
	---
	Input :
	
	tab : pandas.dataframe object
		Table containing all galaxies (normally filtered on morphology prior to this). Table
		should contain CW distances, masses etc. and lambda_R measure.
	
	ax : matplotlib.axis object
	
	color : string
		Optional arg. Color for population
	
	lower : float
		Optional arg. Can set own boundaries on how to divide population into 3.
	
	upper : float
		Optional arg. Can set own boundaries on how to divide population into 3.
	
	label : string
		Optional arg. label for population
	---
	Output : 
	
	ax : 
	'''

	if (upper is None) or (lower is None):
		# If either undefined, finding percentiles (splitting at 33, 66)
		upper = np.percentile(np.log10(tab.halo_mass.values), 66)
		lower = np.percentile(np.log10(tab.halo_mass.values), 33)

	# sub-selecting percentile tabs.
	upper_tab = tab[(np.log10(tab.halo_mass.values) > upper)]
	middle_tab = tab[(np.log10(tab.halo_mass.values) <= upper) & (np.log10(tab.halo_mass.values) >= lower)]
	lower_tab = tab[np.log10(tab.halo_mass.values) < lower]
	
	# plotting filament samples.
	ax.plot(np.log10(lower_tab.halo_mass.values), lower_tab.mag_sJ_stel.values, color=color, marker='v', linestyle='None', alpha=0.3, markersize=2)
	ax.plot(np.log10(middle_tab.halo_mass.values), middle_tab.mag_sJ_stel.values, color=color, marker='v', linestyle='None', alpha=0.3, markersize=2)
	ax.plot(np.log10(upper_tab.halo_mass.values), upper_tab.mag_sJ_stel.values, color=color, marker='v', linestyle='None', alpha=0.3, markersize=2)

	# overplotting the average values in each bin.
	ax.errorbar([np.median(np.log10(lower_tab.halo_mass.values)), np.median(np.log10(middle_tab.halo_mass.values)), np.median(np.log10(upper_tab.halo_mass.values))],
				[np.median(lower_tab.mag_sJ_stel.values), np.median(middle_tab.mag_sJ_stel.values), np.median(upper_tab.mag_sJ_stel.values)], 
				yerr=[stats.sem(lower_tab.mag_sJ_stel.values), stats.sem(middle_tab.mag_sJ_stel.values), stats.sem(upper_tab.mag_sJ_stel.values)], 
				 color=color, marker='H', markersize=5, capsize=5, label=label)

	# adding percentile vertical lines to denote boundaries.
	ax.axvline(upper, color=color, alpha=0.3, linestyle='dashed')
	ax.axvline(lower, color=color, alpha=0.3, linestyle='dashed')

	# formatting final plot.
	#ax.set_xscale('log')
	ax.set_ylabel(r'$\mathrm{j_{stellar}}$', fontsize=16)
	ax.set_xlabel(r'$\mathrm{M_{halo}}$', fontsize=16)
	return ax

