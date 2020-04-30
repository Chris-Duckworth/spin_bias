'''
spin_bias_funcs : set of functions used in making plots 
'''

import numpy as np 
import pandas as pd


def create_matched_mass_sample_unique(masses, tab, col_name='stel_mass_x'):
	'''
	Given a set of masses, this returns a control sample with a similar mass distribution. 
	Each match in the control sample will be a unique galaxy.
	Make sure that the you have removed the comparison sample from your table.
	'''
	iter_tab = tab
	indices = np.array([])
	for mass in masses:
		idx = iter_tab.index[(np.abs(iter_tab[col_name].values - mass)).argmin()]
		indices = np.append(indices, idx)
		iter_tab = iter_tab.drop(idx)
	return tab.loc[indices]


def return_tabs(intab, min_mass=0, max_mass=10**12, mass_col='halo_mass_stel'):
	'''
	Given a complete table of galaxies with associated distances to morphological CW 
	features, this returns individual tables (including a mass matched table for the 
	filament sample). mass_col defines the column on which the control sample is matched.
	'''

	# filament sample.	
	mask = (intab.log_dskel_norm.values < np.log10(1.5)) & (intab.log_dnode_norm.values > np.log10(2)) & (intab[mass_col].values > min_mass) & (intab[mass_col].values < max_mass)   
	filament_tab = intab[mask]
	residual_tab = intab.drop(filament_tab.index)
	control_tab = create_matched_mass_sample_unique(filament_tab[mass_col].values, residual_tab, col_name=mass_col)  

	# node sample.
	node_mask = (intab.log_dnode_norm.values < np.log10(1)) & (intab[mass_col].values > min_mass) & (intab[mass_col].values < max_mass)
	node_tab = intab[node_mask]

	# no cw
	ncw_mask = (intab.log_dnode_norm.values > np.log10(1)) & (intab.log_dskel_norm.values > np.log10(1)) & (intab[mass_col].values > min_mass) & (intab[mass_col].values < max_mass)
	ncw_tab = intab[ncw_mask]

	return filament_tab, control_tab, node_tab, ncw_tab, residual_tab