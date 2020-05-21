''' catalog.py - contains the catalog class'''

import numpy as np
import pandas as pd
import catalog_init
import catalog_process
import catalog_plot


class Catalog:
	def __init__(self, basepath):
		'''initialising loads basic catalog with MaNGA info'''
		self.df = catalog_init.load_basic(basepath)

	def match_to_cw(self, basepath, return_plot=False):
		'''matching to CW info if required'''
		self.df = catalog_init.match_to_cw(basepath, self.df, return_plot)

	def remove_satellites(self):
		'''finding centrals only'''
		self.df = catalog_init.centrals_only(self.df)
    
	def select_morphology(self, morphology, vote_frac = 0.7, Tsplit = 3):
		'''finding galaxies of certain morphology only'''
		self.df = catalog_init.select_morphology(self.df, morphology, vote_frac, Tsplit)
	
	def select_cw_enviro(self, feature, node_dist=2, filament_dist=1.5):
		'''finding galaxies near to cw features only'''
		self.df = catalog_init.select_cw_enviro(self.df, feature, node_dist, filament_dist)
    
	def compute_expected_y(self, x_col, y_col, method="polynomial", return_plot=False, deg=1, n_neighbours=10):
		'''returns average function'''
		return catalog_process.compute_expected_y(self.df[x_col].values, self.df[y_col].values, method, return_plot, deg, n_neighbours)
        
	def compute_binned_percentile_three_props(self, x_col, y_col, z_col, x_bins, z_percentiles):
		'''
		Given three properties (x, y, z), this bins in the x direction. In each x bin, 
		the population is split on percentiles in z (set = [0] if you don't want to split
		on z), and the medians of x and y are returned, along with y_errors.
		'''
		return catalog_process.compute_binned_y_three_prop(self.df[x_col].values, self.df[y_col].values, self.df[z_col].values, x_bins, z_percentiles)

	def plot_binned_percentiles_three_props(self, x_col, y_col, z_col, x_bins, z_percentiles, z_percentile_labels, ax, colors=['lightslategrey', 'slateblue', 'dodgerblue', 'rebeccapurple'], extrema=False):
		'''
		Given three properties (x, y, z), this bins in the x direction. In each x bin, 
		the population is split on percentiles in z, and the medians of x and y are found. 
		The plot returns these median values (x vs y). 
		'''    	
		return catalog_plot.binned_percentiles_three_props(self.df[x_col].values, self.df[y_col].values, self.df[z_col].values, x_bins, z_percentiles, z_percentile_labels, ax, colors, extrema)