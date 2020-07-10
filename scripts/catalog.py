''' catalog.py - contains the catalog class'''

import numpy as np
import pandas as pd
import catalog_init
import catalog_process
import catalog_plot


class Catalog:
	def __init__(self, basepath, version='mpl9', match_to_lim=False):
		'''initialising loads basic catalog with MaNGA info'''
		self.df = catalog_init.load_basic(basepath, version, match_to_lim)

	def match_to_cw(self, basepath, version='mpl9', sigma=3, return_plot=False):
		'''matching to CW info if required'''
		self.df = catalog_init.match_to_cw(basepath, self.df, version, sigma, return_plot)
	
	def group_membership(self, sel='cen', group_cat='lim', keep_zero_mass=False):
		'''selecting group membership'''
		self.df = catalog_init.group_membership(self.df, sel, group_cat, keep_zero_mass)
    
	def select_morphology(self, morphology, vote_frac = 0.7, Tsplit = 3):
		'''finding galaxies of certain morphology only'''
		self.df = catalog_init.select_morphology(self.df, morphology, vote_frac, Tsplit)
	
	def select_cw_enviro(self, feature, node_dist=2, filament_dist=1.5):
		'''finding galaxies near to cw features only'''
		self.df = catalog_init.select_cw_enviro(self.df, feature, node_dist, filament_dist)
    
	def return_matched_subsample(self, col_name, match_values):
		'''slicing table to return nearest match to each supplied value (match_value)'''
		return catalog_init.return_matched_subsample(self.df, col_name, match_values)
    
	def compute_expected_y(self, x_col, y_col, method="polynomial", return_plot=False, plot_x_extent=[11.5, 14], plot_y_extent=[0, 1], deg=1, n_neighbours=10):
		'''returns average function'''
		return catalog_process.compute_expected_y(self.df[x_col].values, self.df[y_col].values, method, return_plot, plot_x_extent, plot_y_extent, deg, n_neighbours)
        
	def compute_binned_percentile_three_props(self, x_col, y_col, z_col, x_bins, z_percentiles):
		'''
		Given three properties (x, y, z), this bins in the x direction. In each x bin, 
		the population is split on percentiles in z (set = [0] if you don't want to split
		on z), and the medians of x and y are returned, along with y_errors.
		'''
		return catalog_process.compute_binned_y_three_prop(self.df[x_col].values, self.df[y_col].values, self.df[z_col].values, x_bins, z_percentiles)

	def plot_scatter(self, x_col, y_col, ax, **kwargs):
		'''scatter plot of defined columns''' 
		return catalog_plot.plot_scatter(self.df[x_col].values, self.df[y_col].values, ax, **kwargs)
		
	def plot_histogram(self, x_col, ax, **kwargs):
		'''histogram plot of defined column''' 
		return catalog_plot.plot_histogram(self.df[x_col].values, ax, **kwargs)	
		
	def plot_hexbin(self, x_col, y_col, ax, **kwargs):
		'''hexbin plot of defined columns''' 
		return catalog_plot.plot_hexbin(self.df[x_col].values, self.df[y_col].values, ax, **kwargs)	
	
	def plot_binned_percentiles_three_props(self, x_col, y_col, z_col, x_bins, z_percentiles, z_percentile_labels, ax, colors=['lightslategrey', 'slateblue', 'dodgerblue', 'rebeccapurple'], extrema=False):
		'''
		Given three properties (x, y, z), this bins in the x direction. In each x bin, 
		the population is split on percentiles in z, and the medians of x and y are found. 
		The plot returns these median values (x vs y). 
		'''    	
		return catalog_plot.plot_binned_percentiles_three_props(self.df[x_col].values, self.df[y_col].values, self.df[z_col].values, x_bins, z_percentiles, z_percentile_labels, ax, colors, extrema)
		
	def plot_binned_percentiles_three_props_residuals(self, x_col, y_col, z_col, x_bins, z_percentiles, z_percentile_labels, p, ax, colors=['lightslategrey', 'slateblue', 'dodgerblue', 'rebeccapurple'], extrema=False):
		'''
		Given three properties (x, y, z), this bins in the x direction. In each x bin, 
		the population is split on percentiles in z, and the medians of x and y are found. 
		The plot returns the median value (x vs y) residuals with respect to the overall 
		population average (as defined by function p; where y = p(x) for total pop. )
		'''    	
		return catalog_plot.plot_binned_percentiles_three_props_residuals(self.df[x_col].values, self.df[y_col].values, self.df[z_col].values, x_bins, z_percentiles, z_percentile_labels, p, ax, colors, extrema)