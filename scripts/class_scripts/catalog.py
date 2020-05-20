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
    
    def compute_expected_y(self, x_col, y_col, method="polynomial", return_plot=False, deg=1, n_neighbours=10):
        '''returns average function'''
        return catalog_process.compute_expected_y(self.df[x_col].values, self.df[y_col].values, method, return_plot, deg, n_neighbours)