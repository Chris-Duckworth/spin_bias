'''
spin_bias_funcs : set of functions used in making plots 
'''

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import kin_morph_funcs as knf

def plot_default(rcParams):
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
	ncw_mask = (intab.log_dnode_norm.values > np.log10(2)) & (intab.log_dskel_norm.values > np.log10(2)) & (intab[mass_col].values > min_mass) & (intab[mass_col].values < max_mass)
	ncw_tab = intab[ncw_mask]

	return filament_tab, control_tab, node_tab, ncw_tab, residual_tab


def plot_lambdaR_mhalo(tab, fil_color='chartreuse', con_color='lightslategrey', match_col='dtfe_gauss_3Mpc', match_lims=[0,1], label=None):
	'''
	Given an overall population of galaxies (with lambdaR, Mhalo, skeleton distances) this 
	separates based on halo mass, and makes a plot comparing those close to filaments to 
	a control sample.
	
	---
	Input :
	
	tab : pandas.dataframe object
		Table containing all galaxies (normally filtered on morphology prior to this). Table
		should contain CW distances, masses etc. and lambda_R measure.
	
	fil_color : string
		Optional arg. Color for plotting filament sample
	
	con_color : string
		Optional arg. Color for plotting control sample
	
	match_col : string
		Column to match control sample based on. Default is overdensity smoothed on 3Mpc.
	
	match_lims : np.array or list
		Limits on match_col to make matches on. [min, max]
	
	label : string
		Optional label to put in top right hand corner of plot.
	
	---
	Output : 
	
	ax : plt.axis object
		Returns axis object to further manipulate before plotting or saving.
	
	'''
		
	# Finding percentiles (splitting at 33, 66)
	upper = np.percentile(tab.halo_mass_stel.values, 66)
	lower = np.percentile(tab.halo_mass_stel.values, 33)

	# sub-selecting percentile tabs.
	upper_tab, upper_control_tab, _, _, _ = return_tabs(tab[(tab.halo_mass_stel.values > upper) & (tab.halo_mass_stel.values < 14.5)], min_mass=match_lims[0], max_mass=match_lims[1], mass_col=match_col)
	middle_tab, middle_control_tab, _, _, _ = return_tabs(tab[(tab.halo_mass_stel.values <= upper) & (tab.halo_mass_stel.values >= lower)], min_mass=match_lims[0], max_mass=match_lims[1], mass_col=match_col)
	lower_tab, lower_control_tab, _, _, _ = return_tabs(tab[tab.halo_mass_stel.values < lower], min_mass=match_lims[0], max_mass=match_lims[1], mass_col=match_col)

	# setting up figure.
	fig, ax = plt.subplots(1, figsize=(10, 4))

	# plotting filament samples.
	ax.plot(lower_tab.halo_mass_stel.values, lower_tab.lambda_re.values, color=fil_color, marker='v', linestyle='None', alpha=0.3, markersize=2)
	ax.plot(middle_tab.halo_mass_stel.values, middle_tab.lambda_re.values, color=fil_color, marker='v', linestyle='None', alpha=0.3, markersize=2)
	ax.plot(upper_tab.halo_mass_stel.values, upper_tab.lambda_re.values, color=fil_color, marker='v', linestyle='None', alpha=0.3, markersize=2)

	# plotting control samples.
	ax.plot(lower_control_tab.halo_mass_stel.values, lower_control_tab.lambda_re.values, color=con_color, marker='^', linestyle='None', alpha=0.3, markersize=2)
	ax.plot(middle_control_tab.halo_mass_stel.values, middle_control_tab.lambda_re.values, color=con_color, marker='^', linestyle='None', alpha=0.3, markersize=2)
	ax.plot(upper_control_tab.halo_mass_stel.values, upper_control_tab.lambda_re.values, color=con_color, marker='^', linestyle='None', alpha=0.3, markersize=2)

	# overplotting the average values in each bin.
	ax.errorbar([np.median(lower_tab.halo_mass_stel.values), np.median(middle_tab.halo_mass_stel.values), np.median(upper_tab.halo_mass_stel.values)],
				[np.median(lower_tab.lambda_re.values), np.median(middle_tab.lambda_re.values), np.median(upper_tab.lambda_re.values)], 
				yerr=[stats.sem(lower_tab.lambda_re.values), stats.sem(middle_tab.lambda_re.values), stats.sem(upper_tab.lambda_re.values)], 
				 color=fil_color, marker='H', markersize=5, capsize=5, label='$\mathrm{D_{skel} < 1.5Mpc; D_{node} > 2Mpc}$')
    
	ax.errorbar([np.median(lower_control_tab.halo_mass_stel.values), np.median(middle_control_tab.halo_mass_stel.values), np.median(upper_control_tab.halo_mass_stel.values)],
				[np.median(lower_control_tab.lambda_re.values), np.median(middle_control_tab.lambda_re.values), np.median(upper_control_tab.lambda_re.values)], 
				yerr=[stats.sem(lower_control_tab.lambda_re.values), stats.sem(middle_control_tab.lambda_re.values), stats.sem(upper_control_tab.lambda_re.values)], 
				 color=con_color, marker='H', markersize=5, capsize=5, linestyle='dashed', label='Control sample')
    
	# adding percentile vertical lines to denote boundaries.
	ax.axvline(upper, color='k', alpha=0.3, linestyle='dashed')
	ax.axvline(lower, color='k', alpha=0.3, linestyle='dashed')
    
	# adding KS-tests in each mini-panel.
	ks, p = stats.ks_2samp(lower_tab.lambda_re.values, lower_control_tab.lambda_re.values)
	ax.annotate(r'$\mathrm{KS}=$'+str(round(ks, 3))+'\n $p=$'+str(np.round(p, 3))+'\n n='+str(lower_tab.shape[0]), 
				xy=(np.median(lower_tab.halo_mass_stel.values)-0.1, 0.8), xycoords='data', fontsize=11)
    
	ks, p = stats.ks_2samp(middle_tab.lambda_re.values, middle_control_tab.lambda_re.values)
	ax.annotate(r'$\mathrm{KS}=$'+str(round(ks, 3))+'\n $p=$'+str(np.round(p, 3))+'\n n='+str(middle_tab.shape[0]), 
				xy=(np.median(middle_tab.halo_mass_stel.values)-0.1, 0.8), xycoords='data', fontsize=11)
    
	ks, p = stats.ks_2samp(upper_tab.lambda_re.values, upper_control_tab.lambda_re.values)
	ax.annotate(r'$\mathrm{KS}=$'+str(round(ks, 3))+'\n $p=$'+str(np.round(p, 3))+'\n n='+str(upper_tab.shape[0]),
				xy=(np.median(upper_tab.halo_mass_stel.values)-0.1, 0.8), xycoords='data', fontsize=11)
    
	# Adding label to top right corner.
	#ax.annotate(label, xy=(13.65, 0.5), xycoords='data', fontsize=14)
	ax.set_title(label, fontsize=14)

	# formatting final plot.
	#ax.set_xscale('log')
	ax.set_ylabel(r'$\mathrm{\lambda_{R}(< 1.5R_{e})}$', fontsize=16)
	ax.set_xlabel(r'$\mathrm{M_{halo}}$', fontsize=16)
	ax.set_xlim((np.min(tab.halo_mass_stel.values), 14))
	ax.set_ylim((0,1))
	return ax


def plot_mstel_comparison(tab, fil_color='chartreuse', con_color='lightslategrey', match_col='dtfe_gauss_3Mpc', match_lims=[0,1], label=None):
	'''
	This splits a population in the same way to plot_lambdaR_mhalo, to plot the 
	distributions of Mstel in each bin. Used to check consistency between control and 
	filament sample.
	
	---
	Input :

	tab : pandas.dataframe object
		Table containing all galaxies (normally filtered on morphology prior to this). Table
		should contain CW distances, masses etc. and lambda_R measure.

	fil_color : string
		Optional arg. Color for plotting filament sample

	con_color : string
		Optional arg. Color for plotting control sample

	match_col : string
		Column to match control sample based on. Default is overdensity smoothed on 3Mpc.

	match_lims : np.array or list
		Limits on match_col to make matches on. [min, max]

	label : string
		Optional label to put in top right hand corner of plot.

	---
	Output : 

	ax : plt.axis object
		Returns axis object to further manipulate before plotting or saving.

	'''

	# Finding percentiles (splitting at 33, 66)
	upper = np.percentile(tab.halo_mass_stel.values, 66)
	lower = np.percentile(tab.halo_mass_stel.values, 33)
	
	# sub-selecting percentile tabs.
	upper_tab, upper_control_tab, _, _, _ = return_tabs(tab[(tab.halo_mass_stel.values > upper) & (tab.halo_mass_stel.values < 14.5)], min_mass=match_lims[0], max_mass=match_lims[1], mass_col=match_col)
	middle_tab, middle_control_tab, _, _, _ = return_tabs(tab[(tab.halo_mass_stel.values <= upper) & (tab.halo_mass_stel.values >= lower)], min_mass=match_lims[0], max_mass=match_lims[1], mass_col=match_col)
	lower_tab, lower_control_tab, _, _, _ = return_tabs(tab[tab.halo_mass_stel.values < lower], min_mass=match_lims[0], max_mass=match_lims[1], mass_col=match_col)
	
	# setting up figure.
	fig, ax = plt.subplots(1,3, figsize=(13, 3.5), sharex='all', sharey='all')
	bins = np.logspace(8.5, 12, 35)
	knf.histerr(lower_tab.nsa_elpetro_mass.values, ax[0], bins=bins, color=fil_color, median=True, label='$\mathrm{D_{skel} < 1.5Mpc; D_{node} > 2Mpc}$')
	knf.histerr_fill(lower_tab.nsa_elpetro_mass.values, ax[0], bins=bins, color=fil_color, median=False, label=None)
	knf.histerr(lower_control_tab.nsa_elpetro_mass.values, ax[0], bins=bins, color=con_color, median=True, label='Control sample')
	knf.histerr_fill(lower_control_tab.nsa_elpetro_mass.values, ax[0], bins=bins, color=con_color, median=False, label=None)
	
	knf.histerr(middle_tab.nsa_elpetro_mass.values, ax[1], bins=bins, color=fil_color, median=True)
	knf.histerr_fill(middle_tab.nsa_elpetro_mass.values, ax[1], bins=bins, color=fil_color, median=False)
	knf.histerr(middle_control_tab.nsa_elpetro_mass.values, ax[1], bins=bins, color=con_color, median=True)
	knf.histerr_fill(middle_control_tab.nsa_elpetro_mass.values, ax[1], bins=bins, color=con_color, median=False)

	knf.histerr(upper_tab.nsa_elpetro_mass.values, ax[2], bins=bins, color=fil_color, median=True)
	knf.histerr_fill(upper_tab.nsa_elpetro_mass.values, ax[2], bins=bins, color=fil_color, median=False)
	knf.histerr(upper_control_tab.nsa_elpetro_mass.values, ax[2], bins=bins, color=con_color, median=True)
	knf.histerr_fill(upper_control_tab.nsa_elpetro_mass.values, ax[2], bins=bins, color=con_color, median=False)
	
	# adding KS-tests in each mini-panel.
	ks, p = stats.ks_2samp(lower_tab.nsa_elpetro_mass.values, lower_control_tab.nsa_elpetro_mass.values)
	ax[0].annotate('$\mathrm{KS}=$'+str(round(ks, 3))+'\n $p=$'+str(np.round(p, 3))+'\n n='+str(lower_tab.shape[0]), 
					xy=(10**9.05, 0.32), xycoords='data', fontsize=11)

	ks, p = stats.ks_2samp(middle_tab.nsa_elpetro_mass.values, middle_control_tab.nsa_elpetro_mass.values)
	ax[1].annotate('$\mathrm{KS}=$'+str(round(ks, 3))+'\n $p=$'+str(np.round(p, 3))+'\n n='+str(middle_tab.shape[0]), 
					xy=(10**9.05, 0.32), xycoords='data', fontsize=11)

	ks, p = stats.ks_2samp(upper_tab.nsa_elpetro_mass.values, upper_control_tab.nsa_elpetro_mass.values)
	ax[2].annotate('$\mathrm{KS}=$'+str(round(ks, 3))+'\n $p=$'+str(np.round(p, 3))+'\n n='+str(upper_tab.shape[0]),
					xy=(10**9.05, 0.32), xycoords='data', fontsize=11) 

	ax[0].set_ylabel('PDF', fontsize=13)    
	ax[0].set_xscale('log')
	ax[0].set_xlim([10**9, 10**11.5])

	ax[0].set_xlabel('$\mathrm{M_{stel}}$', fontsize=13)
	ax[1].set_xlabel('$\mathrm{M_{stel}}$', fontsize=13)
	ax[2].set_xlabel('$\mathrm{M_{stel}}$', fontsize=13)
	fig.subplots_adjust(wspace=0, hspace=0.1)
	fig.suptitle(label, fontsize=14)
	
	return ax


def plot_mhalo_comparison(tab, fil_color='chartreuse', con_color='lightslategrey', match_col='dtfe_gauss_3Mpc', match_lims=[0,1], label=None):
	'''
	This splits a population in the same way to plot_lambdaR_mhalo, to plot the 
	distributions of Mhalo in each bin. Used to check consistency between control and 
	filament sample.

	---
	Input :

	tab : pandas.dataframe object
		Table containing all galaxies (normally filtered on morphology prior to this). Table
		should contain CW distances, masses etc. and lambda_R measure.

	fil_color : string
		Optional arg. Color for plotting filament sample

	con_color : string
		Optional arg. Color for plotting control sample

	match_col : string
		Column to match control sample based on. Default is overdensity smoothed on 3Mpc.

	match_lims : np.array or list
		Limits on match_col to make matches on. [min, max]

	label : string
		Optional label to put in top right hand corner of plot.

	---
	Output : 

	ax : plt.axis object
		Returns axis object to further manipulate before plotting or saving.

	'''

	# Finding percentiles (splitting at 33, 66)
	upper = np.percentile(tab.halo_mass_stel.values, 66)
	lower = np.percentile(tab.halo_mass_stel.values, 33)

	# sub-selecting percentile tabs.
	upper_tab, upper_control_tab, _, _, _ = return_tabs(tab[(tab.halo_mass_stel.values > upper) & (tab.halo_mass_stel.values < 14.5)], min_mass=0, max_mass=1, mass_col='dtfe_gauss_3Mpc')
	middle_tab, middle_control_tab, _, _, _ = return_tabs(tab[(tab.halo_mass_stel.values <= upper) & (tab.halo_mass_stel.values >= lower)], min_mass=0, max_mass=1, mass_col='dtfe_gauss_3Mpc')
	lower_tab, lower_control_tab, _, _, _ = return_tabs(tab[tab.halo_mass_stel.values < lower], min_mass=0, max_mass=1, mass_col='dtfe_gauss_3Mpc')

	# setting up figure.
	fig, ax = plt.subplots(1,3, figsize=(13, 3.5), sharex='all', sharey='all')
	bins = np.linspace(11.5, 14.5, 35)
	knf.histerr(lower_tab.halo_mass_stel.values, ax[0], bins=bins, color=fil_color, median=True, label='$\mathrm{D_{skel} < 1.5Mpc; D_{node} > 2Mpc}$')
	knf.histerr_fill(lower_tab.halo_mass_stel.values, ax[0], bins=bins, color=fil_color, median=False, label=None)
	knf.histerr(lower_control_tab.halo_mass_stel.values, ax[0], bins=bins, color=con_color, median=True, label='Control sample')
	knf.histerr_fill(lower_control_tab.halo_mass_stel.values, ax[0], bins=bins, color=con_color, median=False, label=None)
	
	knf.histerr(middle_tab.halo_mass_stel.values, ax[1], bins=bins, color=fil_color, median=True)
	knf.histerr_fill(middle_tab.halo_mass_stel.values, ax[1], bins=bins, color=fil_color, median=False)
	knf.histerr(middle_control_tab.halo_mass_stel.values, ax[1], bins=bins, color=con_color, median=True)
	knf.histerr_fill(middle_control_tab.halo_mass_stel.values, ax[1], bins=bins, color=con_color, median=False)
	
	knf.histerr(upper_tab.halo_mass_stel.values, ax[2], bins=bins, color=fil_color, median=True)
	knf.histerr_fill(upper_tab.halo_mass_stel.values, ax[2], bins=bins, color=fil_color, median=False)
	knf.histerr(upper_control_tab.halo_mass_stel.values, ax[2], bins=bins, color=con_color, median=True)
	knf.histerr_fill(upper_control_tab.halo_mass_stel.values, ax[2], bins=bins, color=con_color, median=False)
	
	# adding KS-tests in each mini-panel.
	ks, p = stats.ks_2samp(lower_tab.halo_mass_stel.values, lower_control_tab.halo_mass_stel.values)
	ax[0].annotate('$\mathrm{KS}=$'+str(round(ks, 3))+'\n $p=$'+str(np.round(p, 3))+'\n n='+str(lower_tab.shape[0]), 
					xy=(11.6, 0.23), xycoords='data', fontsize=11)
	
	ks, p = stats.ks_2samp(middle_tab.halo_mass_stel.values, middle_control_tab.halo_mass_stel.values)
	ax[1].annotate('$\mathrm{KS}=$'+str(round(ks, 3))+'\n $p=$'+str(np.round(p, 3))+'\n n='+str(middle_tab.shape[0]), 
					xy=(11.6, 0.23), xycoords='data', fontsize=11)

	ks, p = stats.ks_2samp(upper_tab.halo_mass_stel.values, upper_control_tab.halo_mass_stel.values)
	ax[2].annotate('$\mathrm{KS}=$'+str(round(ks, 3))+'\n $p=$'+str(np.round(p, 3))+'\n n='+str(upper_tab.shape[0]),
					xy=(11.6, 0.23), xycoords='data', fontsize=11) 
	
	ax[0].set_ylabel('PDF', fontsize=13)    
	ax[0].set_xlim([11.5, 14.5])

	ax[0].set_xlabel('$\mathrm{M_{halo}}$', fontsize=13)
	ax[1].set_xlabel('$\mathrm{M_{halo}}$', fontsize=13)
	ax[2].set_xlabel('$\mathrm{M_{halo}}$', fontsize=13)
	fig.subplots_adjust(wspace=0, hspace=0.1)
	fig.suptitle(label, fontsize=14)
	
	return ax


def plot_dtfe_comparison(tab, fil_color='chartreuse', con_color='lightslategrey', match_col='dtfe_gauss_3Mpc', match_lims=[0,1], label=None):
	'''
	This splits a population in the same way to plot_lambdaR_mhalo, to plot the 
	distributions of DTFE 3Mpc in each bin. Used to check consistency between control and 
	filament sample.

	---
	Input :

	tab : pandas.dataframe object
		Table containing all galaxies (normally filtered on morphology prior to this). Table
		should contain CW distances, masses etc. and lambda_R measure.

	fil_color : string
		Optional arg. Color for plotting filament sample

	con_color : string
		Optional arg. Color for plotting control sample

	match_col : string
		Column to match control sample based on. Default is overdensity smoothed on 3Mpc.

	match_lims : np.array or list
		Limits on match_col to make matches on. [min, max]

	label : string
		Optional label to put in top right hand corner of plot.

	---
	Output : 

	ax : plt.axis object
		Returns axis object to further manipulate before plotting or saving.

	'''

	# Finding percentiles (splitting at 33, 66)
	upper = np.percentile(tab.halo_mass_stel.values, 66)
	lower = np.percentile(tab.halo_mass_stel.values, 33)

	# sub-selecting percentile tabs.
	upper_tab, upper_control_tab, _, _, _ = return_tabs(tab[(tab.halo_mass_stel.values > upper) & (tab.halo_mass_stel.values < 14.5)], min_mass=0, max_mass=1, mass_col='dtfe_gauss_3Mpc')
	middle_tab, middle_control_tab, _, _, _ = return_tabs(tab[(tab.halo_mass_stel.values <= upper) & (tab.halo_mass_stel.values >= lower)], min_mass=0, max_mass=1, mass_col='dtfe_gauss_3Mpc')
	lower_tab, lower_control_tab, _, _, _ = return_tabs(tab[tab.halo_mass_stel.values < lower], min_mass=0, max_mass=1, mass_col='dtfe_gauss_3Mpc')

	# setting up figure.
	fig, ax = plt.subplots(1,3, figsize=(13, 3.5), sharex='all', sharey='all')
	bins = np.linspace(0, 0.5, 35)
	knf.histerr(lower_tab.dtfe_gauss_3Mpc.values, ax[0], bins=bins, color=fil_color, median=True, label='$\mathrm{D_{skel} < 1.5Mpc; D_{node} > 2Mpc}$')
	knf.histerr_fill(lower_tab.dtfe_gauss_3Mpc.values, ax[0], bins=bins, color=fil_color, median=False, label=None)
	knf.histerr(lower_control_tab.dtfe_gauss_3Mpc.values, ax[0], bins=bins, color=con_color, median=True, label='Control sample')
	knf.histerr_fill(lower_control_tab.dtfe_gauss_3Mpc.values, ax[0], bins=bins, color=con_color, median=False, label=None)
	
	knf.histerr(middle_tab.dtfe_gauss_3Mpc.values, ax[1], bins=bins, color=fil_color, median=True)
	knf.histerr_fill(middle_tab.dtfe_gauss_3Mpc.values, ax[1], bins=bins, color=fil_color, median=False)
	knf.histerr(middle_control_tab.dtfe_gauss_3Mpc.values, ax[1], bins=bins, color=con_color, median=True)
	knf.histerr_fill(middle_control_tab.dtfe_gauss_3Mpc.values, ax[1], bins=bins, color=con_color, median=False)
	
	knf.histerr(upper_tab.dtfe_gauss_3Mpc.values, ax[2], bins=bins, color=fil_color, median=True)
	knf.histerr_fill(upper_tab.dtfe_gauss_3Mpc.values, ax[2], bins=bins, color=fil_color, median=False)
	knf.histerr(upper_control_tab.dtfe_gauss_3Mpc.values, ax[2], bins=bins, color=con_color, median=True)
	knf.histerr_fill(upper_control_tab.dtfe_gauss_3Mpc.values, ax[2], bins=bins, color=con_color, median=False)
	
	# adding KS-tests in each mini-panel.
	ks, p = stats.ks_2samp(lower_tab.halo_mass_stel.values, lower_control_tab.halo_mass_stel.values)
	ax[0].annotate('$\mathrm{KS}=$'+str(round(ks, 3))+'\n $p=$'+str(np.round(p, 3))+'\n n='+str(lower_tab.shape[0]), 
					xy=(0.3, 0.23), xycoords='data', fontsize=11)
	
	ks, p = stats.ks_2samp(middle_tab.halo_mass_stel.values, middle_control_tab.halo_mass_stel.values)
	ax[1].annotate('$\mathrm{KS}=$'+str(round(ks, 3))+'\n $p=$'+str(np.round(p, 3))+'\n n='+str(middle_tab.shape[0]), 
					xy=(0.3, 0.23), xycoords='data', fontsize=11)

	ks, p = stats.ks_2samp(upper_tab.halo_mass_stel.values, upper_control_tab.halo_mass_stel.values)
	ax[2].annotate('$\mathrm{KS}=$'+str(round(ks, 3))+'\n $p=$'+str(np.round(p, 3))+'\n n='+str(upper_tab.shape[0]),
					xy=(0.3, 0.23), xycoords='data', fontsize=11) 
	
	ax[0].set_ylabel('PDF', fontsize=13)  

	ax[0].set_xlabel(r'$\mathrm{\rho_{3Mpc}}$', fontsize=13)
	ax[1].set_xlabel(r'$\mathrm{\rho_{3Mpc}}$', fontsize=13)
	ax[2].set_xlabel(r'$\mathrm{\rho_{3Mpc}}$', fontsize=13)
	fig.subplots_adjust(wspace=0, hspace=0.1)
	fig.suptitle(label, fontsize=14)
	
	return ax

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
		upper = np.percentile(tab.halo_mass_stel.values, 66)
		lower = np.percentile(tab.halo_mass_stel.values, 33)

	# sub-selecting percentile tabs.
	upper_tab = tab[(tab.halo_mass_stel.values > upper) & (tab.halo_mass_stel.values < 14.5)]
	middle_tab = tab[(tab.halo_mass_stel.values <= upper) & (tab.halo_mass_stel.values >= lower)]
	lower_tab = tab[tab.halo_mass_stel.values < lower]

	# plotting filament samples.
	ax.plot(lower_tab.halo_mass_stel.values, lower_tab.lambda_re.values, color=color, marker='v', linestyle='None', alpha=0.3, markersize=2)
	ax.plot(middle_tab.halo_mass_stel.values, middle_tab.lambda_re.values, color=color, marker='v', linestyle='None', alpha=0.3, markersize=2)
	ax.plot(upper_tab.halo_mass_stel.values, upper_tab.lambda_re.values, color=color, marker='v', linestyle='None', alpha=0.3, markersize=2)

	# overplotting the average values in each bin.
	ax.errorbar([np.median(lower_tab.halo_mass_stel.values), np.median(middle_tab.halo_mass_stel.values), np.median(upper_tab.halo_mass_stel.values)],
				[np.median(lower_tab.lambda_re.values), np.median(middle_tab.lambda_re.values), np.median(upper_tab.lambda_re.values)], 
				yerr=[stats.sem(lower_tab.lambda_re.values), stats.sem(middle_tab.lambda_re.values), stats.sem(upper_tab.lambda_re.values)], 
				 color=color, marker='H', markersize=5, capsize=5, label=label)

	# adding percentile vertical lines to denote boundaries.
	ax.axvline(upper, color=color, alpha=0.3, linestyle='dashed')
	ax.axvline(lower, color=color, alpha=0.3, linestyle='dashed')

	# formatting final plot.
	#ax.set_xscale('log')
	ax.set_ylabel(r'$\mathrm{\lambda_{R}(< 1.5R_{e})}$', fontsize=16)
	ax.set_xlabel(r'$\mathrm{M_{halo}}$', fontsize=16)
	ax.set_xlim((np.min(tab.halo_mass_stel.values), 14))
	ax.set_ylim((0,1))
	return ax



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

