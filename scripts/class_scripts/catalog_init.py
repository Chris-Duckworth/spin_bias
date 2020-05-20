''' catalog_init - contains a series of functions for building catalog file 
(i.e. matching various input catalogs.)'''

import pandas as pd 
import numpy as np 
from astropy.io import fits
from astropy.cosmology import Planck15
from scipy import interpolate
import matplotlib.pyplot as plt


def load_basic(basepath):
	''' 
	Function which returns a pandas dataframe object containing the basic MaNGA info
	required for this analysis.
	'''
	
	# loading in reference catalog to define mpl8 sample.
	mpl8 = pd.read_csv(basepath + 'mpl8_main_TNG_ref.csv')
	
	# loading pipe3d information.
	pipe3d = pd.read_csv(basepath + 'manga.Pipe3D_v2_5_3.csv')
	# dropping mangaid from pipe3d because its wrong.
	pipe3d = pipe3d.drop(columns=["mangaid"]) 
	
	# loading in galaxyZoo information.
	gz = pd.read_csv(basepath + 'MaNGA_gz-v1_0_1.csv')
	
	# matching mpl8 to GZ
	mpl8_gz = mpl8.merge(gz, left_on='mangaid', right_on='MANGAID')
	
	# matching on unique observation identifier 'plateifu' for pipe3d
	mpl8_gz_pipe3d = mpl8_gz.merge(pipe3d, left_on='plateifu', right_on='plateifu')
	
	return mpl8_gz_pipe3d


def centrals_only(tab):
	'''
	This removes all satellites from the sample. Requires a group catalog, here matched by
	match_to_cw, so make sure this has been run first.
	'''
	# selecting only centrals.
	tab = tab[(tab.massive_flag.values == 1) & (tab.f_edge.values > 0.6) & (tab.halo_mass_stel.values > 0)]

	# adding stellar to halo mass ratio. 
	tab['stellar_to_halo_ratio'] = np.log10(tab.nsa_elpetro_mass.values) -  tab.halo_mass_stel.values 
	
	return tab


def match_to_cw(basepath, tab, return_plot=False):
	'''
	Assuming load_basic has been called and an initial tab has been loaded, this takes 
	that tabledata and matches to cw info. Normalisation is then computed and distance to 
	nearest CW feature is found. Option to plot the normalisation of distances.
	'''
	
	# loading in cw catalog which has already been matched to group info.
	cw = pd.read_csv(basepath + 'CW_mpl6_yang_s5.csv')
	mid = pd.read_csv(basepath+'MaNGA_IDs')
	
	# merging catalogues and finding manga targets only.
	cw_manga_targets = mid.merge(cw, left_on='ID', right_on='ID')
	
	# matching to input tab based on mangaid.
	cw_manga_targets_tab = cw_manga_targets.merge(tab, left_on='MANGAID', right_on="mangaid")
	
	# computing distances to CW elements and finding normalisation.
	# SDSS DR10 (7966 sq deg) with the redshift limit of z=0.2.
	DR10 = fits.open(basepath + 'CW_mpl6_yang_s5')[1].data
	z_dr10 = DR10['zobs'][(DR10['zobs'] <= 0.2) & (DR10['zobs'] >= 0)]

	# Finding redshift range of MaNGA targets only! 
	mngtarg_min = min(DR10['zobs'][DR10['ID'] != - 9999])
	mngtarg_max = max(DR10['zobs'][DR10['ID'] != - 9999])
	
	# Number of galaxies per redshift bin.
	N_z = np.histogram(z_dr10,bins=np.linspace(mngtarg_min,0.2,301))
	N_z_manga = np.histogram(DR10['zobs'][DR10['ID'] != - 9999],bins=N_z[1])

	# Finding the fraction of the sky covered by DR10.
	whole_sky_area = 4*np.pi*(180/np.pi)**2 # steradians converted to sq degs.
	frac = 7966 / whole_sky_area
	# So e.g. volume up to redshift of 0.1
	total_vols = frac*Planck15.comoving_volume(N_z[1])
	slice_vols = total_vols[1:] - total_vols[:-1]

	# So for each redshift slice n = N/V:
	number_density = N_z[0] / slice_vols
	number_density_manga = N_z_manga[0] / slice_vols

	# Now finding the euclidean inter galaxy separation as a function of redshift.
	D_z = number_density**(-1/3)

	# Finding bin centres
	bin_cen = (N_z[1][:-1] + N_z[1][1:])/2
	D_z_interpolate = interpolate.interp1d(bin_cen[:-1],D_z[:-1], kind='cubic')
	xnew = np.linspace(mngtarg_min,mngtarg_max, num=301, endpoint=True)

	# plotting if verbose.
	if return_plot == True:
		fig, ax = plt.subplots(1,2, figsize=(15,5))
		ax[0].set_title('Number density, n(z)')
		ax[0].plot(bin_cen,number_density.value,label='DR10')
		ax[0].plot(bin_cen,number_density_manga.value,label='MaNGA targets')
		ax[0].set_yscale('log')
		ax[0].legend()
		ax[0].set_xlabel('redshift, z')
		ax[0].set_ylabel(r'$n(z)$')
		ax[0].grid()

		ax[1].plot(bin_cen[:-1],D_z[:-1],label='Euclidean')
		ax[1].set_ylabel(r' Mean intergalaxy separation, $D_z$ (Mpc)')
		ax[1].set_xlabel(r'redshift, z')
		ax[1].grid()

		def get_axis_limits(ax, scale=.9):
			return ax.get_xlim()[1]*scale*0.75, 0.1
		ax[1].annotate('Median = '+str(round(np.median(D_z[:-1]).value,3))+' Mpc\nMean    = '+str(round(np.mean(D_z[:-1]).value,3))+' Mpc', xy=get_axis_limits(ax[1]))
		plt.show()

	# adding normalised distance columns
	cw_manga_targets_tab['log_dskel_norm'] = cw_manga_targets_tab.log_dskel.values - np.log10(D_z_interpolate(cw_manga_targets_tab.z.values))
	cw_manga_targets_tab['log_dwall_norm'] = cw_manga_targets_tab.log_dwall.values - np.log10(D_z_interpolate(cw_manga_targets_tab.z.values))
	cw_manga_targets_tab['log_dnode_norm'] = cw_manga_targets_tab.log_dnode.values - np.log10(D_z_interpolate(cw_manga_targets_tab.z.values))
	
	return cw_manga_targets_tab


def select_morphology(tab, morphology, vote_frac = 0.7, Tsplit = 3):
	'''
	Assuming load_basic has been called (and output is provided as input here). 
	Given a selected morphology (etg, ltg, s0sa, sbsd) this returns a table only 
	containing that type.  
	'''
	assert morphology in ['etg', 'ltg', 's0sa', 'sbsd'], 'not a valid morphology selection!'
	
	ETGs = tab[tab.t01_smooth_or_features_a01_smooth_debiased >= vote_frac]
	LTGs = tab[tab.t01_smooth_or_features_a02_features_or_disk_debiased >= vote_frac]
	# defining T-type. Based entirely on bulge prominence.
	Ttype = 4.63 + 4.17 * LTGs.t05_bulge_prominence_a10_no_bulge_debiased - 2.27 * LTGs.t05_bulge_prominence_a12_obvious_debiased - 8.38 * LTGs.t05_bulge_prominence_a13_dominant_debiased
	
	# Using GZ definitions to sub-divide.
	if morphology == 'etg':
		return ETGs
	
	elif morphology == 'ltg':
		return LTGs
	
	elif morphology == 's0sa':
		return LTGs[Ttype <= Tsplit]
	
	elif morphology == 'sbsd':
		return LTGs[Ttype > Tsplit]

