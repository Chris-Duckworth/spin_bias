# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------------------
# kin_morph_funcs
#
# This script provides functions to select main samples from MPL-8 and define PA defined 
# properties and plots.
#
# Chris J Duckworth cd201@st-andrews.ac.uk
# ---------------------------------------------------------------------------------------

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
# from mangadap.util.bitmask import BitMask
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import Planck15

# ---------------------------------------------------------------------------------------
# func: mpl8_main_sample
#
# This function returns a dataframe containing the main MPL-8 sample (Primary, Secondary and
# Colour-Enhanced) with all repeat observations removed.
# 
# Input: location of file as a str. File must be csv.

def mpl8_main_sample(file):
    mpl8 = pd.read_csv(file, comment='#')
    tab = mpl8[(((mpl8.mngtarg1 != 0) | (mpl8.mngtarg3 != 0)) & ((mpl8.mngtarg3 & int(2)**int(19)+int(2)**int(20)) == 0) )]
    sdssMaskbits = os.path.join(os.environ['IDLUTILS_DIR'], 'data', 'sdss', 'sdssMaskbits.par')
    bm = BitMask.from_par_file(sdssMaskbits, 'MANGA_TARGET1')
    main_sample = bm.flagged(tab.mngtarg1, flag=['PRIMARY_v1_2_0','SECONDARY_v1_2_0','COLOR_ENHANCED_v1_2_0'])
    # Removing ancillary progs and repeat obs.
    mpl8_main = tab[main_sample]
    mpl8_main = mpl8_main.iloc[np.unique(mpl8_main.mangaid.values, return_index=True)[1]]
    return mpl8_main

# ---------------------------------------------------------------------------------------
# func: mpl8_pa_sample 
#
# This function returns a pa defined sample when given the output from mpl8_main_sample.
# This checks there are no repeats in-case main sample comes from another source.

def mpl8_pa_sample(mpl8_main):
    mpl8_pa = mpl8_main[(mpl8_main.stel_feature == 0) & (mpl8_main.halpha_feature == 0) & ((mpl8_main.stel_qual == 1) | (mpl8_main.stel_qual == 2)) & ((mpl8_main.halpha_qual == 1) | (mpl8_main.halpha_qual == 2))]
    mpl8_pa = mpl8_pa.iloc[np.unique(mpl8_pa.mangaid.values, return_index=True)[1]]
    return mpl8_pa

# ---------------------------------------------------------------------------------------
# func: tng100_pa_sample
#
# For the matched manga-like TNG100 sample, this returns the pa defined. Since our TNG100
# sample is already only matched to main sample manga galaxies we dont have to do the pre-
# selection.

def tng100_pa_sample(tng100_main):
    tng100_pa = tng100_main[(tng100_main.stel_feature == 0) & (tng100_main.halpha_feature == 0) & ((tng100_main.stel_qual == 1) | (tng100_main.stel_qual == 2)) & ((tng100_main.halpha_qual == 1) | (tng100_main.halpha_qual == 2))]
    return tng100_pa

# ---------------------------------------------------------------------------------------
# func: SFMS_breakdown
#
# This function splits into individual dataframes based on the flags in annalisa's flags.

def SFMS_breakdown(tab):
    QU = tab[tab.sfms_flag == 0]
    SF = tab[tab.sfms_flag == 1]
    GV = tab[tab.sfms_flag == -1]
    return QU, SF, GV

# ---------------------------------------------------------------------------------------
# func: GZ_match
#
# This function loads in the galaxyZoo tabledata (VAC for MaNGA) and matches to a given
# dataframe on MANGAID. 

def GZ_match(GZfile, match):
    gz_info = pd.read_csv(GZfile)
    return pd.merge(gz_info, match, left_on='MANGAID', right_on='mangaid')

# ---------------------------------------------------------------------------------------
# func: morph_breakdown
#
# This function splits an MPL8 dataframe (matched to GZ) into 3 separate dataframes:
# ETG, S0/Sa and Sb/Sd. LTGs are split based on the linear T-type mapping (at 3) in the
# GZ2 paper. Use with care but should be fine for define 'lateness' of LTGs.

def morph_breakdown(main, vote_frac = 0.7, Tsplit = 3):
    ETGs = main[main.t01_smooth_or_features_a01_smooth_debiased >= vote_frac]
    LTGs = main[main.t01_smooth_or_features_a02_features_or_disk_debiased >= vote_frac]
    # defining T-type. Based entirely on bulge prominence.
    Ttype = 4.63 + 4.17 * LTGs.t05_bulge_prominence_a10_no_bulge_debiased - 2.27 * LTGs.t05_bulge_prominence_a12_obvious_debiased - 8.38 * LTGs.t05_bulge_prominence_a13_dominant_debiased
    # dividing on T-type.
    S0_Sa = LTGs[Ttype <= Tsplit]
    Sb_Sd = LTGs[Ttype > Tsplit]
    return ETGs, S0_Sa, Sb_Sd

# ---------------------------------------------------------------------------------------
# func: histerr 
#
# This function creates a normalised histogram with Poisson noise on each bin. Can deal 
# with log-spaced x axis.

def histerr(data, ax, bins=10, label='all', color='k', linestyle='solid', linewidth=2, wgt=1, cumulative=False, median=False):
    if median == True:
        ax.axvline(np.median(data), linestyle=linestyle, color=color, linewidth=4, alpha=0.6)
    entries, edges, _ = ax.hist(data, weights= wgt * np.ones_like(data)/float(len(data)), histtype='step', linestyle=linestyle, label=label, color=color, bins=bins, cumulative=cumulative, linewidth=linewidth)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    ax.errorbar(bin_centers, entries, yerr=np.sqrt(entries*float(len(data)))/(float(len(data))), fmt='s', markersize=1.5, linewidth=1 , color=color, alpha=0.3)
    return edges

# ---------------------------------------------------------------------------------------
# func: histerr_fill
#
# This function creates a normalised histogram with Poisson noise on each bin. Can deal 
# with log-spaced x axis. Variation which fills histogram.

def histerr_fill(data, ax, bins=10, label='all', color='k', alpha=0.3, linestyle='solid', linewidth=2, wgt=1, cumulative=False, median=False):
    if median == True:
        ax.axvline(np.median(data), linestyle=linestyle, color=color, linewidth=4, alpha=0.7)
    entries, edges, _ = ax.hist(data, weights= wgt * np.ones_like(data)/float(len(data)), alpha=alpha, label=label, color=color, bins=bins, cumulative=cumulative, linewidth=linewidth)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    ax.errorbar(bin_centers, entries, yerr=np.sqrt(entries*float(len(data)))/(float(len(data))), fmt='s', markersize=1.5, linewidth=1 , color=color, alpha=1)
    return edges

# ---------------------------------------------------------------------------------------
# func: xtick_format
#
# This function sorts out major and minor tick spacing for nice plots.
# 
# Input: Major and minor tick spacing: (float), ax, format of tick labels.

def xtick_format(major, minor, ax, format='%1.0f'):
    majorLocator = MultipleLocator(major)
    majorFormatter = FormatStrFormatter(format)
    minorLocator = MultipleLocator(minor)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_locator(minorLocator)
    return 
    
# ---------------------------------------------------------------------------------------
# func: ytick_format
#
# This function sorts out major and minor tick spacing for nice plots.
# 
# Input: Major and minor tick spacing. (float).

def ytick_format(major, minor, ax, format='%1.0f'):
    majorLocator = MultipleLocator(major)
    majorFormatter = FormatStrFormatter(format)
    minorLocator = MultipleLocator(minor)
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_minor_locator(minorLocator)
    return
    
# ---------------------------------------------------------------------------------------
# func: lim_cen_dist
#
# For the Lim+17 group catalogue this function finds the distance to the group centre from 
# the input galaxy.
# Input a dataframe and this exports the updated dataframe with the additional column. 
# Assumes Planck15 cosmology.

def lim_cen_dist(tab):
    group_ra = tab.group_ra.values * u.deg
    group_dec = tab.group_dec.values * u.deg
    group_z = tab.group_z.values
    group_dist = Planck15.comoving_distance(group_z)
    group_pos = SkyCoord(ra=group_ra, dec=group_dec, distance=group_dist)

    gal_ra = tab.ra.values * u.deg
    gal_dec = tab.dec.values * u.deg
    gal_z = tab.z_CMB.values 
    gal_dist = Planck15.comoving_distance(gal_z)
    gal_pos = SkyCoord(ra=gal_ra, dec=gal_dec, distance=gal_dist)

    dist_cen = np.zeros(tab.shape[0])
    # Finding for each galaxy.
    for i, c1 in enumerate(gal_pos):
        dist_cen[i] = c1.separation_3d(group_pos[i]).value
    tab['dist_cen'] = dist_cen
    return tab

# ---------------------------------------------------------------------------------------
# func: pipe_match
#
# This function loads in the pipe3d tabledata (VAC for MaNGA) and matches to a given
# dataframe on PLATEIFU. PIPE3D file has been changed to csv for this to work.

def pipe_match(pipe_file, match):
    pipe_info = pd.read_csv(pipe_file)
    return pd.merge(pipe_info.drop(['mangaid'], axis=1), match, left_on='plateifu', right_on='plateifu')

# ---------------------------------------------------------------------------------------
# func: subpop_plot
#
# Plots for 3 populations for standard PA split and no gas rot. 

def subpop_plot(tab_prop, tab_ngr_prop, tab, tab_ngr, bins, colors, ax):
    histerr(tab_prop[tab.pa_offset.values < 30], ax, bins=bins, linestyle='solid', color=colors[0], label=r'$\Delta$PA < 30$^{\circ}$', median=True)
    histerr(tab_prop[tab.pa_offset.values >= 30], ax, bins=bins, linestyle='solid', color=colors[1], label=r'$\Delta$PA $\geq$ 30$^{\circ}$', median=True)
    histerr(tab_ngr_prop, ax, bins=bins, linestyle='-.', color=colors[2], label=r'No gas rot', median=True) 
    return 

# ---------------------------------------------------------------------------------------
# func: pipe3d_megaplot
#
# This function plots a grid of V/sigma, gas mass, SFR (Ha & ssp), Metallicity in Re and 
# gradient, MW Age in Re and grad. 2 x 4 grid.

def pipe3d_megaplot(tab, tab_ngr, colors):
    fig, ax = plt.subplots(2,4, figsize=(16,7))
    
    # V/sigma
    bins = np.linspace(0,0.8,15)
    subpop_plot(tab.vel_sigma_re.values, tab_ngr.vel_sigma_re.values, tab, tab_ngr, bins, colors, ax[0,0])
    ytick_format(0.1, 0.01, ax[0,0], format='%1.2f')
    ax[0,0].legend(frameon=False, fontsize=10, loc='upper right')
    ax[0,0].set_xlabel(r'V/$\sigma$')
    ax[0,0].set_ylabel('PDF', fontsize=14)

    # Gas Mass
    bins = np.linspace(6,11,15)
    subpop_plot(tab.log_mass_gas.values, tab_ngr.log_mass_gas.values, tab, tab_ngr, bins, colors, ax[1,0])
    ytick_format(0.1, 0.01, ax[1,0], format='%1.2f')
    ax[1,0].set_xlabel(r'log(M$_{gas}$ / M$_{\odot}$)')
    ax[1,0].set_ylabel('PDF', fontsize=14)
    ax[1,0].set_xlim([7,11])

    # SFR ssp
    bins = np.linspace(-2.5, 1.5, 15)
    subpop_plot(tab.log_sfr_ssp.values, tab_ngr.log_sfr_ssp.values, tab, tab_ngr, bins, colors, ax[0,1])
    ytick_format(0.1, 0.01, ax[0,1], format='%1.2f')
    ax[0,1].set_xlabel(r'SFR (ssp)')
    #ax[0,1].set_ylabel('PDF', fontsize=14)

    # SFR Ha    
    bins = np.linspace(-4, 2, 15)
    subpop_plot(tab.log_sfr_ha.values, tab_ngr.log_sfr_ha.values, tab, tab_ngr, bins, colors, ax[1,1])
    ytick_format(0.1, 0.01, ax[1,1], format='%1.2f')
    ax[1,1].set_xlabel(r'SFR (H$\alpha$ flux)')
    #ax[1,1].set_ylabel('PDF', fontsize=14)

    # Mass weighted stellar metallicity within 1 Re.
    bins = np.linspace(-0.6,0.6,15)
    subpop_plot(tab.zh_mw_re_fit.values, tab_ngr.zh_mw_re_fit.values, tab, tab_ngr, bins, colors, ax[0,2])
    ytick_format(0.1, 0.01, ax[0,2], format='%1.2f')
    ax[0,2].set_xlabel(r'Mass weighted stellar metallicity (< 1 R$_e$)')
    ax[0,2].set_ylabel('PDF', fontsize=14)

    # Mass weighted stellar metallicity gradient.
    bins = np.linspace(-0.5,0.5,15)
    subpop_plot(tab.alpha_zh_mw_re_fit.values, tab_ngr.alpha_zh_mw_re_fit.values, tab, tab_ngr, bins, colors, ax[1,2])
    ytick_format(0.1, 0.01, ax[1,2], format='%1.2f')
    ax[1,2].set_xlabel(r'Mass weighted stellar metallicity gradient')
    #ax[1,2].set_ylabel('PDF', fontsize=14)

    # MW age at 1 Re
    bins = np.linspace(8.8,10.3,15)
    subpop_plot(tab.age_mw_re_fit.values, tab_ngr.age_mw_re_fit.values, tab, tab_ngr, bins, colors, ax[0,3])
    ytick_format(0.1, 0.01, ax[0,3], format='%1.2f')
    ax[0,3].set_xlabel(r'Mass weighted age, (log(Yr), < 1 R$_e$)')
    #ax[0,3].set_ylabel('PDF', fontsize=14)

    # MW age gradient
    bins = np.linspace(-0.6,0.4,15)
    subpop_plot(tab.alpha_age_mw_re_fit.values, tab_ngr.alpha_age_mw_re_fit.values, tab, tab_ngr, bins, colors, ax[1,3])
    ytick_format(0.1, 0.01, ax[1,3], format='%1.2f')
    ax[1,3].set_xlabel(r'Mass weighted age gradient')
    #ax[1,3].set_ylabel('PDF', fontsize=14)
    
    return