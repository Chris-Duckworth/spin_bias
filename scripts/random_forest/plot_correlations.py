'''
plot_correlations - correlation and random forest importance ranking plotting functions
'''

import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


def correlation_corner(df, cols, colors, names=None, coef='spearman', limits=None, 
					   figsize=(15, 15)):
	'''
	correlation_corner - summary corner plot between defined columns in a dataframe. Each
	subplot in the corner has a value for a correlation coefficient and a linear fit line.
	
	----------
	Parameters
	
	df : pd.DataFrame object
	
	cols : list/array of str 
		Column names of interest (to include in the corner plot)
		
	colors : list/array of str
		Colors for each defined column in cols.
		
	names : list/array of str 
		(Optional) names of column on plot. Defaults to string column names.
	
	coef : str
		Choice of correlation coefficient (pearson/ spearman). 
	
	limits : np.array(2, n)
		Plotting limits of each column in the dataframe.
	
	figsize : (float, float)
	
	-------
	Returns
	
	Matplotlib figure and axis objects.
	'''
	fig, axes = plt.subplots(len(cols), len(cols), figsize=figsize, 
							 sharex='col', sharey='row')
	
	# plotting rows.
	for i in np.arange(len(cols)):
	
		# removing duplicate axes.
		axes[i, i].set_visible(False)
	
		for j in np.arange(len(cols)):
		
			# scatter plot with fitted line for each combo of cols.
			sns.regplot(x=cols[i], y=cols[j], data=df, 
						ax=axes[j, i], color=colors[j], 
						scatter_kws={'alpha':0.3, 'edgecolor':colors[i]})
			
			# adding limits to subplots.
			if limits != None:
				axes[j, i].set_xlim(limits[i])
				axes[j, i].set_ylim(limits[j])
		
			# annotating with corr. coef.
			if coef == 'spearman' :
				text = ('spearmanr=' + str(np.round(stats.spearmanr(df[cols[i]], 
						df[cols[j]])[0],3)) +'; p= {:.1e}'.format(stats.spearmanr(df[cols[i]], df[cols[j]])[1]) )
			elif coef == 'pearson' :
				text = ('pearson=' + str(np.round(stats.pearsonr(df[cols[i]], 
						df[cols[j]])[0],3)) +'; p= {:.1e}'.format(stats.pearsonr(df[cols[i]], df[cols[j]])[1]) )
			else :
				text = ''
		
			axes[j, i].annotate(text, xy=(0.1, 0.9), 
								xycoords="axes fraction", fontsize=13)
	
		# clearing seaborn labels.
		for ax in axes[:, i] :
			ax.set_xlabel('')
			ax.set_ylabel('')
		
		# if plotting names empty, using column names in dataframe
		if names == None :
			names = cols
		
		# row labels.
		axes[-1, i].set_xlabel(names[i], fontsize=15)
		# column labels.
		axes[i, 0].set_ylabel(names[i], fontsize=15)
	
	# removing duplicates.
	for i, j in zip(*np.triu_indices_from(axes, 1)):
		axes[i, j].set_visible(False)

	fig.subplots_adjust(wspace=0.05, hspace=0.05)
	return fig, axes


def importance_ranking(feature_labels, importances, sorted=True, figsize=(7, 5) ):
	'''
	importance ranked bar plot for random forests.
	
	----------
	Parameters
	
	feature_labels : np.array / list
		String names of features.
	
	importances : np.array / list
		Numeric importances of each feature in same order.
	
	sorted : bool
		Whether you want them sorted into increase importance or not.
	
	figsize : (float, float)
	
	-------
	Returns
	
	Matplotlib figure and axes objects
	'''
	
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot()
	
	x_values = np.arange(len(importances))
	
	if sorted == True:
		ax.bar(x_values, np.sort(importances), orientation = 'vertical')
		# Tick labels for x axis
		plt.xticks(x_values, feature_labels[np.argsort(importances)], 
				   rotation=60, fontsize=14)
	
	else :
		ax.bar(x_values, importances, orientation = 'vertical')
		# Tick labels for x axis
		plt.xticks(x_values, feature_labels, rotation=60, fontsize=14)

	ax.set_ylabel('Importance', fontsize=14)
	ax.set_xlabel('Feature', fontsize=14)	
	return fig, ax 


def normed_step_poisson_histogram(data, bins=None, color='k', weights=None, median=False, ax=None, **kwargs):
    '''
    Normalised step histogram with poisson errorbars on each defined bin.
    Optional weighting for each data-point.
    Additional plotting arguments for plt.errorbar()
    
    ----------
    Parameters
    
    data : np.array() or list
        Data to histogram
    
    bins : np.array() or list
        (Optional) Defined bins to histogram data
    
    color : matplotlib defined color
        (Optional) Provide a color for the histogram, 
        errorbars and median line.
    
    weights : np.array() or list
        (Optional) Weights of same size as data
    
    median : bool
        (Optional) Boolean on whether to include median line 
        of data on histogram plot
    
    ax : matplotlib.pyplot.axis() object
        (Optional) Matplotlib axis object to add histogram to
    
    **kwargs : 
        Additional arguments passed to plt.hist()
    
    -------
    Returns
    
    ax : matplotlib.pyplot.axis() object
    '''
    # if axes undefined, creating a new plot.
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    
    # if weights not supplied, assuming uniform weighting.
    if weights is None:
        weights = 1
    
    # if bins aren't supplied
    if bins is None:
        bins = np.linspace(np.min(data), np.max(data), 20)

    # histogram with weighted normalised bins.
    y, binEdges, _ = ax.hist(data, bins=bins, weights= weights * np.ones_like(data)/float(len(data)), 
                             color=color, **kwargs)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    ax.errorbar(bincenters, y, yerr=np.sqrt(y*float(len(data))) / (float(len(data))), 
                color=color, fmt='s', markersize=1.5, linewidth=1, alpha=0.5)
    
    # adding median line for distribution if required.
    if median == True:
        ax.axvspan(np.median(data) - stats.sem(data), np.median(data) + stats.sem(data), 
                   ymin=0.8, color=color, alpha=0.3)
        ax.axvline(np.median(data), ymin=0.8, color=color, linewidth=3, alpha=0.6)

    return ax


def normed_line_poisson_histogram(data, bins=None, color='k', weights=None, median=False, ax=None, **kwargs):
    '''
    Normalised line histogram with poisson errorbars on each defined bin.
    Optional weighting for each data-point.
    Additional plotting arguments for plt.errorbar()
    
    ----------
    Parameters
    
    data : np.array() or list
        Data to histogram
    
    bins : np.array() or list
        (Optional) Defined bins to histogram data
    
    color : matplotlib defined color
        (Optional) Provide a color for the histogram, 
        errorbars and median line.
    
    weights : np.array() or list
        (Optional) Weights of same size as data
    
    median : bool
        (Optional) Boolean on whether to include median line 
        of data on histogram plot
    
    ax : matplotlib.pyplot.axis() object
        (Optional) Matplotlib axis object to add histogram to
    
    **kwargs : 
        Additional arguments passed to plt.errorbar()
    
    -------
    Returns
    
    ax : matplotlib.pyplot.axis() object
    '''
    # if axes undefined, creating a new plot.
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    
    # if weights not supplied, assuming uniform weighting.
    if weights is None:
        weights = 1
    
    # if bins aren't supplied
    if bins is None:
        bins = np.linspace(np.min(data), np.max(data), 20)
    
    # histogram with weighted normalised bins.
    y, binEdges = np.histogram(data, bins, weights= weights * np.ones_like(data)/float(len(data)))
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    ax.errorbar(bincenters, y, yerr=np.sqrt(y*float(len(data))) / (float(len(data))), 
                color=color, **kwargs)
    
    # adding median line for distribution if required.
    if median == True:
        ax.axvspan(np.median(data) - stats.sem(data), np.median(data) + stats.sem(data), 
                   ymin=0.8, color=color, alpha=0.3)
        ax.axvline(np.median(data), ymin=0.8, color=color, linewidth=3, alpha=0.6)

    return ax



