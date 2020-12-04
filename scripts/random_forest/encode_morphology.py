'''
encode_morphology - functions to encode morphology for input into the random forest
'''


import pandas as pd
import numpy as np
import catalog_init
import seaborn as sns
import matplotlib.pyplot as plt


def one_hot(df, split_ltgs = True, category_plot = False, vote_frac=0.7):
	'''
	Returns one-hot encoded morphology definitions (added as columns to pd.df as input) 
	based on galaxyZoo2 classifications.
	
	----------
	Parameters
	
	df : pd.DataFrame
		Pandas dataframe object containing galaxies with galaxyZoo2 information (i.e. 
		debiased vote fractions).
	
	split_ltgs : bool
		Flag to split ltgs into individual classifications (i.e. S0-Sa and Sb-Sd) or leave 
		as combined classification.
		
	category_plot : bool
		Return countplot of galaxies in each morphological class.
	
	vote_frac : float
		Debiased vote fraction to split classifications on. (e.g. 0.7 = 70% of votes 
		put this galaxy in this morph class).
		
	-------
	Returns
	
	df : pd.DataFrame
		Original pandas dataframe object with addition one-hot encoded morphological 
		classifications added.
	'''
	
	# testing to see if has been matched to galaxyzoo catalogue.
	assert 't01_smooth_or_features_a01_smooth_debiased' in df.columns, "Have you matched to galaxyZoo?" 
	
	# defining empty new column with str morphologies.
	morph = np.full(df.shape[0], 'unclassified')
	morph[catalog_init.select_morphology(df, morphology='etg', 
										 vote_frac = vote_frac).index] = 'etg'

	if split_ltgs == True :
		morph[catalog_init.select_morphology(df, morphology='s0sa', 
											 vote_frac = vote_frac).index] = 's0sa'
		morph[catalog_init.select_morphology(df, morphology='sbsd', 
											 vote_frac = vote_frac).index] = 'sbsd'
	else : 
		morph[catalog_init.select_morphology(df, morphology='ltg', 
											 vote_frac = vote_frac).index] = 'ltg'

	# plotting counts of each morphological category
	if category_plot == True:
		sns.countplot(morph);
		plt.show()

	# one-hot encoding morphology classifications.
	one_hot_morph = pd.get_dummies(morph)

	for col in np.unique(morph):
		print('{:15} encoded class column added.'.format(col))
	
	return pd.concat([df, one_hot_morph], axis=1)


def add_Ttype_col(df):
	'''
	Adds empirical formula for morphological T-type estimated using galaxyZoo2 vote 
	fractions.
	
	----------
	Parameters
	
	df : pd.DataFrame
		Pandas dataframe object containing galaxies with galaxyZoo2 information (i.e. 
		debiased vote fractions).

	-------
	Returns
	
	df : pd.DataFrame
		Original pandas dataframe with additional Ttype column.
	
	'''
	
	# testing to see if has been matched to galaxyzoo catalogue.
	assert 't01_smooth_or_features_a01_smooth_debiased' in df.columns, "Have you matched to galaxyZoo?"
	
	Ttype = (4.63 + 4.17 * df.t05_bulge_prominence_a10_no_bulge_debiased - 2.27 
			 * df.t05_bulge_prominence_a12_obvious_debiased - 8.38 
			 * df.t05_bulge_prominence_a13_dominant_debiased )
	
	df['Ttype'] = Ttype
	
	return df

