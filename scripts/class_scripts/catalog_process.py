''' catalog_process - contains a series of functions which process all of the data in the
catalog class
'''

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
from scipy.interpolate import interp1d

def compute_expected_y(x, y, method='polynomial', return_plot=False, deg=1, n_neighbours=5): 
	'''
	Given a set of x and y values, this computes the expected y value (or average) as a 
	function of x. The defined method will compute this via fitting a polynomial, rolling
	average, ...
	'''

	assert method in ["polynomial", "running_mean"], "Undefined method!"

	if method == "polynomial":
		# fitting polynomial to x - y plane.
		z = np.polyfit(x, y, deg)
		p = np.poly1d(z)

	elif method == "running_mean":
		# sorting y values by x ordering.
		x_sorted = np.sort(x)
		y_sorted = y[np.argsort(x)]
		y_running_mean = uniform_filter1d(y_sorted, size=n_neighbours)

		# creating interpolation function to be used in future.
		p = interp1d(x_sorted, y_running_mean)

	# returning plot of fitted polynomial with points if required.
	if return_plot == True:
		plt.cla()
		plt.close()
		fig, ax = plt.subplots()
		ax.scatter(x, y, s=2, alpha=0.3, c='salmon')
		# plotting polynomial over same range as the data.
		xrange = np.linspace(np.min(x), np.max(x))
		yrange = p(xrange)
		ax.plot(xrange, yrange, color='skyblue', linewidth=5, alpha=0.7)
		plt.show()
		
	return p

