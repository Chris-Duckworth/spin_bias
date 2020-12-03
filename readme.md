# Galaxy spin and environment
This repo contains code to understand how a galaxy spin depends on properties such as stellar mass, halo mass, group membership, and, large-scale environment. 

A galaxy's spin appears to be driven by stellar mass, however, shows secondary dependences on cluster-centric distance and even filamentary structure. This is detailed in chapter 4 of my [thesis](https://github.com/Chris-Duckworth/Thesis), and, best quickly summaried in ./scripts/random_forest. 

## Data 
Data is taken from various sources and a basic summary of each catalogue is given here, however, for more detail see [here](https://github.com/Chris-Duckworth/Thesis) and the references therein.

Integral field unit observations are from the [MaNGA](https://www.sdss.org/surveys/manga/) galaxy survey, which is processed by the internal Data Analysis Pipeline [DAP](https://www.sdss.org/dr15/manga/manga-analysis-pipeline/). This is used to compute $\lambda_R$, a flux weighted measure of coherent rotation of a galaxy [see here](https://ui.adsabs.harvard.edu/abs/2007MNRAS.379..401E/abstract). 

Additional information is taken from the [NASA-Sloan Atlas](https://www.sdss.org/dr13/manga/manga-target-selection/nsa/) targetting catalogue which provides stellar mass, and, galaxy inclination.

For all galaxies in MaNGA, morphological classifications from citizen science project [galaxyZoo](https://www.sdss.org/dr15/data_access/value-added-catalogs/?vac_id=manga-morphologies-from-galaxy-zoo) are found 


These catalogues are cross-matched with [group catalogues](https://gax.sjtu.edu.cn/data/Group.html) found from galaxies in the SDSS-DR7 spectroscopic sample, which provides halo mass, and, central/satellite definition.

[Cosmic web catalogues](https://arxiv.org/abs/1710.02676) are also cross-matched to this data 



Data is taken from the Data Analysis Pipeline (DAP) which provides stellar and (ionized) gas velocity fields. Velocity fields are found by fitting the stellar continuum and Hα spectral lines respectively.


Data is taken from the [MaNGA](https://www.sdss.org/surveys/manga/) galaxy survey and combined with [group catalogues](https://gax.sjtu.edu.cn/data/Group.html), and, [cosmic web catalogues](https://arxiv.org/abs/1710.02676).

### Catalog class object

### Random Forest

### Thesis plots
