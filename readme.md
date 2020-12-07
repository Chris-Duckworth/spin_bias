# Galaxy spin and environment
This repo contains code to understand how a galaxy spin depends on properties such as stellar mass, halo mass, group membership, and, large-scale environment. 

A galaxy's spin appears to be strongly correlated with morphology, with secondary dependences on stellar mass and inclination (i.e. λ<sub>R</sub> is a biased obervational estimate), however, local and large-scale environment (filamentary structure) are informative (to a lesser degree). This is detailed in chapter 4 of my [thesis](https://github.com/Chris-Duckworth/Thesis), and, best quickly summaried in ./scripts/random_forest. 

## Data 
Data is taken from various sources and a basic summary of each catalogue is given here, however, for more detail see [here](https://github.com/Chris-Duckworth/Thesis) and the references therein.

- Integral field unit observations are from the [MaNGA](https://www.sdss.org/surveys/manga/) galaxy survey, which is processed by the internal Data Analysis Pipeline [DAP](https://www.sdss.org/dr15/manga/manga-analysis-pipeline/). This is used to compute λ<sub>R</sub>, a flux weighted measure of coherent rotation of a galaxy [see here](https://ui.adsabs.harvard.edu/abs/2007MNRAS.379..401E/abstract). 

- Additional information is taken from the [NASA-Sloan Atlas](https://www.sdss.org/dr13/manga/manga-target-selection/nsa/) targetting catalogue which provides stellar mass, and, galaxy inclination. For all galaxies in MaNGA, morphological classifications from citizen science project [galaxyZoo](https://www.sdss.org/dr15/data_access/value-added-catalogs/?vac_id=manga-morphologies-from-galaxy-zoo) are found. 

- These catalogues are cross-matched with [group catalogues](https://gax.sjtu.edu.cn/data/Group.html) found from galaxies in the SDSS-DR7 spectroscopic sample, which provides halo mass, and, central/satellite definition.

- [Cosmic web catalogues](https://arxiv.org/abs/1710.02676) are also cross-matched to this data to provide distances to morphological features of the cosmic web such as distances to filaments and nodes. 

The total number of MaNGA galaxies (for MPL-9) after cross-matching information from each of these catalogues (top row) and cumulatively cross-matching (bottom row) are given here:
| | MaNGA (w GalaxyZoo) | Group membership | Cosmic-web | 
| ------------- | ------------- | ------------- | ------------- | 
| Cross-matched  | 7398 | 6343 | 6378 |
| Cumulative cross-matched | 7398 | 6343 | 6117 |

### Catalog class object
Data catalogues (and various versions of MaNGA data releases) are brought together by the catalog class object found [here](./lib/catalog.py), which performs the cross-matching. Catalog class objects store cross-matched information in the form of a pandas.DataFrame object (stored as property catalog.df). 

The catalog class objects also contain various methods to select galaxy sub-samples based on these properties (`./lib/catalog_init.py`), for data processing `./lib/catalog_process.py` and plotting `./lib/catalog_plot.py`. These methods are tied together in `./catalog.py`, however, are mainly used for the `./scripts/thesis_plots` directory.

## Random Forest

To evaluate the importance of various galaxy properties (including local and large-scale environment) in predicting a galaxy's spin 

## Thesis plots
