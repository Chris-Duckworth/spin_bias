![tree](./plots/random_forest/tree.png)

# Galaxy spin and environment
This repo contains code to understand how a galaxy spin depends on properties such as stellar mass, halo mass, group membership, and, _large-scale environment_. 

A galaxy's spin appears to be strongly correlated with morphology, with secondary dependences on stellar mass and inclination (i.e. λ<sub>R</sub> is a biased obervational estimate), however, local and large-scale environment (filamentary structure) are informative (to a lesser degree). This is detailed in chapter 4 of my [thesis](https://github.com/Chris-Duckworth/Thesis), and, best quickly summarised here based on results from `./scripts/random_forest/` and `./plots/random_forest/`.

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

To evaluate the importance of various galaxy properties (including local and large-scale environment) in predicting a galaxy's spin, we generate a random forest to predict λ<sub>R</sub>. 

### Input features

- (n<sub>sersic</sub>) sersic index
- (b/a) inclination
- (M<sub>stel</sub>) stellar mass
- (M<sub>halo</sub>) halo mass
- (D<sub>node</sub>) distance to nearest node (of the cosmic web)
- (D<sub>skel</sub>) distance to nearest filament segment (of the cosmic web)
- (cen/sat) group membership (i.e. if the galaxy is the most massive (central) in the group or not (satellite))

We test various parameterisations of _morphology_ including one-hot encoding of galaxyZoo classifications (i.e. ETGs, LTGs etc.), empirically estimated morphological T-type (using vote fractions in GZ), however, sersic index appears to be most informative for predicting galaxy spin (see `./plots/random_forest/morphology_importances.pdf`.

### Hyperparamter tuning 

The best performing random forest is found using the `sklearn` optimized (cross-validated) grid-search method [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html). In contrast to a full grid-search, not all parameter values are tried, however, a fixed number of setting combinations are sampled from the distribution. We find there is no significant drop-off in performance between various hyperparameter combinations, however, our best random forest set-up is found to be:

```
{'n_estimators': 1000,
 'min_samples_split': 5,
 'min_samples_leaf': 4,
 'max_features': 'sqrt',
 'max_depth': 10}
 ```
 
### Model fitting
Using the following parameters and hyperparameters, a random forest is generated using the cross-matched data from the catalog class (6117 galaxies), where 80% is used for training and 20% for testing. We find that the model predictions have a mean absolute error of 0.12538. To compare the performance of the network (and importance of secondary parameters), we also generate a random forest (with the same hyperparameters) just using n<sub>sersic</sub> and b/a. We find the model predictions to have a mean absolute error of 0.12937. The λ<sub>R</sub> predicted distributions for the random forests (using all parameters and just the two most important (n<sub>sersic</sub>, b/a)) and the actual values are shown here:

![hist](./plots/random_forest/lambda_R_predicted_actual_distn.png)

To evaluate the performance of the predictions we also plot the correlation (with pearson correlation coeffients) with the actual λ<sub>R</sub> values for the test data. We find that the random forests make reasonably informed predictions of λ<sub>R</sub>, however, struggle to predict low or high values. The correlation plot is shown here:

![corner](./plots/random_forest/lambda_R_predicted_actual_correlation_corner.png)

### Feature importances
A natural output of random forests is the relative importances of input features (based on the frequency of a given feature in nodes across all of the decision trees in the forest). The relative importances (i.e. normalised so they all sum to one) are given here:

![imp](./plots/random_forest/importances.png)

We find that sersic index and inclination are most informative of predicting λ<sub>R</sub>, however there are reasonable contributions from stellar and halo mass along, and, small but significant contributions from cosmic-web environment. Group membership appears to be insignificant (however is encoded as a binary discrete parameter, which may impact its importance ranking relative to all other features which are continuous). 
