# Data_convex_hull
Creates a convex hull arround a training set and defines which observations from another set are within the hull

A convex hull can be used to draw boundaries around the training set. This can be useful to verify regions of feature space in which a dataset of interest  is "covered" by the training set.

data_hull.py creates a convex hull using the n first principal components of the training set, and outputs the indexes of the dataset of interest that are whithin the hull

The datasets must be csv dataframes containing the normalized features in each column

Usage

python3.8 data_hull.py --training training_set.csv --prediction prediction.csv -npcs 3

(here, using the 3 first princiapl components)
