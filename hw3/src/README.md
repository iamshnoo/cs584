# File structure

```data_loader.py``` - Defines a representation for the dataset, and functions
to load the data from the given input files.

```pre_process.py``` - Defines steps involved in pre-processing the data as a
single prep-processing class.

```adaboost.py``` - Defines the Adaboosting ensemble algorithm.

```utils.py``` - Utility functions used throughout the other files (cross
validation, printing, etc.)

```main.py``` - The main program. Run this file to verify results.

```configs.yaml``` - Defines the configurations used in ```main.py```

```main.py``` will load configurations from ```configs.yaml```.
Then it will load the data, apply pre-processing steps to reduce dimensionality,
load Adaboost classifier with the specified decision stump and then
run cross-validation on the data, before making final test predictions.
