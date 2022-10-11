# File structure

```data_loader.py``` - Defines a representation for the dataset, and functions
to load the data from the given input files.
```pre_process.py``` - Defines steps involved in pre-processing the data as a
single prep-processing class.
```model.py``` -  Defines a wrapper class for calling sklearn models.
```utils.py``` - Utility functions used throughout the other files.
```cross_validation.py``` - Defines the cross-validation logic.
```main.py``` - The main program. Run this file to verify results.
```configs.yaml``` - The final configurations used in ```main.py```
```experiments.py``` - Some ablation studies.

```main.py``` will load configurations from ```configs.yaml```.
Then it will load the data, apply pre-processing steps to reduce dimensionality,
load the three models (Multinomial NB, Decision Tree, and Perceptron) and then
run cross-validation on the data, before making final test predictions.
The outputs expected on running ```main.py```  with the current configurations
is given in ```outputs.txt```.

```experiments.py``` is a file that I used for running different grid searches
and experiments on combinations of parameters.
