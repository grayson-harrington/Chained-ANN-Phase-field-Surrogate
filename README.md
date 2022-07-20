
This directory contains all of the python files used for creating the full chained-ANN model for learning the relationship between Mg<sub>2</sub>Si<sub>x</sub>Sn<sub>1-x</sub> spinodal decomposition simulation parameters and output microstructures.

MLP_Classifier.py is where the classifier MLP archetecture is implemented using PyTorch and is used for training the MLP classifier
NN_classifier_main.py is where the classifier MLP is created, tested, and saved for later use.
CDataset.py is a helper for preparing the dataset input and output values for classification.

MLP_Regressor.py is where the regressor MLP archetecture is implemented using PyTorch and is used for training the MLP regressor.
NN_regressor_main.py is where the regressor MLP is created, tested, and saved for later use.
RDataset.py is a helper for preparing the dataset input and output values for regression.


PS-Linkage.py is the final python file which combines both the trained classification model and the trained regression model to create the final PS Linkage.
