# CNN-Hyperparameter-Tuning
The repository of the Hyperparameter Tuning research as executed by Richard Staszkiewicz and described in bechelor thesis and soon to follow article.

To recreate the Experiments conducted, run the *Run_Trials.ipynb* notebook, preferably in [Google Colaboratory](https://colab.research.google.com/). To recreate the research Result analysis, run the *Analyze_Trials.ipynb* notebool, also preferably in Google Colab.

## Logs
The results acquired during research. The direcotries responsible for Bayes Search (bs), Grid Search (gs) and Random Search (rs) contain complete tensorboards of all evaluations run and .csv concatenated results and metrics of each evaluation run. The directory of Differential Evolutionary Strategy (des) contain complete tesorboards of all evaluations run and an .npy pickled dictionary with results as a all-log mode run of utils.DES(). To load the results as dictionary use
```.py
import numpy as np
results = np.load('logs/des/DES-Trial1-29458s.npy', allow_pickle=True)[()]
```

## Model
### Multilayer Perceptron (resnet.py)
Module implements the classical MLP defined by MLP class constructed of SLPBlocks. Each block is defined by 5 parameters: in_size, out_size, activation_fun, batch_norm and dropout.
### ResNet (resnet.py)
Module implements the classical ResNet-like convolutional architecture constructed of ResBlocks. Eacch block is defined by 5 parameter: in_channnels, out_channels, kernel_size, stride and padding.
### Configs
Configs directory consists of an examples of configurations that could be used to construct the Model.
### Search Spaces
Search Spaces directory consisting of example configurations of search spaces in form of each algorithms specific configuration.

## Modules
A module implementing two main PyTorch Lightning Modules: *MNISTDataModule* that manages data (including batches) - in scope of this research an FashionMNIST Dataset and *MNISTClassifier* that implements the CNN based on ResNet and MLP modules. It is parametrized by a dictionary configuration, holding a MLP and ResNet configurations along with some global hyperparameters like Learning Rate, Weight Decay or ADAM both beta parameters.

## Utils
### Differenital evolution strategy (DES.py)
This module contains a full implementation of a vanilla DES algorithm represented by a des_classic function along with des_tuner_wrapper class that mimics the call of Ray.tune library on the original function.
### Transpilation (Transpilation.py)
This module is acting as a buffor between the PyTorch and Ray libraries by providing the executable functions of first one digested by the other. It also provides the repair function that makes the configurations consistant after introducing changes to the specified parameters. It also implements the loss function as utilized by a DES implementation.

*Note* All modules have a python-doc style documentations provided within.