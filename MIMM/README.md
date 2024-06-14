# Datasets
1. The Morpho-MNIST dataset can be found [here](https://github.com/dccastro/Morpho-MNIST). Download the 'global' dataset and extract the files to the *MIMM/src/Data/MorphoMNIST* directory.
2. The CheXpert-Small dataset has been downloaded in the iss server at */data/public/chexpert*.
# Prerequisits
```
pip install -r requirements.txt
```

# Running
The config file for morphoMNIST experiments is morphomnist.yml and for the CheXpert experiments the config file is chXpert.yml.
```
python MIMM/src/main.py config.yml
```
