import os
import torch
import json
import pandas as pd
from Dataset.Utils import load_idx,map_age_binary, remove_uncertain, remove_unmentioned, remove_PAviews, remove_lateralViews
from sklearn.model_selection import train_test_split

def load_data_MorphoMNIST(usedDataset='global'):
    """ Download data from https://github.com/dccastro/Morpho-MNIST
    Loads the dataset from file "Data/Paths.json". The datasetType defines if train or test dataset is loaded.

    Args:
        usedDataset (str): Decides which dataset is used. 'global' means the original MNIST. Defaults to 'global'.
    Returns:
        dataset (dict): returns the dataset in a dict. With Sample, Label digit, Label Pert.
    """

    # Get the folder paths of the datasets
    with open(os.path.expanduser('~')+'/MIMM/src/Data/MorphoMNIST/Paths.json') as f:
        paths = json.load(f)
    datasetTypes = ["train", "test"]
    nrSamples = [60000, 10000]
    datasets = [{}, {}]
    for i, (datasetType, nr, dataset) in enumerate(zip(datasetTypes, nrSamples, datasets)):
        # Save dataset as dict, each dataset consits of data, label (number) and pertlabel (writing style)
        dataset[datasetType+"Data"] = torch.FloatTensor(
            load_idx(os.path.expanduser('~')+paths[usedDataset+"_"+datasetType+"Path"]).reshape(nr, 1, 28, 28))
        dataset[datasetType+"DataLabel"] = torch.LongTensor(
            load_idx(os.path.expanduser('~')+paths[usedDataset+"_"+datasetType+"PathLabel"]))

        # Add the perturbation label, if global use the given pert label, if only plain: 0, if only thin: 1, if only thick: 2
        dataset[datasetType+"PertDataLabel"] = torch.LongTensor(
            load_idx(os.path.expanduser('~')+paths[usedDataset+"_"+datasetType+"PertPathLabel"]))
        datasets[i] = dataset

    # Return trainData, testData
    return datasets[0], datasets[1]


def load_data_chXpert(labels):
    file_path_train = '/data/public/chexpert/CheXpert-v1.0-small/train.csv'
    file_path_test_native = '/data/public/chexpert/CheXpert-v1.0-small/valid.csv'
    file_paths = [file_path_train, file_path_test_native]
    
    for file_path in file_paths:
        chXpert_data = pd.read_csv(file_path)
        sex_mapping = {'Male': 0, 'Female': 1}
        chXpert_data = map_age_binary(chXpert_data, boundary_low=50, boundary_high=60)
        chXpert_data['Sex'] = chXpert_data['Sex'].map(sex_mapping)
        chXpert_data['patient_id'] = chXpert_data["Path"].str.split('/').str[2]

        chXpert_data = chXpert_data[["patient_id", "Frontal/Lateral", "AP/PA"]+labels]
        chXpert_data = remove_lateralViews(chXpert_data)
        chXpert_data = remove_PAviews(chXpert_data)
        chXpert_data = remove_unmentioned(chXpert_data)
        chXpert_data = remove_uncertain(chXpert_data)

        #trim down to necessary
        chXpert_data = chXpert_data[labels]
        if 'valid' in file_path.casefold():
            chXpert_data_test_native = chXpert_data    
        else:
            chXpert_data_train_val, chXpert_data_test = train_test_split(chXpert_data, test_size=0.17, shuffle=True, random_state=42)
            chXpert_data_train, chXpert_data_val = train_test_split(chXpert_data_train_val, test_size=0.1, shuffle=True, random_state=42)

    return chXpert_data_train, chXpert_data_val, chXpert_data_test, chXpert_data_test_native