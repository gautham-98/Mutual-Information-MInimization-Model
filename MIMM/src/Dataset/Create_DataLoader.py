import sys
import os
sys.path.insert(0, os.getcwd()+'/MIMM/src')

from Data.load_data import load_data_MorphoMNIST, load_data_chXpert
from Dataset.Create_Dataset import create_dataset
from Dataset.Create_Dataset_chXpert import create_dataset_chXpert


def init_morphoMNIST_training(params_dict, training_dataset):
    # Load data and create loaders
    trainData, valData = load_data_MorphoMNIST()  # trainData and valData are dictionaries containing sample, label and pertlabel created from the directory 
    #creates confounded trainloader, valloader and testloader
    trainLoader, valLoader = create_dataset(trainData, valData, params_dict, training_dataset)
    _, testLoader = create_dataset("", valData, params_dict, training_dataset, valData=True)
    #creates equal confounding test dataset i.e for every primary task label 50% will be from thick and 50% from thin same with rotation
    params_dict["confoundingRatio"]["value"] = [0.5] * params_dict["num_sc_variables"]["value"]
    _, testEqualLoader = create_dataset("", valData, params_dict, training_dataset, valData=True)
    params_dict["confoundingRatio"]["value"] = [0.9] * params_dict["num_sc_variables"]["value"]

    return trainLoader, valLoader, testLoader, testEqualLoader

def init_chXpert_training(params_dict):
    labels = params_dict["labels"]["value"]
    chXpert_data_train, chXpert_data_val, chXpert_data_test, chXpert_data_test_native = load_data_chXpert(['Path']+labels)
    train_loader = create_dataset_chXpert(chXpert_data_train, labels, params_dict["confoundingRatio"]["value"], rotation=0, batchSize=params_dict['batchSize']['value'],shuffle=True)
    val_loader = create_dataset_chXpert(chXpert_data_val, labels, params_dict["confoundingRatio"]["value"], rotation=0)
    test_loader = create_dataset_chXpert(chXpert_data_test, labels, params_dict["confoundingRatio"]["value"], rotation=1)
    equal = [0.5,0.5]
    test_equal_loader = create_dataset_chXpert(chXpert_data_test, labels, confounding_ratio=equal, rotation=1)
    test_native_loader = create_dataset_chXpert(chXpert_data_test_native, labels, None, rotation=1)
    train_equal_loader = create_dataset_chXpert(chXpert_data_train, labels, confounding_ratio=equal, rotation=0)
    return train_loader, val_loader, test_loader, test_equal_loader, test_native_loader, train_equal_loader 