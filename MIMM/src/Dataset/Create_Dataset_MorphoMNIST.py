
import torch
import random
from Dataset.Utils import getDataLoader, set_new_labels, get_distribution, get_images
from Dataset.Dataset import DatasetMultiLabels



def create_MorphoMNIST_dataset_confounded(datasetType, originalDataset, params_dict, rotation=0, batchSize = 10, shuffle=False):
    # Get the preset hyperparameters

    # Extract data, original class labels (0-9), perturbed labels (writing style)
    dataset = originalDataset[datasetType + "Data"].cuda()
    datasetLabels = originalDataset[datasetType + "DataLabel"].cuda()
    datasetPertLabels = originalDataset[datasetType + "PertDataLabel"].cuda()

    # change the PerLabels, since 1: thin, 2: thick to 0:thin and 1:thick
    datasetPertLabels = [x-1 for x in datasetPertLabels]

    # create the new label CL (0, 1) from the original numbers 0-9
    classGroupLabels = groupNumbersAsNewClassLabels_2_ptasses(datasetLabels)

    full_dataset = list(zip(dataset, classGroupLabels, datasetPertLabels))
    
    # create multiple confounding
    confounded_dataset = create_MultiConfounded_dataset(full_dataset, params_dict, rotation)
    # Get dataloader
    dataLoader = getDataLoader(DatasetMultiLabels, confounded_dataset, batchSize, shuffle=shuffle)
    get_distribution(dataLoader, datasetType)
    #get_images(dataLoader, datasetType)
    return dataLoader


def create_MultiConfounded_dataset(full_dataset, params_dict, rotation):
    # perform single confounding for different sc_variables
    for sc_variable in range(params_dict["num_sc_variables"]["value"]):
        full_dataset = perform_single_variable_confounding(full_dataset, params_dict, sc_variable, rotation)
    confounded_dataset = full_dataset
    return confounded_dataset


def perform_single_variable_confounding(full_dataset, params_dict, sc_variable, rotation):
    # thick thin confounding is the first confounding applied
    if sc_variable==0:
        return ThickThin_confounded_dataset(full_dataset, params_dict, sc_variable, rotation)
    
    else:
        return scVariable_confounded_dataset(full_dataset, params_dict, sc_variable, rotation)
    

def scVariable_confounded_dataset(full_dataset, params_dict, sc_variable, rotation):
    # split the dataset for primary tasks
    dataset_pt_0 = [data for data in full_dataset if data[1]==0]
    dataset_pt_1 = [data for data in full_dataset if data[1]==1]
    confounding_ratio = params_dict["confoundingRatio"]["value"][sc_variable]
    
    # create new label for additional sc_variable according to confounding ratio and the primary task
    dataset_pt_0_sc = set_new_labels(dataset_pt_0, confounding_ratio, rotation)
    dataset_pt_1_sc = set_new_labels(dataset_pt_1, confounding_ratio, rotation)
   
    sc_labelled_dataset = dataset_pt_0_sc + dataset_pt_1_sc
    random.shuffle(sc_labelled_dataset)
    
    if sc_variable==1:
        confounded_dataset = apply_rot(sc_labelled_dataset)

    # if required add more confounding here

    return confounded_dataset


def ThickThin_confounded_dataset(full_dataset, params_dict, sc_variable, rotation):
    # two types of classes and confounding
    classes_pt = list(range(2))
    classes_sc = [x for x in list(range(2))]

    # sort the dataset by classes so that they can be seperated later
    full_dataset = sorted(full_dataset, key=lambda x: x[2])
    full_dataset = sorted(full_dataset, key=lambda x: x[1])

    dataset_splitted_pt_and_sc = []
    for i in classes_pt:
        dataset_splitted_pt_and_sc.append([])
        for j in classes_sc:
            dataset_splitted_pt_and_sc[i].append([])

    for i, _ in enumerate(full_dataset):
        for pt_idx, pt in enumerate(classes_pt):
            for sc_idx, sc in enumerate(classes_sc):
                if full_dataset[i][1] == pt and full_dataset[i][2] == sc:
                    dataset_splitted_pt_and_sc[pt_idx][sc_idx].append(
                        full_dataset[i])
    # returns a list containing dataset arranged as ds[
    #                                                   pt_list_0[sc_list_0[full_dataset_list[]], sc_list_1[full_dataset_list[]]],
    #                                                   pt_list_1[sc_list_0[full_dataset_list[]], sc_list_1[full_dataset_list[]]]]
    #                                                 ]

    minlen = min([len(dataset_splitted_pt_and_sc[0][0]), len(dataset_splitted_pt_and_sc[0][1]), len(dataset_splitted_pt_and_sc[1][0]), len(dataset_splitted_pt_and_sc[1][1])])
    for pt_idx, pt in enumerate(classes_pt):
        for sc_idx, sc in enumerate(classes_sc):
            dataset_splitted_pt_and_sc[pt_idx][sc_idx] = dataset_splitted_pt_and_sc[pt_idx][sc_idx][:minlen]


    if params_dict["confoundingRatio"]["value"][sc_variable] != 0:
        confounding_ratio = params_dict["confoundingRatio"]["value"][sc_variable]
        non_confounding_ratio = (1-confounding_ratio)/(len(classes_sc)-1)
        #division by len(classes_sc)-1 to equally distribute the rest of the ncr% of pt_list_0/1 to the rest of the sc_classes 
    else:
        confounding_ratio, non_confounding_ratio = 1

    confounded_dataset = []
    for pt_idx, pt in enumerate(classes_pt):
        for sc_idx, sc in enumerate(classes_sc):
            if (rotation == 0 and sc_idx == pt_idx) or ((rotation == 1) and (sc_idx == ((pt_idx+rotation) % len(dataset_splitted_pt_and_sc)))):
                nrSamples = int(
                    len(dataset_splitted_pt_and_sc[pt_idx][sc_idx])*confounding_ratio)
            else:
                nrSamples = int(
                    len(dataset_splitted_pt_and_sc[pt_idx][sc_idx])*non_confounding_ratio)

            dataset_splitted_pt_and_sc[pt_idx][sc_idx] = random.sample(
                dataset_splitted_pt_and_sc[pt_idx][sc_idx], nrSamples)
            confounded_dataset.extend(
                dataset_splitted_pt_and_sc[pt_idx][sc_idx])

    return confounded_dataset
    

def groupNumbersAsNewClassLabels_2_ptasses(datasetLabels, counts=False):
    """
    Used to split the dataset from the original class label 0-9 to new class groups, where:
    class 0 contains the numbers 0,1,2, class 1 contains 3-6, class 2 contains 7-9

    Args:
        datasetLabels (list(int)): list of the original labels 0-9 of all images

    Returns:
        newDataLabel (list(int)): list of the new labels 0-2 of all images
    """
    newDataLabel = []
    smaller = 0
    greater = 0
    for label in datasetLabels:
        if label < 5:
            newDataLabel.append(0)
            smaller += 1
        elif label >= 5:
            newDataLabel.append(1)
            greater += 1
    newDataLabel = torch.LongTensor(newDataLabel).cuda()

    if counts:
        return newDataLabel, smaller, greater
    return newDataLabel

def apply_rot(dataset):
    """ function to apply rotation confounding
      to apply new confounding create a similar function seperately"""
    for i, (image, *labels) in enumerate(dataset):
        if labels[-1] == 1:
            image = torch.rot90(image, k=1, dims=[1,2])
        else:
            image = image
        dataset[i] = (image, *labels)
    return dataset
