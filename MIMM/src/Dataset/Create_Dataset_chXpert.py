from Dataset.Utils import create_multiple_correlation, getDataLoader
from Dataset.Dataset import ImageLabelDataset

def create_dataset_chXpert(dataframe, labels, confounding_ratio, rotation, batchSize=10, shuffle=False):
    if confounding_ratio:
        dataframe = create_multiple_correlation(dataframe, labels, confounding_ratio, rotate=rotation)

    dataLoader = getDataLoader(ImageLabelDataset, dataframe, batchSize, shuffle=shuffle)
    return dataLoader