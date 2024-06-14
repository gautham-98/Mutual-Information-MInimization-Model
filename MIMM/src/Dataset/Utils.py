
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import struct
import gzip
import numpy as np
import random
import torch
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import cv2 as cv
from torchvision.transforms.functional import pad, resize
import config.Load_Parameter
import os

def getDataLoader(datasetType, dataset, batchSize, shuffle):

    dataLoader = DataLoader(dataset=datasetType(dataset),
                            batch_size=batchSize,
                            shuffle=shuffle, drop_last=True)
    return dataLoader
    
def logDataLoaderInfoFM(dataLoader):
    l = np.zeros((10,10), dtype=int)
    for elem in dataLoader.dataset.dataset:
        for i in range(10):
            for j in range(10):
                if elem[1]==i and elem[2]==j:
                    l[i,j]+=1
    print(l)

def _load_uint8(f):
    _, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def load_idx(path: str) -> np.ndarray:
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'rb') as f:
        return _load_uint8(f)

def load_file(path):
        return np.load(path).astype(np.float32)


def tensor_cuda(nr):
    return torch.tensor(nr, dtype=torch.long).cuda()


def set_new_labels(dataset, confounding_ratio, rotation):
    random.shuffle(dataset)
    total_length = len(dataset)
    major_length = int(confounding_ratio*total_length)

    if ((dataset[1][1]==1 and rotation==0) or (dataset[1][1]==0 and rotation==1)):
        label_major = tensor_cuda(1)
        label_minor = tensor_cuda(0)

    elif ((dataset[1][1]==0 and rotation==0) or (dataset[1][1]==1 and rotation==1)):
        label_major = tensor_cuda(0)
        label_minor = tensor_cuda(1)

    for i in range(major_length):
        dataset[i] = dataset[i] + (label_major,)
    for i in range(major_length, total_length):
        dataset[i] = dataset[i] + (label_minor,)
    return dataset


def get_distribution(dataLoader, datasetType):
    dataset = dataLoader.dataset.dataset
    _, *labels = dataset[1]
    len_dataset = len(dataset)
    num_labels = len(labels)
    num_images_label=[]

    for labels in range(num_labels):
        num_images_label.append([])

    count_0=0
    count_1=0
    for data in dataset:
        if data[1]==0:
            count_0 = count_0+1
        elif data[1]==1:
            count_1 = count_1+1
    num_images_label[0] = (count_0, count_1)


    for label in range(1, num_labels):
        count_00 = 0
        count_01 = 0
        count_10 = 0
        count_11 = 0
        for data in dataset:
            if data[1]==0 and data[label+1] == 0:
                count_00 = count_00+1
            elif data[1]==0 and data[label+1] == 1:
                count_01 = count_01+1
            elif data[1]==1 and data[label+1] == 0:
                count_10 = count_10+1
            elif data[1]==1 and data[label+1] == 1:
                count_11 = count_11+1    
        num_images_label[label] = (count_00, count_01, count_10, count_11)

    print(f"\n================{datasetType} Dataset Distribution==================")
    print(f"\nlength of dataset = {len_dataset}")
    print(f"\npt_label : count[(0,1)]={num_images_label[0]}")
    for label in range(1, num_labels):
        print(f"\nsc_label_{label-1} : count[(00,01,10,11)]={num_images_label[label]}")


def get_images(dataLoader, datasetType):
    dataset = dataLoader.dataset.dataset
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")

    images_file_path= f"./images_from_dataset/{datasetType}_{formatted_datetime}/"
    os.makedirs(images_file_path, exist_ok=True)

    for i, (image, *labels) in enumerate(dataset):
        image_name = f"image{i}_labels{labels[0]}{labels[1]}{labels[2]}.png"
        image_path = images_file_path + image_name
        save_image(image, image_path, normalize=False)



def visualise_distribution(dataLoader, experiment):
    if not hasattr(visualise_distribution, 'distribution_count'):
        visualise_distribution.distribution_count=0
    visualise_distribution.distribution_count+=1
    dataset = dataLoader.dataset.dataset

    if 'ChXpert'.casefold() not in experiment.casefold() :
        images, *labels_list = zip(*dataset)
        # copy the data to cpu for pandas
        images = [image.cpu().numpy() for image in images]
        labels_list = [tuple(label.item() for label in label_tuple) for label_tuple in labels_list]
    else:
        _dataset = []
        labels_list = []
        for column in dataset.columns:
            column_list = dataset[column].tolist()
            _dataset.append(column_list)
        for task in _dataset[1:]:
            labels_list.append(task)
        images = _dataset[0]

    pt_labels = [f"Y={label}" for label in labels_list[0]]
    dataset_dict = {'image': images, 'Y': pt_labels}
    home = os.path.expanduser("~")

    label_names_visualisation = ['Y']
    for i, sc_label in enumerate(labels_list[1:]):
        key = f'Z_{i}'
        sc_label = [f'{key}={label}' for label in sc_label]
        dataset_dict[key] = sc_label
        label_names_visualisation.append(key)
    
    df = pd.DataFrame(dataset_dict)
    unique_combinations = df.drop_duplicates(subset=df.columns[1:])
    grouped_df = df.groupby(by=list(unique_combinations.columns[1:]))['image'].count().reset_index(name='count')
    print(f"\ndistribution in details:\n{grouped_df.to_markdown(index=True)}")

    fig = px.sunburst(grouped_df, path=list(unique_combinations.columns[1:]), values='count', hover_data=label_names_visualisation)
    fig.data[0].textinfo='label+text+value'
    fig.update_layout( title_text=f'{experiment.replace("_", " ")}',
                       title_x=0.5,
                       width=800,
                       height=800,
                       title_font=dict(size=32)
                     )
    fig.update_traces(textfont=dict(size=32)) 
    # Directory path
    directory = '/usrhomes/s1455/MIMM/figures/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.write_image(f'{home}/MIMM/figures/data_distribution_{experiment}_{visualise_distribution.distribution_count}.svg')
    print(f"\nvisualised in {home}/MIMM/figures/data_distribution_{experiment}_{visualise_distribution.distribution_count}.svg")

def remove_lateralViews(data):
    mask = data["Frontal/Lateral"]=="Lateral"
    data = data[~mask]
    return data

def remove_PAviews(data):
    mask = (data["AP/PA"]=="PA") | (data["AP/PA"]=="LL")
    data = data[~mask]
    return data
    
def remove_unmentioned(data):
    data = data.dropna(axis='index')
    return data

def fill_unmentionedWithZero(data):
    data = data.fillna(0)
    return data

def remove_uncertain(data):
    mask = (data == -1).any(axis=1)
    data = data[~mask]
    return data

def map_age_binary(data, boundary_low, boundary_high):
    data['Age'] = np.where((data['Age']<boundary_high) & (data['Age']>boundary_low), np.nan, data['Age'])
    data.dropna(subset=['Age'], inplace=True)
    data['Age'] = np.where(data['Age']>=boundary_high, 1, 0)
    return data

def create_multiple_correlation(data, labels, confounding_ratios, rotate):
    grouped_data = data.groupby(by=labels)['Path'].count()
    max_count = int(min(grouped_data)/np.prod(confounding_ratios))  
    data_arr = []

    for label_values in grouped_data.index:
        mask = True
        cfr = 1

        for label, label_value in zip(labels, label_values):
            mask &= (data[label]==label_value)
        
        for idx in range(len(confounding_ratios)):
            pt_label = label_values[0]
            current_label = label_values[idx + 1]
            if (rotate == 0 and pt_label == 0) or (rotate == 1 and pt_label == 1):
                cfr *= confounding_ratios[idx] if current_label == 0 else 1 - confounding_ratios[idx]
            else:
                cfr *= confounding_ratios[idx] if current_label == 1 else 1 - confounding_ratios[idx]

        count = int(cfr*max_count)          
        _data = data[mask].sample(n=count, replace=False)
        data_arr.append(_data)   

    return pd.concat(data_arr, ignore_index=True) 

def get_padding(image):
    max_w = 400 
    max_h = 400
    
    imsize = image.size
    h_padding = (max_w - imsize[0]) / 2
    v_padding = (max_h - imsize[1]) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    
    return padding

def pad_resize_image(image):
        padded_img = pad(image, get_padding(image))
        resize_img = resize(padded_img, config.Load_Parameter.params.input_img_size)
        return resize_img
def apply_clahe(image):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
    image = clahe.apply(image)
    return image




        
            
            


                
        

