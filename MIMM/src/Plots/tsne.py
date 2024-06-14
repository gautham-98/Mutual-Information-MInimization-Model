import os
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.manifold import TSNE 

import config.Load_Parameter

from Models.FeatureEncoder import FeatureEncoderNetwork, MedicalFeatureEncoder
from Models.ClassificationHeads import PTModel, SCModel

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

def numpy(t):
    return t.cpu().numpy()

def  tsne(dataLoader, fv_modelPathToFile, pt_modelPathToFile,sc_modelPathToFile_list, params, datatype=''):
    
    # Get the hyperparameters saved in yaml.
    torch.manual_seed(params.randomSeed)
    training_dataset = params.training_dataset.casefold()
    # Set up model
    if "MorphoMNIST".casefold() in training_dataset:
        fv_model = FeatureEncoderNetwork().cuda()
    elif "chXpert".casefold() in training_dataset.casefold():
        fv_model = MedicalFeatureEncoder().cuda()
    else:
        raise ValueError('No valid trainType selected. No model found')
    
    fv_model.load_state_dict(torch.load(fv_modelPathToFile))
    fv_model.cuda()
    fv_model.eval()

    pt_model = PTModel()
    if pt_modelPathToFile:
        pt_model.load_state_dict(torch.load(pt_modelPathToFile))
        pt_model.cuda()
        pt_model.eval()
    if sc_modelPathToFile_list:
        sc_model_list = []
        for sc_modelPathToFile in sc_modelPathToFile_list:
            sc_model = SCModel()
            sc_model.load_state_dict(torch.load(sc_modelPathToFile))
            sc_model.cuda()
            sc_model.eval()
            sc_model_list.append(sc_model)


    # Save images in list
    samples = []
    if "chXpert".casefold() not in training_dataset.casefold():
        for imgInit, *labels in dataLoader.dataset.dataset:
            img = imgInit.clone()
            img.requires_grad_()
            samples.append((img, *labels))
    else:
        for imgInit, *labels in dataLoader.dataset:
            img = imgInit.clone()
            img.requires_grad_()
            samples.append((img, *labels))

    fv_list_pt = []
    listOf_fv_list_sc = []
    listOf_label_list= []
    for _ in range(params.num_sc_variables):
        listOf_fv_list_sc.append([])
    for _ in range(params.num_sc_variables+1):
        listOf_label_list.append([])

    for img, *labels in samples:
        if "chXpert".casefold() not in training_dataset.casefold():
            fv = fv_model(img.reshape((1,1,params.input_img_size,params.input_img_size)))
        else:
            if config.Load_Parameter.params.selectFeatureEncoder == 0:
                fv = fv_model(img.reshape((1,1, params.input_img_size,params.input_img_size)))
            elif config.Load_Parameter.params.selectFeatureEncoder == 1:
                fv = fv_model(img.reshape((1,3, params.input_img_size,params.input_img_size)))
        # Split fv (feature partioning)
        fv_len = int(fv.size(1) / (params.num_sc_variables+1))

        fv_pt = fv[:, :fv_len]
        fv_sc_list = []
        for sc_task in range(params.num_sc_variables):
            start = (sc_task+1) * fv_len
            end   = (sc_task+2) * fv_len
            fv_sc_list.append(fv[:, start:end])

        fv_list_pt.append(fv_pt.detach().cpu().numpy())
        for sc_task, fv_list_sc in enumerate(listOf_fv_list_sc):
            fv_list_sc.append(fv_sc_list[sc_task].detach().cpu().numpy())
        
        for task, label_list in enumerate(listOf_label_list):
            label_list.append(labels[task])

    fv_arr_pt = np.stack(fv_list_pt, axis = 1)[0,:,:]
    listOf_fv_arr_sc = []
    for fv_list_sc in listOf_fv_list_sc:
        fv_arr_sc = np.stack(fv_list_sc, axis = 1)[0,:,:]
        listOf_fv_arr_sc.append(fv_arr_sc)
    # Create TSNE Vectors
    tsne_mtl =  TSNE(n_components=2, random_state=0).fit_transform(fv_arr_pt)
    listOf_tsne_mtl_sc = []
    for fv_arr_sc in listOf_fv_arr_sc:
        tsne_mtl_sc =  TSNE(n_components=2, random_state=0).fit_transform(fv_arr_sc)
        listOf_tsne_mtl_sc.append(tsne_mtl_sc)
    
    # Create plots
    Path(os.path.expanduser('~')+"/MIMM/src/Plots/"+params.training_dataset+"_"+datatype+"/"+params.runName).mkdir(parents=True, exist_ok=True)
    listOf_labels = params.listOf_labels

    for task,(label_list, labels) in enumerate(zip(listOf_label_list, listOf_labels)):
        create_tsne_plot(label_list, tsne_mtl, plottitle=params.trainType, plotname=params.training_dataset+"_"+datatype+'/'+params.runName+"/"+f"_V_PT_L_task{task}", label=labels)
    
    for sc_task,tsne_mtl_sc in enumerate(listOf_tsne_mtl_sc):
        for task, (label_list, labels) in enumerate(zip(listOf_label_list, listOf_labels)):
            create_tsne_plot(label_list, tsne_mtl_sc, plottitle=params.trainType, plotname=params.training_dataset+"_"+datatype+'/'+params.runName+"/"+f"_V_SC{sc_task}_L_task{task}", label=labels)
  
def create_tsne_plot(label_list, tsne_vector, plottitle, plotname, label):
    params = config.Load_Parameter.params
    l0_mtl_x, l1_mtl_x, l0_mtl_y, l1_mtl_y= [], [], [], []
    for l, elem in zip(label_list, tsne_vector):
        if l == 0:
            l0_mtl_x.append(elem[0])
            l0_mtl_y.append(elem[1])
        else:
            l1_mtl_x.append(elem[0])
            l1_mtl_y.append(elem[1])
    plt.figure(figsize=(8,6))
    plt.scatter(l0_mtl_x, l0_mtl_y, color = "r", s= [3], label=label[0])
    plt.scatter(l1_mtl_x, l1_mtl_y, color = "b", s= [3], label=label[1])
    plt.title(plottitle, fontsize=16)
    plt.legend(fontsize=16)
    plt.xlabel("Dimension 1", fontsize=16)
    plt.ylabel("Dimension 2", fontsize=16)
    plt.savefig(os.path.expanduser('~')+"/MIMM/src/Plots/"+plotname+".svg")
    plt.close()
