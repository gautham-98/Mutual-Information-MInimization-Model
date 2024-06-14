import os
import sys

sys.path.insert(0, os.getcwd() + '/MIMM/src')

import wandb
import time
import torch
import torch.nn as nn
from torch import optim

import config.Load_Parameter
from Models.FeatureEncoder import FeatureEncoderNetwork, MedicalFeatureEncoder
from Models.ClassificationHeads import PTModel, SCModel
from Models.MINE import MIComputer
from Training.trainers import FETrainer, MITrainer, Validate
from SaveWandbRuns.initWandb import save_best_models_in_wandb, create_experiments_model_folder


def mtl_mi_train(trainLoader, valLoader):
    print("===============================start training=================================")
    params = config.Load_Parameter.params
    torch.manual_seed(params.randomSeed)

    fv_model, pt_model, sc_model_list, mi_model_list = get_models(params)
    set_models_to_train(fv_model, pt_model, sc_model_list, mi_model_list)
    optimizer, mi_optimizer = get_optimizers(fv_model, pt_model, sc_model_list, mi_model_list, learning_rate=params.lr)
    
    # Training
    start_time = time.time()
    num_epochs = params.epochs 
    best_accuracy_pt = 0
    best_accuracy_sc = 0
    best_epoch = 0
    best_mi = 0

    fe_trainer = FETrainer(fv_model, pt_model, sc_model_list, mi_model_list, trainLoader, optimizer)
    mi_trainer = MITrainer(fv_model, mi_model_list, trainLoader, mi_optimizer)
    validate = Validate(fv_model, pt_model, sc_model_list, mi_model_list, valLoader)
    batches_per_epoch = len(trainLoader)
    fe_max_batches = params.fe_max_batches - 1
    mi_max_batches = params.mi_max_batches - 1

    if (fe_max_batches == 0):
        steps_per_epoch = batches_per_epoch
    else:
        steps_per_epoch = int(batches_per_epoch/(fe_max_batches+1))

    for epoch in range(1, num_epochs+1):
        wandb.log({'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}, commit=False)
        # train for an epoch
        for _ in range(steps_per_epoch):
            train_step(epoch, fe_trainer, mi_trainer, fe_max_batches, mi_max_batches)
        # validate
        set_models_to_eval(fv_model, pt_model, sc_model_list, mi_model_list)
        y_val_mi_mean, y_val_mi_mean_loss, val_pt_sc_loss, val_loss, val_accuracy_pt, val_accuracy_sc, mean_val_accuracy = validate(epoch)
        set_models_to_train(fv_model, pt_model, sc_model_list, mi_model_list)

        # get the best epoch -- use this for early stopping if rquired
        if (val_accuracy_pt > best_accuracy_pt) & (mean_val_accuracy > best_accuracy_sc):
            best_epoch = epoch
            best_accuracy_pt = val_accuracy_pt
            best_accuracy_sc = mean_val_accuracy
            best_mi = y_val_mi_mean.item()
            wandb.log({'best_epoch': best_epoch, 'best_acc_pt': val_accuracy_pt,
                        **{f'best_val_accuracy_sc_{i}': acc for i, acc in enumerate(val_accuracy_sc)}, 'best_mi': best_mi})

    # get the trained models at the end of an epoch
    fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile_list, mi_modelPathToFile_list = create_experiments_model_folder(params, fv_model, pt_model, sc_model_list, mi_model_list)
    save_best_models_in_wandb(params.trainType, fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile_list,
                              mi_modelPathToFile_list)
    
    # Report
    training_time = round((time.time() - start_time) / 60)  # in minutes
    print(f'Training done in {training_time} minutes')
    wandb.finish()
    print("=================================end training=================================")
    return fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile_list, mi_modelPathToFile_list


def train_step(epoch, fe_trainer, mi_trainer, fe_max_batches, mi_max_batches):
    # train MINE for MTL_MI
    if config.Load_Parameter.params.trainType == 'MTL_MI':
        fe_trainer(epoch, max_batches=fe_max_batches)
        mi_trainer(epoch, repeat = 1, max_batches=mi_max_batches)
    # Do not train MINE for baseline model
    if config.Load_Parameter.params.trainType == 'MTL_no_MI':
        fe_trainer(epoch,  max_batches=fe_max_batches)

def get_models(params):
    if params.choose_mi_loss==0:
        num_of_mi_models = int((params.num_sc_variables*(params.num_sc_variables+1))/2)
    else:
        num_of_mi_models = params.num_sc_variables

    training_dataset = params.training_dataset
    if "MorphoMNIST".casefold() in training_dataset.casefold():
        fv_model = FeatureEncoderNetwork().cuda()
    elif "chXpert".casefold() in training_dataset.casefold():
        fv_model = MedicalFeatureEncoder().cuda()
    else:
        raise ValueError('No valid trainType selected. No model found')
    pt_model = PTModel().cuda()
    sc_model_list = []
    mi_model_list = []
    
    for _ in range(num_of_mi_models):
        sc_model_list.append(SCModel().cuda())

    for i in range(num_of_mi_models):
        mi_model_list.append(MIComputer(i).cuda())
    return fv_model, pt_model, sc_model_list, mi_model_list

def set_models_to_train(fv_model, pt_model, sc_model_list, mi_model_list):
    fv_model.train()
    pt_model.train()
    for sc_model in sc_model_list:
        sc_model.train()
    for mi_model in mi_model_list:
        mi_model.train()

def set_models_to_eval(fv_model, pt_model, sc_model_list, mi_model_list):
    fv_model.eval()
    pt_model.eval()
    for sc_model in sc_model_list:
        sc_model.eval()
    for mi_model in mi_model_list:
        mi_model.eval()    

def get_optimizers(fv_model, pt_model, sc_model_list, mi_model_list , learning_rate=1e-5):
    fv_params = [p for p in fv_model.parameters()]
    pt_params = [p for p in pt_model.parameters()]

    sc_params = []
    for sc_model in sc_model_list:
        sc_params_to_add = [p for p in sc_model.parameters()]
        sc_params.extend(sc_params_to_add)
    
    mi_params = []
    for mi_model in mi_model_list:
        mi_params_to_add = [p for p in mi_model.parameters()]
        mi_params.extend(mi_params_to_add)
    optimizer = optim.Adam(fv_params + pt_params + sc_params, lr=learning_rate, weight_decay=1e-4)
    mi_optimizer = optim.Adam(mi_params, lr=learning_rate)
    return optimizer, mi_optimizer

                                 

