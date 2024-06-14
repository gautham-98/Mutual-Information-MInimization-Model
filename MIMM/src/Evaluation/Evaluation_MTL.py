from Models.FeatureEncoder import FeatureEncoderNetwork, MedicalFeatureEncoder
from Models.ClassificationHeads import PTModel, SCModel
from SaveWandbRuns.initWandb import initWandb, saveWandbRun
from Training.Metrics_Utils import compute_metrics
from torchvision import models
import torch.nn as nn
import torch
import wandb
import os
import sys

import config.Load_Parameter
sys.path.insert(0, os.getcwd()+'/MIMM/src')

def get_switched_labels(i, y_val_pt, labels_sc_val):
    rotate = i+1
    labels = []
    labels.append(y_val_pt)
    labels.extend(labels_sc_val)
    labels = labels[rotate:] + labels[:rotate] 
    return labels[0], labels[1:]

def print_switched_labels(i):
    num_sc_variables = config.Load_Parameter.params.num_sc_variables
    fv = ["fv_pt"]
    labels = ["pt"]
    rotate = i+1
    for sc_var in range(num_sc_variables):
        fv.append(f'fv_sc{sc_var}')
    for sc_var in range(num_sc_variables):
        labels.append(f'sc{sc_var}')
    
    labels = labels[rotate:] + labels[:rotate]

    for fv, label in zip(fv, labels):
        print(f"{fv} predicts {label}") 

def evaluation_mtl(valLoader, testLoader, testEqualLoader, testNativeLoader, fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile_list, params_dict, paramsFile, saveFile, syncFile):
    print('=============================starting evaluation===========================')
    # Get the hyperparameters saved in yaml.
    params = config.Load_Parameter.params

    training_dataset = params_dict["training_dataset"]["value"]

    torch.cuda.empty_cache()
    # Set up model
    if "MorphoMNIST".casefold() in training_dataset.casefold():
        dataLoaders = valLoader + testLoader + testEqualLoader*2
        testTypes = ["Val"] + ["Test"]  + ["Test_Equal"] + ["Test_Switched"] 
        fv_model = FeatureEncoderNetwork().cuda()
    elif "chXpert".casefold() in training_dataset.casefold():
        dataLoaders = valLoader + testLoader + testNativeLoader + testEqualLoader*2
        testTypes = ["Val"] + ["Test"] + ["test_Native"] + ["Test_Equal"] + ["Test_Switched"] 
        fv_model = MedicalFeatureEncoder().cuda()
    else:
        raise ValueError('No valid trainType selected. No model found')

    fv_model.load_state_dict(torch.load(fv_modelPathToFile))
    fv_model.cuda()
    fv_model.eval()

    pt_model = PTModel()
    pt_model.load_state_dict(torch.load(pt_modelPathToFile))
    pt_model.cuda()
    pt_model.eval()

    sc_model_list = []
    for _ in range(params.num_sc_variables):
        sc_model_list.append(SCModel())
    for sc_model,sc_modelPathToFile in zip(sc_model_list, sc_modelPathToFile_list):
        sc_model.load_state_dict(torch.load(sc_modelPathToFile))
        sc_model.cuda()
        sc_model.eval()

    for testType, dataLoader in zip(testTypes, dataLoaders):

        # Save all predictions of all batches for an overall evaluation.
        y_val_pt_trues_as_list = []
        y_val_pt_preds_as_list = []
        y_val_sc_trues_as_list = []
        y_val_sc_preds_as_list = []

        # Evaluation
        with torch.no_grad():
            if "Switched" in testType:
                num_tests = params.num_sc_variables
            else:
                num_tests = 1
            
            for i in range(num_tests):
                params, run = initWandb(params_dict, paramsFile, testType=testType+"_")
                
                if "Switched" in testType:
                    print_switched_labels(i)
                # Batchwise evaluation
                for batch, (X_val, y_val_pt, *labels_sc_val) in enumerate(dataLoader):
                    # add a batch bin '[]' for every new batch
                    y_val_sc_trues_as_list.append([])
                    y_val_sc_preds_as_list.append([])

                    # Estimated CL and SC for one batch
                    if "Switched" in testType:
                        y_val_pt, labels_sc_val = get_switched_labels(i, y_val_pt, labels_sc_val)

                    fv = fv_model(X_val)
                    fv_len = int(fv.size(1) / (params.num_sc_variables+1))
                    fv_pt = fv[:, :fv_len]
                    # initalise the fv_sc_list add the fv for each task in the loop
                    fv_sc_list = []
                    for sc_task in range(params.num_sc_variables):
                        start = (sc_task+1) * fv_len
                        end   = (sc_task+2) * fv_len
                        fv_sc_list.append(fv[:, start:end])

                    # Estimation/Predict on both MTL classes
                    y_val_pt_preds_as_list.append(pt_model(fv_pt))
                    # populate the latest batch bin with predictions about sc_variables
                    for fv_sc,sc_model in zip(fv_sc_list,sc_model_list):
                        y_val_sc_preds_as_list[-1].append(sc_model(fv_sc))
                    # add the true values to the lists
                    y_val_pt_trues_as_list.append(y_val_pt)
                    for label in labels_sc_val:
                        y_val_sc_trues_as_list[-1].append(label)

                # Evaluation of all batches
                y_val_pred_sc_cat = []
                y_val_true_sc_cat = []
                # create *sc_cat lists which contain separate lists (populated with preds and true values) for each sc variable 
                for sc_variable in range(params.num_sc_variables):
                    y_val_pred_sc_cat.append([batch_bin[sc_variable] for batch_bin in y_val_sc_preds_as_list])
                    y_val_true_sc_cat.append([batch_bin[sc_variable] for batch_bin in y_val_sc_trues_as_list])
                    y_val_pred_sc_cat[-1] = torch.cat(y_val_pred_sc_cat[-1])
                    y_val_true_sc_cat[-1] = torch.cat(y_val_true_sc_cat[-1])

                y_val_pt_preds = torch.cat(y_val_pt_preds_as_list)
                y_val_pt_trues = torch.cat(y_val_pt_trues_as_list)

                # Compute val metrics
                val_accuracy_pt = compute_metrics(
                    y_val_pt_preds, y_val_pt_trues)
                val_accuracy_sc = []
                for y_val_pred_sc_task, y_val_true_sc_task in zip(y_val_pred_sc_cat, y_val_true_sc_cat):
                    val_accuracy_sc.append(compute_metrics(y_val_pred_sc_task, y_val_true_sc_task))

                # Log the evaluation
                wandb.log({'val_accuracy_pt': val_accuracy_pt,
                        **{f'val_accuracy_sc_{i}': acc for i, acc in enumerate(val_accuracy_sc)}})
                wandb.finish()
                print(testType + " val_acc CL " + str(val_accuracy_pt))
                for i,acc in enumerate(val_accuracy_sc):
                    print(testType + f" val_acc SC_{i} " + str(acc))
                saveWandbRun(run, saveFile, syncFile, params_dict)

    print("===============================evaluation done================================")
