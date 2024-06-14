import os
import sys
import torch
from Training.MTL_MI_adv import mtl_mi_train
from Evaluation.Evaluation_MTL import evaluation_mtl
from Dataset.Create_DataLoader import init_morphoMNIST_training, init_chXpert_training
from config.Load_Parameter import load_preset_parameters
from SaveWandbRuns.loadPathsToSaveRun import loadPaths
from SaveWandbRuns.initWandb import initWandb, saveWandbRun, upload_wandb
from Plots.tsne import tsne
from Dataset.Utils import visualise_distribution

# clear GPU memory
torch.cuda.empty_cache()
def main():
    # Set wandb to offline mode
    os.environ['WANDB_MODE'] = 'online'

    # Load parameter file
    if len(sys.argv) == 2:
        params_file = sys.argv[1]
    else:
        params_file = "morphomnist.yml"

    # Save parameters as dict
    params_dict = load_preset_parameters(params_file)
    training_dataset = params_dict["training_dataset"]["value"]
    trainType = params_dict["trainType"]["value"]
    saveConfoundingRatio = params_dict["confoundingRatio"]["value"]

    #  Create dataloader
    if training_dataset == "MorphoMNIST":
        trainLoader, valLoader, testLoader, testEqualLoader = init_morphoMNIST_training(params_dict, training_dataset)
        params_dict["confoundingRatio"]["value"] = [0.5] * params_dict["num_sc_variables"]["value"]
        trainEqualLoader, _, _, _ = init_morphoMNIST_training(params_dict, training_dataset)
        params_dict["confoundingRatio"]["value"] = [0.5] * params_dict["num_sc_variables"]["value"]
        testNativeLoader = ""

        visualise_distribution(trainLoader, experiment="MorphoMNIST_training_dataset")
        visualise_distribution(valLoader, experiment="MorphoMNIST_validation_dataset")
        visualise_distribution(testLoader, experiment="MorphoMNIST_test_dataset")
        visualise_distribution(testEqualLoader, experiment="MorphoMNIST_Balanced-test_dataset")

    elif training_dataset == "chXpert":
        trainLoader, valLoader, testLoader, testEqualLoader, testNativeLoader, trainEqualLoader = init_chXpert_training(params_dict)
        visualise_distribution(trainLoader, experiment="chXpert__training_dataset")
        visualise_distribution(valLoader, experiment="chXpert_validation_dataset")
        visualise_distribution(testLoader, experiment="chXpert_test_dataset")
        visualise_distribution(testEqualLoader, experiment="chXpert_Balanced-test_dataset")
        visualise_distribution(testNativeLoader, experiment="chXpert_Native-test_dataset")

    else:
        raise ValueError('No valid dataset selected.')
    params_dict["confoundingRatio"]["value"] = saveConfoundingRatio

    # Initialize and start training 
    if "MTL"  in trainType:
        # train MIMM
        syncFile=set_mtl_mi_train(trainLoader, valLoader, testLoader, testEqualLoader, testNativeLoader, trainEqualLoader, params_dict, params_file)
    else:
        raise ValueError('No valid training selected')
    
    # Upload offline runs to wandb
    upload_wandb(syncFile)

def set_mtl_mi_train(trainLoader, valLoader, testLoader, testEqualLoader,testNativeLoader, trainEqualLoader, params_dict, params_file):

    # Load paths to save runs and wandb path
    saveFile, syncFile = loadPaths(params_dict)
            
    print("====================================================train and evaluate========================================================")
    print(f"mi_batches={params_dict['mi_max_batches']['value']}, mi_lambda={params_dict['mi_lambda']['value']} ")
    # Init wandb

    config, run  = initWandb(params_dict, params_file)
    
    # Start training
    fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile_list, _ = mtl_mi_train(trainLoader, valLoader)
    torch.cuda.empty_cache()

    saveWandbRun(run, saveFile, syncFile, params_dict)
    evaluation_mtl([valLoader], [testLoader], [testEqualLoader], [testNativeLoader], fv_modelPathToFile, pt_modelPathToFile, sc_modelPathToFile_list, params_dict, params_file, saveFile, syncFile)
    torch.cuda.empty_cache()

    tsne(testEqualLoader, fv_modelPathToFile, pt_modelPathToFile,sc_modelPathToFile_list, config, datatype="testequal")
    tsne(trainEqualLoader, fv_modelPathToFile, pt_modelPathToFile,sc_modelPathToFile_list, config, datatype="trainequal")

    return syncFile

if __name__ == '__main__':
    main()
