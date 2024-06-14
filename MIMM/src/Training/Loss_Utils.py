import torch.nn as nn
import config.Load_Parameter
import torch

def crossEntropyLoss_for_MTL(y_pred, y_true):
    """ Compute CrossEntropyLoss for multi-task learning

    Args:
        y_pred (list): list of predictions, each entry contains a list with computed predictions of one batch
        y_true (list): list of corresponding ground truth labels, each entry contains a list with gt of one batch

    Returns:
        loss_average[float]: returns loss averaged over the number of multi-tasks  
    """
    loss = 0
    criterionLoss = nn.CrossEntropyLoss()
    for i in range(len(y_pred)):
        if i==0:
            loss += 1 * criterionLoss(y_pred[i], y_true[i])
        else:
            loss += config.Load_Parameter.params.sc_lambda * criterionLoss(y_pred[i], y_true[i])
    sum_of_weights = (len(y_pred)-1) * config.Load_Parameter.params.sc_lambda + 1       
    return loss/sum_of_weights

def crossEntropyLoss_for_STL(y_pred, y_true):
    """ Compute CrossEntropyLoss for a single task learning

    Args:
        y_pred (list): computed predictions of one batch
        y_true (list): list of corresponding ground truth labels

    Returns:
        loss_average[float]: returns loss 
    """

    criterionLoss = nn.CrossEntropyLoss()
    return criterionLoss(y_pred, y_true)


def miLoss_for_MTL(mi_model_list, fv_pt, fv_sc_list):
    """ Compute CrossEntropyLoss for a single task learning

    Args:
        fv_pt (list): feature vector of primary task
        fv_sc_list (list): list of feature vectors of corresponding sc_variables

    Returns:
        mi (float): returns average mi 
        mi_loss (float): returns average mi_loss
    """
    i=0
    mi = 0
    mi_loss = 0
    fv_list = fv_sc_list
    fv_list.append(fv_pt)

    while len(fv_list) != 1:
        fv_1 = fv_list.pop(0)
        for fv_2 in fv_list:
            mi_model = mi_model_list[i]
            mi_current, mi_loss_current = mi_model(fv_1, fv_2)
            mi = mi+mi_current
            mi_loss = mi_loss + mi_loss_current
            i+=1
    divide_by = ((config.Load_Parameter.params.num_sc_variables+1)*config.Load_Parameter.params.num_sc_variables)/2
    mi = mi/divide_by
    mi_loss = mi_loss/divide_by
    return mi, mi_loss

def miLoss_eff_for_MTL(mi_model_list, fv_pt, fv_sc_list):
    """ Compute CrossEntropyLoss for a Multi task learning

    Args:
        fv_pt (list): feature vector of primary task
        fv_sc_list (list): list of feature vectors of corresponding sc_variables

    Returns:
        mi (float): returns average mi 
        mi_loss (float): returns average mi_loss
    """
    mi = 0
    mi_loss = 0
    fv_list=[]
    fv_list.append(fv_pt)
    fv_list.extend(fv_sc_list)
    for i in range(config.Load_Parameter.params.num_sc_variables):
        current_fv = fv_list[i]
        current_joint = torch.cat(fv_list[i+1:], dim=1)
        mi_model = mi_model_list[i]
        mi_current, mi_loss_current = mi_model(current_fv, current_joint)
        mi = mi+mi_current
        mi_loss = mi_loss + mi_loss_current

    mi = (mi)/(config.Load_Parameter.params.num_sc_variables)
    mi_loss = (mi_loss)/(config.Load_Parameter.params.num_sc_variables)
    return mi, mi_loss