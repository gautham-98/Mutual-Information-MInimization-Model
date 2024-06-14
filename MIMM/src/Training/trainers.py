import os
import sys
import numpy as np

sys.path.insert(0, os.getcwd() + '/MIMM/src')

import wandb
import torch

import config.Load_Parameter
from Training.Loss_Utils import crossEntropyLoss_for_MTL, miLoss_for_MTL, miLoss_eff_for_MTL
from Training.Metrics_Utils import compute_metrics

class FETrainer():

    def __init__(self, fv_model, pt_model, sc_model_list, mi_model_list, trainLoader, optimizer):
        self.optimizer = optimizer
        self.fv_model = fv_model
        self.pt_model = pt_model
        self.sc_model_list = sc_model_list
        self.mi_model_list = mi_model_list
        self.trainLoader = trainLoader
        self.num_batches = len(trainLoader)
        self.params = config.Load_Parameter.params

    def __call__(self, epoch, max_batches):
        self.train(epoch, max_batches)
        return
    
    def train(self, epoch, max_batches):

        for batch, (X_train, y_train_pt, *labels_sc_train) in enumerate(self.trainLoader):
            self.optimizer.zero_grad()
            # Join labels to list
            y_train = [y_train_pt.to('cuda'), *[label.to('cuda') for label in labels_sc_train]]
            # forward
            y_pred,fv_pt, fv_sc_list = self.forward(X_train)
            # loss
            loss_grad, pt_sc_loss, loss, mi, mi_loss = self.loss_function(y_train, y_pred, fv_pt, fv_sc_list)
            # metrics
            accuracy_pt, accuracy_sc = self.get_metrics(y_train, y_pred)

            # Log loss and metrics
            if self.params.trainType == 'MTL_MI':
                wandb.log({'epoch':epoch, 'batch': batch, 'pt_sc_loss': pt_sc_loss.item(), 'loss': loss.item(),
                            'accuracy_pt': accuracy_pt, 
                            **{f'accuracy_sc_{i}': acc for i, acc in enumerate(accuracy_sc)},
                            'mi': mi.item(), 'mi_loss': mi_loss.item()})
            if self.params.trainType == 'MTL_no_MI':
                wandb.log({'epoch':epoch, 'batch': batch, 'pt_sc_loss': pt_sc_loss.item(), 'loss': loss.item(),
                            'accuracy_pt': accuracy_pt, 
                            **{f'accuracy_sc_{i}': acc for i, acc in enumerate(accuracy_sc)},
                            })
            
            # Check for divergence
            if torch.isnan(loss) or torch.isinf(loss):
                wandb.run.summary['diverged'] = True
                wandb.run.finish()
                raise ValueError('Training loss diverged')
            # backward
            self.backward(loss_grad, mi, mi_loss)
            # step
            self.optimizer.step()

            #check if max_batches
            if (max_batches!=-1 and batch==max_batches):
                break
       
    def loss_function(self, y_train, y_pred, fv_pt, fv_sc_list):
        pt_sc_loss = crossEntropyLoss_for_MTL(y_pred, y_train)

        if self.params.trainType == 'MTL_MI':
            if self.params.choose_mi_loss==0:
                mi, mi_loss = miLoss_for_MTL(self.mi_model_list,fv_pt, fv_sc_list)
            else:
                mi, mi_loss = miLoss_eff_for_MTL(self.mi_model_list,fv_pt, fv_sc_list)
            if self.params.adaScale:
                with torch.no_grad():
                        mi_scale = self.grad_scale(pt_sc_loss, mi_loss)
            else:
                mi_scale = 1
            loss_grad = pt_sc_loss + self.params.mi_lambda * mi_scale * mi_loss
            loss =  pt_sc_loss + self.params.mi_lambda * mi

        if self.params.trainType == 'MTL_no_MI':
            mi = 0
            mi_loss = 0
            loss = pt_sc_loss
            loss_grad = loss

        return loss_grad,pt_sc_loss, loss, mi, mi_loss
    
    def get_metrics(self, y_train, y_pred):
        y_train_pt = y_train[0]
        y_pred_pt = y_pred[0]
        y_train_sc = y_train[1:]
        y_pred_sc = y_pred[1:]
        accuracy_pt = compute_metrics(y_pred_pt, y_train_pt)
        accuracy_sc = []
        for pred, train in zip(y_pred_sc, y_train_sc):
            accuracy_sc.append(compute_metrics(pred, train))
        return accuracy_pt, accuracy_sc
    
    def forward(self, X_train):
         # Forward to get feature vector (fv)
        fv = self.fv_model(X_train)

        # Split fv (feature partioning)
        fv_len = int(fv.size(1) / (self.params.num_sc_variables+1))
        fv_pt = fv[:, :fv_len]
        fv_sc_list = []
        for sc_task in range(self.params.num_sc_variables):
            start = (sc_task+1) * fv_len
            end   = (sc_task+2) * fv_len
            fv_sc_list.append(fv[:, start:end])

        # Forward splitted part to model_pt and model_sc
        y_pred = []
        out_pt = self.pt_model(fv_pt)
        y_pred.append(out_pt)
        for sc_model,fv_sc in zip(self.sc_model_list, fv_sc_list):
            out_sc = sc_model(fv_sc)
            y_pred.append(out_sc)

        return y_pred, fv_pt, fv_sc_list
    
    def backward(self, loss, mi, mi_loss):
        self.fv_model.zero_grad()
        self.pt_model.zero_grad()
        for sc_model in self.sc_model_list:
            sc_model.zero_grad()
        if self.params.trainType != 'MTL_no_MI':
            for mi_model in self.mi_model_list:    
                mi_model.zero_grad()
        loss.backward()
    
    def grad_scale(self, loss, mi_loss):
        fe_params = [p for p in self.fv_model.parameters()]
        class_params = []
        class_params.extend(fe_params)
        class_params.extend([p for p in self.pt_model.parameters()])
        for sc_model in self.sc_model_list:
            sc_params_to_add = [p for p in sc_model.parameters()]
            class_params.extend(sc_params_to_add)
        mi_grads = torch.autograd.grad(mi_loss, fe_params, retain_graph=True, allow_unused=True)
        class_grads = torch.autograd.grad(loss, class_params, retain_graph=True, allow_unused=True)
        mi_grads = torch.tensor([torch.norm(grad) for grad in [*mi_grads] if grad is not None])
        class_grads = torch.tensor([torch.norm(grad) for grad in [*class_grads] if grad is not None])
        scale_mi = torch.minimum(torch.norm(class_grads), torch.norm(mi_grads))/torch.norm(mi_grads)
        return scale_mi
    
class MITrainer():

    def __init__(self, fv_model, mi_model_list, trainLoader, optimizer):
        self.optimizer = optimizer
        self.fv_model = fv_model
        self.mi_model_list = mi_model_list
        self.trainLoader = trainLoader
        self.params = config.Load_Parameter.params
        self.num_batches = len(trainLoader)

    def __call__(self, epoch, repeat, max_batches):
        self.repeat = repeat
        for _ in range(self.repeat):
            self.train(epoch, max_batches)
        return
    
    def train(self, epoch, max_batches):

        for batch, (X_train, _, *labels_sc_train) in enumerate(self.trainLoader):
            self.optimizer.zero_grad()

            # Join labels to list
            fv_pt, fv_sc_list = self.forward(X_train)

            mi, mi_loss = self.loss_function(fv_pt, fv_sc_list)

            # Log loss and metrics
            wandb.log({'epoch':epoch, 'batch': batch, 'mi': mi.item(), 'mi_loss': mi_loss.item()})
            # Check for divergence
            if torch.isnan(mi) or torch.isinf(mi):
                wandb.run.summary['diverged'] = True
                wandb.run.finish()
                raise ValueError('Training loss diverged')
            
            self.backward(-mi_loss)
            
            if (max_batches!=-1 and batch==max_batches):
                break

    def loss_function(self, fv_pt, fv_sc_list):
        fv_sc_list_detached = [tensor.detach() for tensor in fv_sc_list]
        fv_pt_detached = fv_pt.detach()
        if self.params.choose_mi_loss==0:
            mi, mi_loss = miLoss_for_MTL(self.mi_model_list, fv_pt_detached, fv_sc_list_detached)
        else:
            mi, mi_loss = miLoss_eff_for_MTL(self.mi_model_list, fv_pt_detached, fv_sc_list_detached)
        return mi, mi_loss
  
    def forward(self, X_train):
         # Forward to get feature vector (fv)
        fv = self.fv_model(X_train)
        
        # Split fv (feature partioning)
        fv_len = int(fv.size(1) / (self.params.num_sc_variables+1))
        fv_pt = fv[:, :fv_len]
        fv_sc_list = []
        for sc_task in range(self.params.num_sc_variables):
            start = (sc_task+1) * fv_len
            end   = (sc_task+2) * fv_len
            fv_sc_list.append(fv[:, start:end])
        return fv_pt, fv_sc_list

    def backward(self, loss):
        for mi_model in self.mi_model_list:    
            mi_model.zero_grad()
        loss.backward()
        self.optimizer.step()


class Validate():
    def __init__(self,fv_model, pt_model, sc_model_list, mi_model_list, valLoader):
        self.fv_model = fv_model
        self.pt_model = pt_model
        self.sc_model_list = sc_model_list
        self.mi_model_list = mi_model_list
        self.valLoader = valLoader
        self.params = config.Load_Parameter.params
 
    def __call__(self, epoch):
        self.y_val_pred_pt = []
        self.y_val_true_pt = []
        self.y_val_true_sc = []
        self.y_val_pred_sc = []
        self.val_mis = 0
        self.val_mi_losses = 0
        return self.validator(epoch)

    def validator(self, epoch):
        for batch, (X_val, y_val_pt, *labels_sc_val) in enumerate(self.valLoader):
            self.y_val_pred_sc_cat = []
            self.y_val_true_sc_cat = []
            # add batch bins for sc classes
            self.y_val_true_sc.append([])
            self.y_val_pred_sc.append([])
            fv_val_pt, fv_val_sc_list = self.forward(X_val)
            self.push_true_and_preds(fv_val_pt, fv_val_sc_list, y_val_pt, labels_sc_val) 

            val_mi, val_mi_loss = self.loss_function(fv_val_pt, fv_val_sc_list)
            self.val_mis += val_mi
            self.val_mi_losses += val_mi_loss
        y_val_mi_mean, y_val_mi_mean_loss, val_pt_sc_loss, val_loss, val_accuracy_pt, val_accuracy_sc, mean_val_accuracy = self.get_metrics()
        # Log val loss and metrics
        wandb.log({
            'epoch': epoch, 'val_loss': val_loss.item(), 'val_pt_sc_loss': val_pt_sc_loss,
            'val_mi': y_val_mi_mean.item(), 'val_mi_loss': y_val_mi_mean_loss.item(),
            'val_accuracy_pt': val_accuracy_pt, 
            **{f'val_accuracy_sc_{i}': acc for i, acc in enumerate(val_accuracy_sc)}})
        
        print(f'Validation epoch: {epoch}, Validation loss:  {val_loss.item():.3f} val_acc_pt: {val_accuracy_pt:.3f}' 
              + ','.join([f'val_accuracy_sc_{i}: {acc:.3f}' for i, acc in enumerate(val_accuracy_sc)]) + f' val_MI: {y_val_mi_mean.item():.3f}'
              )
        
        return y_val_mi_mean, y_val_mi_mean_loss, val_pt_sc_loss, val_loss, val_accuracy_pt, val_accuracy_sc, mean_val_accuracy

            
    def push_true_and_preds(self, fv_val_pt, fv_val_sc_list, y_val_pt, labels_sc_val):
        self.y_val_pred_pt.append(self.pt_model(fv_val_pt)) # [y_val_pred_pt]
        for fv_val_sc, sc_model in zip(fv_val_sc_list, self.sc_model_list):
            self.y_val_pred_sc[-1].append(sc_model(fv_val_sc)) # [[y_val_pred_sc, , , ], [...], [...]]
        self.y_val_true_pt.append(y_val_pt.to('cuda'))
        for label in labels_sc_val:
            self.y_val_true_sc[-1].append(label.to('cuda'))
    
    def loss_function(self, fv_pt, fv_sc_list):
        with torch.no_grad():
            fv_sc_list_detached = [tensor.detach() for tensor in fv_sc_list]
            fv_pt_detached = fv_pt.detach()
            if self.params.choose_mi_loss==0:
                mi, mi_loss = miLoss_for_MTL(self.mi_model_list, fv_pt_detached, fv_sc_list_detached)
            else:
                mi, mi_loss = miLoss_eff_for_MTL(self.mi_model_list, fv_pt_detached, fv_sc_list_detached)
            return mi, mi_loss
    
    def get_metrics(self):
        for sc_variable in range(self.params.num_sc_variables):
            self.y_val_pred_sc_cat.append([batch_bin[sc_variable] for batch_bin in self.y_val_pred_sc])
            self.y_val_true_sc_cat.append([batch_bin[sc_variable] for batch_bin in self.y_val_true_sc])
            self.y_val_pred_sc_cat[-1] = torch.cat(self.y_val_pred_sc_cat[-1])
            self.y_val_true_sc_cat[-1] = torch.cat(self.y_val_true_sc_cat[-1])

        y_val_pred_pt = torch.cat(self.y_val_pred_pt)
        y_val_true_pt = torch.cat(self.y_val_true_pt)

        # Compute average MI and MI_loss
        len_val = len(self.valLoader.dataset.dataset)

        y_val_mi_mean = (self.val_mis * self.valLoader.batch_size) / len_val
        y_val_mi_mean_loss = (self.val_mi_losses * self.valLoader.batch_size) / len_val

        # Compute val loss
        y_val_preds = [y_val_pred_pt, *self.y_val_pred_sc_cat] #[tensor, *[tensor, tensor] ]
        y_val_trues = [y_val_true_pt, *self.y_val_true_sc_cat]
        val_pt_sc_loss = crossEntropyLoss_for_MTL(y_val_preds, y_val_trues)
        val_loss = val_pt_sc_loss + self.params.mi_lambda * y_val_mi_mean

        # Compute val metrics of PT, SC
        val_accuracy_pt = compute_metrics(y_val_pred_pt, y_val_true_pt)
        val_accuracy_sc = []
        for y_val_pred_sc_task, y_val_true_sc_task in zip(self.y_val_pred_sc_cat, self.y_val_true_sc_cat):
            val_accuracy_sc.append(compute_metrics(y_val_pred_sc_task, y_val_true_sc_task))
        mean_val_accuracy = np.mean(val_accuracy_sc)
        return y_val_mi_mean, y_val_mi_mean_loss, val_pt_sc_loss, val_loss, val_accuracy_pt, val_accuracy_sc, mean_val_accuracy
            
    def forward(self, X_val):
        # Forward to get feature vector (fv)
        with torch.no_grad():    
            fv = self.fv_model(X_val)

            # Split fv (feature partioning)
            fv_len = int(fv.size(1) / (self.params.num_sc_variables+1))
            fv_pt = fv[:, :fv_len]
            fv_sc_list = []
            for sc_task in range(self.params.num_sc_variables):
                start = (sc_task+1) * fv_len
                end   = (sc_task+2) * fv_len
                fv_sc_list.append(fv[:, start:end])
            return fv_pt, fv_sc_list
        