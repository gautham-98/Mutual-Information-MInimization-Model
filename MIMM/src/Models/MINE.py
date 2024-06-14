import torch
import torch.nn as nn
import numpy as np
import config.Load_Parameter

class T(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_feature, 400, bias=False),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 400, bias=False),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 400, bias=False),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 1))

    def forward(self, x):
        return self.layers(x)


class MIComputer(nn.Module):
    def __init__(self,i):
        super().__init__()
        self.classes_PT = config.Load_Parameter.params.numberClasses
        self.classes_SC = config.Load_Parameter.params.numberClassesConfounding 
        self.feature_vector_length = config.Load_Parameter.params.feature_vector_length
        if config.Load_Parameter.params.choose_mi_loss == 0:
            self.pt_sc_feature_length = int((self.feature_vector_length / (config.Load_Parameter.params.num_sc_variables+1)) * 2)
        else:
            self.pt_sc_feature_length = int((self.feature_vector_length / (config.Load_Parameter.params.num_sc_variables+1)) * ((config.Load_Parameter.params.num_sc_variables+1)-i))
        # Create statistics network
        self.stat_network = T(self.pt_sc_feature_length)

        # Initialize the exponential moving average of e^T(x, y)
        self.register_buffer('running_exp', torch.tensor(float('nan')))

    def forward(self, x, y):
        num_samples = x.size(0)
        xy = torch.cat([x.repeat_interleave(num_samples, dim=0),
                        y.tile(num_samples, 1)], -1)
        stats = self.stat_network(xy).reshape(num_samples, num_samples)

        # Compute DV estimate (MINE)
        diag = torch.diagonal(stats).mean()
        logmeanexp = logmeanexp_offDiagonal(stats)
        with torch.no_grad():      
            self.update_running_exp(torch.exp(logmeanexp), alpha=0.1) #add mean()

        dv = diag - logmeanexp
        if config.Load_Parameter.params.useCorrectedMIgrad:
            dv_loss = diag - (torch.exp(logmeanexp)/self.running_exp) # defined seperately for corrected gradients
        else:
            dv_loss = dv
        return dv, dv_loss 

    def update_running_exp(self, meanexp, alpha=0.1):
        if torch.isnan(self.running_exp):
            self.running_exp = meanexp
        else:
            self.running_exp = alpha * self.running_exp + (1-alpha) * meanexp 


def logmeanexp_offDiagonal(x, dim=(0,1), device='cuda'):
    batch_size = x.size(0)
    offDiag = x - torch.diag(np.inf * torch.ones(batch_size).to(device))
    logsumexp = torch.logsumexp(offDiag, dim=dim)
    return logsumexp - torch.log(torch.tensor(batch_size*(batch_size-1))).to(device)


