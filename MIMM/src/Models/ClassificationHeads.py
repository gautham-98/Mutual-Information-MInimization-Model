import torch.nn as nn
import torch.nn.functional as F
import config.Load_Parameter


class PTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_vector_length = config.Load_Parameter.params.feature_vector_length
        self.classes_PT = config.Load_Parameter.params.numberClasses
        self.classes_SC = config.Load_Parameter.params.numberClassesConfounding * config.Load_Parameter.params.num_sc_variables
        self.ratio = config.Load_Parameter.params.num_sc_variables + 1

        self.layers = nn.Sequential(
            nn.Linear(int(self.feature_vector_length / self.ratio), int(self.feature_vector_length / 2*self.ratio)),
            nn.Linear(int(self.feature_vector_length / 2*self.ratio), self.classes_PT)
        )

    def forward(self, X):
        if self.feature_vector_length != self.classes_PT + self.classes_SC:
            X = self.layers(X)
            return F.log_softmax(X, dim=1)
        else:
            return F.log_softmax(X, dim=1)


class SCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_vector_length = config.Load_Parameter.params.feature_vector_length
        self.classes_PT = config.Load_Parameter.params.numberClasses
        self.classes_SC = config.Load_Parameter.params.numberClassesConfounding * config.Load_Parameter.params.num_sc_variables
        self.ratio = config.Load_Parameter.params.num_sc_variables + 1
        self.layers = nn.Sequential(
            nn.Linear(int(self.feature_vector_length / self.ratio), int(self.feature_vector_length / 2*self.ratio)),
            nn.Linear(int(self.feature_vector_length / 2*self.ratio), self.classes_SC)
        )

    def forward(self, X):
        if self.feature_vector_length != self.classes_PT + self.classes_SC:
            X = self.layers(X)
            return F.log_softmax(X, dim=1)
        else:
            return F.log_softmax(X, dim=1)

