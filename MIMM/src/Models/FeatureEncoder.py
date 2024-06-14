import torch.nn as nn
import torch.nn.functional as F
import config.Load_Parameter
import torch


class FeatureEncoderNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_vector_length = config.Load_Parameter.params.feature_vector_length
        self.fc = nn.Linear(256, 256)
        self.fc_pt_sc = nn.Linear(256, self.feature_vector_length)

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 3, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, X):
        X = self.features(X)
        X = X.view(-1, X.shape[1]*X.shape[2]*X.shape[3])
        X = F.relu(self.fc(X))
        X = self.fc_pt_sc(X)
        return X
    
class MedicalFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_vector_length = config.Load_Parameter.params.feature_vector_length

        if config.Load_Parameter.params.selectFeatureEncoder == 0:
            self.features = nn.Sequential(
                nn.Conv2d(1, 3, 3, 1),
                nn.ReLU(inplace=True),

                nn.Conv2d(3, 6, 3, 1),
                nn.ReLU(inplace=True),

                nn.Conv2d(6, 8, 3, 1),
                nn.ReLU(inplace=True),

                nn.Conv2d(8, 8, 3, 1),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(3, 3),
                nn.BatchNorm2d(8),

                nn.Conv2d(8, 16, 3, 1),
                nn.ReLU(inplace=True),

                nn.Conv2d(16, 16, 3, 1),
                nn.ReLU(inplace=True),

                nn.Conv2d(16, 16, 3, 1),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(3, 3),
                nn.BatchNorm2d(16),

                nn.Conv2d(16, 32, 3, 1),
                nn.ReLU(inplace=True),

                nn.Conv2d(32, 32, 3, 1),
                nn.ReLU(inplace=True),

                nn.Conv2d(32, 32, 3, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 3),

                nn.Conv2d(32, 64, 3, 1),
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 64, 2, 1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5,5)
            )

            self.fc = nn.Sequential(

                nn.Linear(64, 32),
                nn.ReLU(inplace=True),

                nn.Linear(32, 12),
                nn.ReLU(inplace=True),

                nn.Linear(12, self.feature_vector_length),
            )

        elif config.Load_Parameter.params.selectFeatureEncoder == 1:
            self.densenet121 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
            num_channels = self.densenet121.classifier.in_features
            self.densenet121.classifier = nn.Linear(num_channels, self.feature_vector_length)

    def forward(self, X):
        if config.Load_Parameter.params.selectFeatureEncoder == 0:
            X = self.features(X)
            X = X.view(-1, X.shape[1]*X.shape[2]*X.shape[3])
            X = self.fc(X)
        elif config.Load_Parameter.params.selectFeatureEncoder == 1:
            X = self.densenet121(X)
        return X
