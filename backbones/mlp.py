import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self, in_features=784, out_features=10, hidden_features=256, drop_rate=0.1):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop_rate)
        self.act_func = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, -1)
        x = self.act_func(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    


