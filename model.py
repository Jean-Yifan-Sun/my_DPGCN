import torch,torch_geometric,os,sys
import pandas as pd
import numpy as np

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class two_layer_GCN(torch.nn.Module):
    """
    basic 2 layer GCN module
    """
    def __init__(self, ss:dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k_layer = ss["k_layer"]
        self.dropout = ss["dropout"]
        self.optimizer = ss["optimizer"]
        self.hidden_channels = ss["chanels"]
        self.num_features = ss["num_features"]
        self.num_classes = ss["num_classes"]
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        if ss['activation'] == 'relu':
            self.activation = F.relu
        elif ss['activation'] == 'tanh':
            self.activation = F.tanh
        elif ss['activation'] == 'selu':
            self.activation = F.selu
        else:
            self.activation = F.elu
        
        self.conv1 = GCNConv(in_channels=self.num_features,out_channels=self.hidden_channels)
        self.conv2 = GCNConv(in_channels=self.hidden_channels,out_channels=self.num_classes)

    def forward(self,data):
        x = data.x.float()
        edge_index = data.edge_index
        x = self.conv1(x,edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout)
        x = self.conv2(x,edge_index)
        return F.log_softmax(x, dim=1)


class three_layer_GCN(torch.nn.Module):
    """
    basic 3 layer GCN module
    """
    def __init__(self, ss:dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k_layer = ss["k_layer"]
        self.dropout = ss["dropout"]
        self.optimizer = ss["optimizer"]
        self.hidden_channels = ss["chanels"]
        self.num_features = ss["num_features"]
        self.num_classes = ss["num_classes"]
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        
        if ss['activation'] == 'relu':
            self.activation = F.relu
        elif ss['activation'] == 'tanh':
            self.activation = F.tanh
        elif ss['activation'] == 'selu':
            self.activation = F.selu
        else:
            self.activation = F.elu
        
        self.conv1 = GCNConv(in_channels=self.num_features,out_channels=self.hidden_channels[0])
        self.conv2 = GCNConv(in_channels=self.hidden_channels[0],out_channels=self.hidden_channels[1])
        self.conv3 = GCNConv(in_channels=self.hidden_channels[1],out_channels=self.num_classes)

    def forward(self,data):
        x = data.x.float()
        edge_index = data.edge_index
        x = self.conv1(x,edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout)
        x = self.conv2(x,edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout)
        x = self.conv3(x,edge_index)
        return F.log_softmax(x, dim=1)


class one_layer_GCN(torch.nn.Module):
    """
    basic 1 layer GCN module
    """
    def __init__(self, ss:dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.dropout = ss["dropout"]
        self.optimizer = ss["optimizer"]
        self.num_features = ss["num_features"]
        self.num_classes = ss["num_classes"]

        if ss['activation'] == 'relu':
            self.activation = F.relu
        elif ss['activation'] == 'tanh':
            self.activation = F.tanh
        elif ss['activation'] == 'selu':
            self.activation = F.selu
        else:
            self.activation = F.elu

        self.conv = GCNConv(self.num_features,self.num_classes)

    def forward(self,data):
        x = data.x.float()
        edge_index = data.edge_index
        x = self.conv(x,edge_index)
        return F.log_softmax(x, dim=1)
    

class mia_mlpclassifier(torch.nn.Module):
    """
    mia blackbox shadow classifier in mlp
    """
    def __init__(self, ss:dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.dropout = ss["dropout"]
        self.num_features = ss["num_features"]
        self.num_classes = ss["num_classes"]
        self.hidden_channels = ss["chanels"]
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')
        self.activation = F.relu

        self.layer1 = torch.nn.Linear(in_features=self.num_features, out_features=self.hidden_channels,device=self.device)
        self.layer2 = torch.nn.Linear(in_features=self.hidden_channels,out_features=self.num_classes,device=self.device)

    def forward(self,data):
        x = self.layer1(data)
        x = F.dropout(x,p=self.dropout)
        x = self.activation(x)
        x = self.layer2(x)
        return F.sigmoid(x).squeeze(-1)   