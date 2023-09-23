import torch,torch_geometric,os,sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
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
        return x


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

class Shadow_MIA_mlp():
    """
    Running MIA binary mlp classifier
    """
    def __init__(self,shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y, ss_dict) -> None:
        self.model = mia_mlpclassifier(ss_dict)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.num_epochs = ss_dict['num_epochs']
        self.train(shadow_train_x,shadow_train_y)
        self.evaluate(shadow_test_x,shadow_test_y)

    def train(self,shadow_train_x,shadow_train_y):
        print("\nTraining MIA classifier:\n")
        for epoch in range(self.num_epochs):
            # 前向传播
            outputs = self.model(shadow_train_x)
            loss = self.criterion(outputs, shadow_train_y)

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 每训练一轮打印一次损失值
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}\n")

    def evaluate(self,shadow_test_x,shadow_test_y):
        print("\nTraining MIA classifier done. Begin MIA attacks:\n")
        self.model.eval()
        with torch.no_grad():
            shadow_pred = self.model(shadow_test_x).detach().cpu().numpy().astype(float)
        shadow_res = (shadow_pred >= .5).astype(int)
        shadow_test_y = shadow_test_y.detach().cpu().numpy().astype(int)
        print("\nMLP MIA attacks:\n")
        output_shadowres(shadow_test_y, shadow_res)

class Shadow_MIA_svm():
    """
    mia blackbox shadow classifier in svm
    """
    def __init__(self, shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y, ss_dict, *args, **kwargs) -> None:
        super(Shadow_MIA_svm, self).__init__()
        self.model = SVC(kernel=ss_dict['kernel'],random_state=ss_dict['random_state'],verbose=True)
        self.train(shadow_train_x,shadow_train_y)
        self.evaluate(shadow_test_x,shadow_test_y)

    def train(self,shadow_train_x,shadow_train_y):
        self.model.fit(shadow_train_x,shadow_train_y)
        
    def evaluate(self,shadow_test_x,shadow_test_y):
        shadow_res = self.model.predict(shadow_test_x)
        output_shadowres(shadow_test_y,shadow_res)  

class Shadow_MIA_ranfor():
    """
    mia blackbox shadow classifier in randomforest
    """
    def __init__(self, shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y, ss_dict, *args, **kwargs):
        super(Shadow_MIA_ranfor, self).__init__()
        self.random_state = ss_dict['random_state']
        self.n_estimators = ss_dict['n_estimators']
        self.criterion = ss_dict['criterion']#'gini', 'entropy', 'log_loss'
        self.model = RandomForestClassifier(n_estimators=self.n_estimators,random_state=self.random_state,verbose=1)
        
        self.train(shadow_train_x,shadow_train_y)
        self.evaluate(shadow_test_x,shadow_test_y)

    def train(self, shadow_train_x,shadow_train_y):
        self.model.fit(shadow_train_x,shadow_train_y)

    def evaluate(self,shadow_test_x,shadow_test_y):
        shadow_res = self.model.predict(shadow_test_x)
        output_shadowres(shadow_test_y,shadow_res)
                       
    

def output_shadowres(shadow_test_y,shadow_res):
    shadow_target = len(shadow_test_y)
    shadow_target_in = np.count_nonzero(shadow_test_y)
    shadow_target_out = shadow_target - shadow_target_in
    shadow_hit = np.count_nonzero(shadow_test_y == shadow_res)
    shadow_hit_in = np.count_nonzero((shadow_test_y == 1) & (shadow_res == 1))
    shadow_hit_out = np.count_nonzero((shadow_test_y == 0) & (shadow_res == 0))
    print("Metrics for MIA:")
    print(metrics.classification_report(shadow_test_y, shadow_res, labels=range(2)))
    print("\nAll targets for MIA:",shadow_target)
    print(f"\nWith {shadow_target_in} member targets and {shadow_target_out} non-mamber targets.\n")
    print(f"Members hit:{shadow_hit_in}\nNon members hit:{shadow_hit_out}\nTotal hit:{shadow_hit} with ACC {shadow_hit/shadow_target}")
    print('\nROC_AUC score is:')
    print(metrics.roc_auc_score(shadow_test_y, shadow_res))