import torch,torch_geometric,os,sys,random
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import metrics
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from utils import *
from matplotlib import pyplot as plt
from torch_geometric.utils import one_hot
from tqdm import tqdm,trange 


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
    return metrics.roc_auc_score(shadow_test_y, shadow_res)

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
        return x

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
        return x
    
class Shadow_MIA_svm():
    """
    mia blackbox shadow classifier in svm
    """
    def __init__(self, shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y, ss_dict, *args, **kwargs) -> None:
        super(Shadow_MIA_svm, self).__init__()
        self.model = SVC(kernel=ss_dict['kernel'],random_state=ss_dict['random_state'],verbose=True)
        self.train(shadow_train_x,shadow_train_y)
        # self.evaluate(shadow_test_x,shadow_test_y)

    def train(self,shadow_train_x,shadow_train_y):
        self.model.fit(shadow_train_x,shadow_train_y)
        
    def evaluate(self,shadow_test_x,shadow_test_y):
        shadow_res = self.model.predict(shadow_test_x)
        score = output_shadowres(shadow_test_y,shadow_res)
        return score

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
        # self.evaluate(shadow_test_x,shadow_test_y)

    def train(self, shadow_train_x,shadow_train_y):
        self.model.fit(shadow_train_x,shadow_train_y)

    def evaluate(self,shadow_test_x,shadow_test_y):
        shadow_res = self.model.predict(shadow_test_x)
        score = output_shadowres(shadow_test_y,shadow_res)
        return score
                       
class Shadow_MIA_mlp():
    """
    mia blackbox shadow classifier in mlp
    """
    def __init__(self, shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y, ss_dict, *args, **kwargs) -> None:
        super(Shadow_MIA_mlp, self).__init__()
        self.model = MLPClassifier(random_state=ss_dict['random_state'],verbose=False,solver='adam', max_iter=ss_dict['max_iter'],learning_rate='adaptive')
        self.train(shadow_train_x,shadow_train_y)
        # self.evaluate(shadow_test_x,shadow_test_y)

    def train(self,shadow_train_x,shadow_train_y):
        self.model.fit(shadow_train_x,shadow_train_y)
        
    def evaluate(self,shadow_test_x,shadow_test_y):
        shadow_res = self.model.predict(shadow_test_x)
        score = output_shadowres(shadow_test_y,shadow_res)
        return score 

class vanilla_GCN_node():
    def __init__(self,ss,data,shadow) -> None:
        self.epochs = ss.args.epochs
        self.dataset = ss.args.dataset
        self.hidden_dim = ss.args.hidden_dim
        self.learning_rate = ss.args.learning_rate/2 if shadow == 'shadow' else ss.args.learning_rate
        self.weight_decay = ss.args.weight_decay
        self.amsgrad = ss.args.amsgrad
        self.verbose = ss.args.verbose
        self.activation = ss.args.activation
        self.dropout = ss.args.dropout
        self.momentum = ss.args.momentum
        self.early_stopping = ss.args.early_stopping
        self.patience = ss.args.patience
        self.optim_type = ss.args.optim_type
        self.parallel = ss.args.parallel
        self.k_layers = ss.args.k_layers
        self.model_name = ss.model_name
        self.root_dir = ss.root_dir
        self.log_dir = ss.log_dir
        self.time_dir = ss.time_dir
        self.learning_rate_decay = False
        self.scheduler = None
        self.max_epochs_lr_decay = 200
        self.scheduler_gamma = 1

        self.train_accs = []
        self.train_losses = []
        self.train_f1s = []
        self.valid_accs = []
        self.valid_losses = []
        self.valid_f1s = []
        self.test_loss = None
        self.test_acc = None
        self.test_f1 = None
        self.random_baseline = True
        self.majority_baseline = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.shadow = 'Shadow' if shadow == 'shadow' else 'Vanilla'
        self.learning_rate 
        self.seed = ss.args.seed
        self.total_params = 0
        self.trainable_params = 0
        data = data.to(self.device)
        self.data = data
        self.num_nodes = data.x.shape[0]
        self.num_edges = data.edge_index.shape[1]
        self._init_model()

    
    def _init_model(self):

        self.num_classes = len(torch.unique(self.data.y))
        print("Using {}".format(self.device))
        
        ss_dict = {
            "k_layer":self.k_layers,
            "chanels":self.hidden_dim,
            "dropout":self.dropout,
            "activation":self.activation,
            "optimizer":self.optim_type,
            "num_features":self.data.num_node_features,
            "num_classes":self.num_classes
        }
        
        if self.k_layers == 2:
            model = two_layer_GCN(ss_dict).to(self.device)
        elif self.k_layers == 3:
            model = three_layer_GCN(ss_dict).to(self.device)
        else:
            model = one_layer_GCN(ss_dict).to(self.device)
        
        total_params = 0
        for param in list(model.parameters()):
            nn = 1
            for sp in list(param.size()):
                nn = nn * sp
            total_params += nn
        self.total_params = total_params
        print("Total parameters", self.total_params)

        model_params = filter(lambda param: param.requires_grad,
                              model.parameters())
        trainable_params = sum([np.prod(param.size())
                                for param in model_params])
        self.trainable_params = trainable_params
        print("Trainable parameters", self.trainable_params)

        
        if self.optim_type == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(),
                                                lr=self.learning_rate,
                                                weight_decay=self.weight_decay)

        elif self.optim_type == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(),
                                                lr=self.learning_rate,
                                                weight_decay=self.weight_decay)
        else:
            raise Exception(f"{self.optim_type} not a valid optimizer (adam or sgd).")

        if self.learning_rate_decay:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1,
                                                             gamma=self.scheduler_gamma)

        self.loss = torch.nn.CrossEntropyLoss()

        self.model = model

    
    def train(self):
        model = self.model
        optimizer = self.optimizer
        early_stopping = EarlyStopping(self.patience)

        model.train()

        print('Training...\n')
        train_bar = tqdm(range(self.epochs),position=0,leave=True,colour='#3399FF')
        val_bar = tqdm(range(self.epochs),position=1,leave=True,colour='#33CC00')
        
        for epoch,_ in zip(train_bar,val_bar):
            optimizer.zero_grad()
            pred_prob_node = model(self.data)
            loss = self.loss(pred_prob_node[self.data.train_mask],
                                self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
            
            if self.scheduler != None and epoch < self.max_epochs_lr_decay:
                print("Old LR:", self.optimizer.param_groups[0]['lr'])
                self.scheduler.step()
                print("New LR:", self.optimizer.param_groups[0]['lr'])

            train_acc, prec, rec, train_f1 = self.calculate_accuracy(
                    pred_prob_node[self.data.train_mask],
                    self.data.y[self.data.train_mask])
            train_loss = loss.item()
            
            self.log(train_bar,epoch, train_loss, train_acc, prec, rec, train_f1,
                     split=f'{self.shadow} train')
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.train_f1s.append(train_f1)
            val_loss = self.evaluate_on_valid(model, epoch, val_bar)
            
            self.plot_learning_curve()

            if self.early_stopping:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    break

        if self.early_stopping:
            return -early_stopping.best_score
        else:
            return val_loss


    def calculate_accuracy(self, pred, target, rand_maj_baseline=False):

        if rand_maj_baseline:
            pred_node = pred
        else:
            _, pred_node = pred.max(dim=1)
        acc = float(pred_node.eq(target).sum().item()) / (len(pred_node))

        results = metrics.precision_recall_fscore_support(
                target.cpu().numpy(), pred_node.cpu().numpy(), average='micro')
        prec, rec, f1, _ = results

        return acc, prec, rec, f1

    def log(self, train_bar, epoch, loss, accuracy, prec, rec, f1, split='val'):
        train_bar.set_description(f"{split} Epoch:{epoch}/{self.epochs} Loss:{loss:.4f} F1:{f1:.4f} ACC:{accuracy:.4f} " )
            # train_bar.set_postfix(Loss=f"{loss:.4f}" ,F1=f"{f1:.4f}" ,ACC=f"{accuracy:.4f}")
            # print("Epoch {} ({})\tLoss: {:.4f}\tA: {:.4f}\tP: {:.4f}\t"
            #       "R: {:.4f}\tF1: {:.4f}".format(epoch, split, loss, accuracy,
            #                                      prec, rec, f1), flush=True)


    def plot_learning_curve(self):
        '''
        Result png figures are saved in the log directory.
        '''
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        privacy = 'None'
        fig.suptitle('{} Model Learning Curve ({}, % data {}, epsilon {})'.format(
            self.shadow,self.dataset, 1, privacy))
        epochs = list(range(len(self.train_losses)))
        ax1.plot(epochs, self.train_losses, 'o-', markersize=2, color='b',
                label='Train')
        ax1.plot(epochs, self.valid_losses, 'o-', markersize=2, color='c',
                label='Validation')
        ax1.set(ylabel='Loss')

        ax2.plot(epochs, self.train_accs, 'o-', markersize=2, color='b',
                label='Train')
        ax2.plot(epochs, self.valid_accs, 'o-', markersize=2, color='c',
                label='Validation')
        ax2.set(xlabel='Epoch', ylabel='Accuracy')
        ax1.legend()
        plt.savefig(os.path.join(self.time_dir, f'learning_curve_{self.shadow}.png'))
        plt.close()


    def evaluate_on_valid(self, model, epoch, train_bar):

        model.eval()

        pred_prob_node = model(self.data)
        loss = self.loss(pred_prob_node[self.data.val_mask],
                        self.data.y[self.data.val_mask])

        accuracy, prec, rec, f1 = self.calculate_accuracy(
                pred_prob_node[self.data.val_mask],
                self.data.y[self.data.val_mask])
        self.log(train_bar, epoch, loss, accuracy, prec, rec, f1, split=f'{self.shadow} valid')
        self.valid_losses.append(loss.item())
        self.valid_accs.append(accuracy)
        self.valid_f1s.append(f1)
        return loss.item()


    def evaluate_on_test(self):
        model = self.model
        model.eval()
        confusion_matrix = torch.zeros(self.num_classes, self.num_classes,dtype=torch.long)

        with torch.no_grad():
            pred_prob_node = model(self.data)
            loss = self.loss(pred_prob_node[self.data.test_mask],
                            self.data.y[self.data.test_mask])
            preds = pred_prob_node.max(dim=1)[1]

            accuracy, prec, rec, f1 = self.calculate_accuracy(
                    pred_prob_node[self.data.test_mask],
                    self.data.y[self.data.test_mask])

            for t, p in zip(self.data.y[self.data.test_mask],
                            preds[self.data.test_mask]):
                if p.long() in range(self.num_classes):
                    confusion_matrix[t.long(), p.long()] += 1
                else:
                    confusion_matrix[t.long(), -1] += 1

            confusion_out = confusion_matrix.data.cpu().numpy()
            np.savetxt(os.path.join(self.time_dir, 'confusion_matrix.csv'),
                    confusion_out, delimiter=',', fmt='% 4d')

            # Output predictions
            print("Preparing predictions file...\n")
            pred_filename = os.path.join(self.time_dir,
                                        f'preds_seed{self.seed}.csv')
            with open(pred_filename, 'w') as pred_f:
                pred_f.write("Pred,Y\n")
                for idx in range(preds[self.data.test_mask].shape[0]):
                    pred_f.write(f"{preds[self.data.test_mask][idx]},{self.data.y[self.data.test_mask][idx]}\n")

            self.test_loss = loss.item()
            self.test_acc = accuracy
            self.test_f1 = f1

            print("Test set results\tLoss: {:.4f}\tNode Accuracy: {:.4f}\t"
                "Precision: {:.4f}\tRecall: {:.4f}\tF1: {:.4f}".format(
                    loss.item(), accuracy, prec, rec, f1))
        return loss.item(), accuracy, prec, rec, f1

    def output_results(self, best_score,shadow=None):
        '''
        Adds final test results to a csv file.
        '''
    
        filepath = os.path.join(self.time_dir, f'{shadow}_results.csv')
        best_val_loss = best_score
        epoch = self.valid_losses.index(best_val_loss)
        best_val_acc = self.valid_accs[epoch]
        best_val_f1 = self.valid_f1s[epoch]

        with open(filepath, 'w') as out_f:
            out_f.write('BestValidLoss,BestValidAcc,BestValidF1,'
                        'BestValidEpoch,TestLoss,TestAcc,TestF1,'
                        'NumTrainableParams,NumNodes(per_sg),NumEdges(per_sg),ModelConfig\n')
            out_f.write(f'{best_val_loss:.4f},{best_val_acc:.4f},'
                        f'{best_val_f1:.4f},{epoch},{self.test_loss:.4f},'
                        f'{self.test_acc:.4f},{self.test_f1:.4f},'
                        f'{self.trainable_params},{self.num_nodes},'
                        f'{self.num_edges},'
                        f'{self.model_name}\n')
    
    def get_shadow_data(self,shadow_data):
        model = self.model
        model.eval()
        with torch.no_grad():
            shadow_train_neg = model(shadow_data)[shadow_data.test_mask]
            shadow_train_pos = model(shadow_data)[shadow_data.train_mask]
            shadow_train_y = [1]*shadow_train_pos.shape[0]+[0]*shadow_train_neg.shape[0]
            shadow_train_y = torch.tensor(shadow_train_y,dtype=torch.float).to(self.device)
            shadow_train_x = torch.cat([shadow_train_pos,shadow_train_neg],dim=0)
            shadow_train_x = torch.softmax(shadow_train_x,dim=1)
            indices = torch.randperm(shadow_train_x.size(0))
            shadow_train_x = shadow_train_x[indices]
            shadow_train_y = shadow_train_y[indices]
        return shadow_train_x,shadow_train_y
    
    def get_vanilla_data(self,data):
        model = self.model
        model.eval()
        with torch.no_grad():
            shadow_test_neg = model(data)[data.test_mask]
            shadow_test_pos = model(data)[data.train_mask]
            shadow_test_neg_y = data.y[data.test_mask]
            shadow_test_pos_y = data.y[data.train_mask]
            shadow_test_y = [1]*shadow_test_pos.shape[0]+[0]*shadow_test_neg.shape[0]
            shadow_test_y = torch.tensor(shadow_test_y,dtype=torch.float).to(self.device)
            shadow_test_x_label = torch.cat([shadow_test_pos_y,shadow_test_neg_y],dim=0)
            shadow_test_x = torch.cat([shadow_test_pos,shadow_test_neg],dim=0)
            shadow_test_x = torch.softmax(shadow_test_x,dim=1)
            indices = torch.randperm(shadow_test_x.size(0))
            shadow_test_x = shadow_test_x[indices]
            shadow_test_y = shadow_test_y[indices]
            shadow_test_x_label = shadow_test_x_label[indices]
            shadow_test_x_label = one_hot(shadow_test_x_label)
        return shadow_test_x,shadow_test_y,shadow_test_x_label
        
 