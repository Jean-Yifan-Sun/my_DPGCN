import torch,sys,os,torch_geometric,time,random,math
from torch.nn import Linear
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.utils import *
from dataset import get_dataset,node_split
from sklearn.metrics import precision_recall_fscore_support
from model import *
from settings import Settings
from utils import *
from sklearn import metrics
from tqdm import tqdm
import pandas as pd

class node_GCN():
    def __init__(self) -> None:
        now = time.time()

        # Setting the seed
        ss = Settings()
        ss.make_dirs()
        try:
            torch.cuda.set_device(3)
        except:
            torch.cuda.set_device(0) 
        self.seed = ss.args.seed
        random.seed(ss.args.seed)
        torch.manual_seed(ss.args.seed)
        torch.cuda.manual_seed(ss.args.seed)
        np.random.seed(ss.args.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        
        if ss.args.mia == 'shadow':
            self.mia_shadow_mode = ss.args.mia_shadow_mode
            self.shadow_res = {
                "MIA mlp":[],
                "MIA svm":[],
                "MIA ranfor":[],
                "MIA confidence mse":[],
                "MIA confidence thr":[],
                "MIA seed":[]
            }
            for i in trange(10,desc='10 Shadow dataset',leave=True):
                seed = self.seed+i*self.seed
                self.train_vanilla(ss,seed)
                print(f'Using {self.mia_shadow_mode} as MIA mode')
                
                self.shadow_model, self.shadow_data = self.train_shadow(ss,seed)
                self.shadow_res['MIA seed'].append(seed)
                self.shadow_MIA_mlp(shadow_model=self.shadow_model,shadow_data=self.shadow_data)
                self.shadow_MIA_svm(shadow_model=self.shadow_model,shadow_data=self.shadow_data)
                self.shadow_MIA_ranfor(shadow_model=self.shadow_model,shadow_data=self.shadow_data)
                self.confidence_MIA(shadow_model=self.shadow_model,shadow_data=self.shadow_data)
        else:
            self.train_vanilla(ss)
        new_res = pd.DataFrame(self.shadow_res)
        path = os.path.join(ss.privacy_dir,f'MIA_{self.mia_shadow_mode}_{ss.args.mia_subsample_rate}_result.csv')
        try: 
            old_res = pd.read_csv(path)
            new_res = pd.concat([old_res,new_res])
        except:
            new_res.to_csv(path)
        print(new_res)
        then = time.time()
        runtime = then - now
        print(f"\n--- Script completed in {runtime} seconds ---\n")

    def train_vanilla(self,ss,seed=None):
        if seed:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True
        dataset = ss.args.dataset
        self.data = self.get_dataloader(dataset,num_test=ss.args.num_test,num_val=ss.args.num_val,shadow_set=False,mia_subsample_rate=ss.args.mia_subsample_rate)
        self.vanilla_model = vanilla_GCN_node(ss,
                                              data=self.data,
                                              shadow='vanilla')
        
        best_score = self.vanilla_model.train()
        test_loss, test_acc, test_prec, test_rec, test_f1 = self.vanilla_model.evaluate_on_test()
        
        self.vanilla_model.output_results(best_score,shadow='vanilla')
        print(f"Test score: {test_loss:.4f} with accuracy {test_acc:.4f} and f1 {test_f1:.4f}")
        
        with open(os.path.join(ss.root_dir, 'adam_hyperparams.csv'), 'a') as f: 
            f.write(f"{ss.args.dataset},{ss.args.noise_scale},{ss.args.learning_rate},{test_f1:.4f}\n")

    def train_shadow(self,ss,seed=None):
        if seed:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True

        dataset = ss.args.dataset
        shadow_data = self.get_dataloader(dataset,num_test=ss.args.num_test,num_val=ss.args.num_val,shadow_set=True,mia_subsample_rate=ss.args.mia_subsample_rate)
        model = vanilla_GCN_node(ss,data=shadow_data,shadow='shadow')
        
        best_score = model.train()
        test_loss, test_acc, test_prec, test_rec, test_f1 = model.evaluate_on_test()
        
        model.output_results(best_score,shadow='shadow')
        print(f"Shadow Model Test score: {test_loss:.4f} with accuracy {test_acc:.4f} and f1 {test_f1:.4f}")
    
        with open(os.path.join(ss.root_dir, 'adam_hyperparams.csv'), 'a') as f: 
            f.write(f"{ss.args.dataset},{ss.args.noise_scale},{ss.args.learning_rate},{test_f1:.4f}\n")
        return model,shadow_data
    
    def get_dataloader(self,dataset='cora', num_val=None, num_test=None,shadow_set=False,mia_subsample_rate=1):
        '''
        Prepares the dataloader for a particular split of the data.
        '''
        if dataset == 'cora':
            data = get_dataset(cls="Planetoid",name="Cora",num_test=num_test,num_val=num_val)

        elif dataset == 'citeseer':
            data = get_dataset(cls="Planetoid",name="CiteSeer",num_test=num_test,num_val=num_val)

        elif dataset == 'pubmed':
            data = get_dataset(cls="Planetoid",name="PubMed",num_test=num_test,num_val=num_val)

        elif dataset == 'computers':
            data = get_dataset(cls="Amazon",name="Computers",num_test=num_test,num_val=num_val)

        elif dataset == 'photo':
            data = get_dataset(cls="Amazon",name="Photo",num_test=num_test,num_val=num_val)

        elif dataset == 'cs':
            data = get_dataset(cls="Coauthor",name="CS",num_test=num_test,num_val=num_val)

        elif dataset == 'physics':
            data = get_dataset(cls="Coauthor",name="Physics",num_test=num_test,num_val=num_val)
        else:
            raise Exception("Incorrect dataset specified.")
        data = data[0]
        vanila,shadow = subsample_graph_both_pyg(data=data,rate=mia_subsample_rate)
        if shadow_set:
            print("Shadow dataset Subsampling graph...")
            # subsample_graph(shadow_data, rate=self.mia_subsample_rate,
                            # maintain_class_dists=True)
            data = shadow
            data = node_split(data,num_val=num_val,num_test=num_test)
            
            print(f"Shadow: Total number of nodes: {data.x.shape[0]}")
            print(f"Shadow: Total number of edges: {data.edge_index.shape[1]}")
            print(f"Shadow: Number of train nodes: {data.train_mask.sum().item()}")
            print(f"Shadow: Number of validation nodes: {data.val_mask.sum().item()}")
            print(f"Shadow: Number of test nodes: {data.test_mask.sum().item()}")
        else:
            data = vanila
            data = node_split(data,num_val=num_val,num_test=num_test)
            print(f"Total number of nodes: {data.x.shape[0]}")
            print(f"Total number of edges: {data.edge_index.shape[1]}")
            print(f"Number of train nodes: {data.train_mask.sum().item()}")
            print(f"Number of validation nodes: {data.val_mask.sum().item()}")
            print(f"Number of test nodes: {data.test_mask.sum().item()}")
        return data
    
    def get_shadow_data(self,shadow_model:vanilla_GCN_node,shadow_data):
        model = shadow_model
        shadow_train_x,shadow_train_y, = model.get_shadow_data(shadow_data)
        
        model = self.vanilla_model
        shadow_test_x,shadow_test_y,shadow_test_x_label = model.get_vanilla_data(self.data)
            
        return shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,shadow_test_x_label

    def shadow_MIA_mlp(self,shadow_model,shadow_data):
        shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,_ = self.get_shadow_data(shadow_model,shadow_data)

        ss_dict = {
            "max_iter":800,
            "random_state":self.seed
        }
        mia_mlp = Shadow_MIA_mlp(shadow_train_x.detach().cpu().numpy(),shadow_train_y.detach().cpu().numpy(),shadow_test_x.detach().cpu().numpy(),shadow_test_y.detach().cpu().numpy(),ss_dict)   
        mia_mlp_score = mia_mlp.evaluate(shadow_test_x.detach().cpu().numpy(),shadow_test_y.detach().cpu().numpy()) 
        self.shadow_res['MIA mlp'].append(f'{mia_mlp_score:.4f}')

    def shadow_MIA_svm(self,shadow_model,shadow_data):
        shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,_ = self.get_shadow_data(shadow_model,shadow_data)

        ss_dict = {
            "kernel":"rbf",#linear rbf poly sigmoid
            "random_state":self.seed
        }
        mia_svm = Shadow_MIA_svm(shadow_train_x.detach().cpu().numpy(),shadow_train_y.detach().cpu().numpy(),shadow_test_x.detach().cpu().numpy(),shadow_test_y.detach().cpu().numpy(),ss_dict)
        mia_svm_score = mia_svm.evaluate(shadow_test_x.detach().cpu().numpy(),shadow_test_y.detach().cpu().numpy())
        self.shadow_res['MIA svm'].append(f'{mia_svm_score:.4f}')

    def shadow_MIA_ranfor(self,shadow_model,shadow_data):
        shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,_ = self.get_shadow_data(shadow_model,shadow_data)

        ss_dict = {
            "criterion":"gini",#'gini', 'entropy', 'log_loss'
            "random_state":self.seed,
            "n_estimators":100
        }
        mia_ranfor = Shadow_MIA_ranfor(shadow_train_x.detach().cpu().numpy(),shadow_train_y.detach().cpu().numpy(),shadow_test_x.detach().cpu().numpy(),shadow_test_y.detach().cpu().numpy(),ss_dict)
        mia_ranfor_score = mia_ranfor.evaluate(shadow_test_x.detach().cpu().numpy(),shadow_test_y.detach().cpu().numpy())
        self.shadow_res['MIA ranfor'].append(f'{mia_ranfor_score:.4f}')

    def confidence_MIA(self,shadow_model,shadow_data):
        confidences = [0.02,0.04,0.06,0.07,0.08,0.091,0.092,0.093,0.094,0.095,0.096,0.097,0.098,0.099,0.1,0.101,0.102,0.103,0.104,0.105,0.106,0.107,0.108,0.2,0.3,0.4,0.5]
        res = {}
        for i in confidences:
            res[i] = []
        _,_,shadow_test_x,shadow_test_y,shadow_test_x_label = self.get_shadow_data(shadow_model,shadow_data)
        shadow_test_x_label = shadow_test_x_label.detach().cpu().numpy()
        shadow_test_x = shadow_test_x.detach().cpu().numpy()
        shadow_test_y = shadow_test_y.detach().cpu().numpy().reshape(-1)
        
        for i in range(shadow_test_x.shape[0]):
            item = shadow_test_x[i]
            index = np.argmax(shadow_test_x_label[i])
            label = shadow_test_x_label[i]
            for j in confidences:
                temp = metrics.mean_squared_error(label,item,squared=False)
                if (temp<j) :
                    res[j].append(1.)
                else:
                    res[j].append(0.)
        best_score = 0            
        for j in confidences:
            
            temp = metrics.accuracy_score(shadow_test_y,res[j])
            if temp>best_score:
                best_score = temp
                best_j = j
        print(f"\nBest Confidence MIA attack with threshold {best_j}:\n")
        print(metrics.classification_report(shadow_test_y,res[best_j],labels=range(2)))
        self.shadow_res['MIA confidence mse'].append(f'{best_score:.4f}')
        self.shadow_res['MIA confidence thr'].append(f'{best_j:.4f}')

if __name__ == '__main__':
    model = node_GCN()
