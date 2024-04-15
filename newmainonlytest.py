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
        self.ss = ss
        device_num = ss.args.device_num
        try:
            torch.cuda.set_device(device_num)
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
        self.dp_type = ss.args.dp_type

        self.shadow_res = {
                "Dataset":[],
                "Vanilla train acc":[],
                "Vanilla train loss":[],
                "Vanilla train f1":[],
                "Vanilla test acc":[],
                "Vanilla test loss":[],
                "Vanilla test f1":[],
                "Shadow train acc":[],
                "Shadow train loss":[],
                "Shadow train f1":[],
                "Shadow test acc":[],
                "Shadow test loss":[],
                "Shadow test f1":[],
                "MIA subsample rate":[],
                "MIA mlp":[],
                "MIA svm":[],
                "MIA ranfor":[],
                "MIA logi":[],
                "MIA ada":[],
                "MIA confidence mse":[],
                "MIA confidence thr":[],
                "MIA seed":[],
                "Vanilla runtime per":[],
                "Shadow runtime per":[],
                "Epsilon":[],
                "Delta":[],
                "Dp":[],
                "Rdp":[],
                "Ldp":[],
                "Sampler":[],
                "Sampler batchsize":[],
                "Occurance k":[],
                "Cluster numparts":[],
                "Saint rootnodes":[],
                "Saint samplecoverage":[],
                "Saint walklenth":[],
                "Epochs":[],
                "Shadow epochs":[],
                "Num val":[],
                "Num test":[],
                "Layers":[],
                "Hidden dims":[],
                "Learning rate":[],
                "Shadow learning rate":[],
                "Dropout":[],
                "Activation":[],
                "Early stopping":[],
                "Patience":[],
                "Optim type":[]}
        
        if ss.args.mia == 'shadow':
            self.mia_shadow_mode = ss.args.mia_shadow_mode
            
            for i in trange(10,desc='10 Shadow dataset',leave=True):
                seed = self.seed+i*self.seed
                self.begin_log()
                if ss.args.private:
                    
                    self.train_dp(ss,seed)
                    
                else:
                    self.train_vanilla(ss,seed)
                
                print(f'Using {self.mia_shadow_mode} as MIA mode')
                
                self.train_shadow(ss,seed)
                self.shadow_res['MIA seed'].append(seed)
                print('\n<<<Begin attack data construction>>>\n')
                self.get_shadow_data(shadow_model=self.shadow_model,shadow_data=self.shadow_data)
                print('<<<Done attack data construction, begin attack>>>\n')
                self.shadow_MIA_mlp()
                self.shadow_MIA_svm()
                self.shadow_MIA_ranfor()
                self.shadow_MIA_logi()
                self.shadow_MIA_ada()
                self.confidence_MIA()
            
            # print(self.shadow_res)
        else:
            raise ValueError("No shadow mode deprecated now. Use shadow.")
        
        self.path_total_params = os.path.join(ss.root_dir, 'total_params.csv')

        then = time.time()
        runtime = then - now
        
        print(f"\n--- Script completed in {runtime} seconds ---\n")
        self.output_total_res()

    def train_dp(self,ss,seed=None):
        nownow = time.time()
        if seed:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True
        dataset = ss.args.dataset
        self.data = self.get_dataloader(dataset,num_test=ss.args.num_test,num_val=ss.args.num_val,shadow_set=False,mia_subsample_rate=ss.args.mia_subsample_rate)

        if ss.args.private:
            if self.dp_type == 'rdp':
                self.shadow_res['Dp'].append('False')
                self.shadow_res['Rdp'].append('True')
                self.shadow_res['Ldp'].append('False') 
                shadow = 'RDP'
                self.vanilla_model = vanilla_GCN_node(ss,
                                            data=self.data,
                                            shadow=shadow)
                best_score = self.vanilla_model.train_rdp() 
            elif self.dp_type == 'dp':
                self.shadow_res['Dp'].append('True')
                self.shadow_res['Rdp'].append('False')
                self.shadow_res['Ldp'].append('False')
                shadow = 'DP'
                self.vanilla_model = vanilla_GCN_node(ss,
                                            data=self.data,
                                            shadow=shadow)
                best_score = self.vanilla_model.train_dp()
            elif self.dp_type == 'ldp':
                self.shadow_res['Dp'].append('False')
                self.shadow_res['Rdp'].append('False')
                self.shadow_res['Ldp'].append('True')
                shadow = 'LDP'
                self.vanilla_model = vanilla_GCN_node(ss,
                                            data=self.data,
                                            shadow=shadow)
                best_score = self.vanilla_model.train_ldp() 
            
            private_paras = self.vanilla_model.private_paras
            self.shadow_res['Epsilon'].append(f'{private_paras[0]:.4f}')
            self.shadow_res['Delta'].append(f'{private_paras[1]:.4f}')
            
            
        else:
            raise TypeError("dp train not properly used, check your setting.")
        
        test_loss, test_acc, test_prec, test_rec, test_f1 = self.vanilla_model.evaluate_on_test()
        
        self.vanilla_model.output_results(best_score,shadow=shadow)
        print(f"Test score: {test_loss:.4f} with accuracy {test_acc:.4f} and f1 {test_f1:.4f}")
        self.shadow_res["Vanilla train acc"].append(f'{self.vanilla_model.train_accs[-1]:.4f}')
        self.shadow_res["Vanilla train loss"].append(f'{self.vanilla_model.train_losses[-1]:.4f}')
        self.shadow_res["Vanilla train f1"].append(f'{self.vanilla_model.train_f1s[-1]:.4f}')
        
        self.shadow_res["Vanilla test acc"].append(f'{test_acc:.4f}')
        self.shadow_res["Vanilla test loss"].append(f'{test_loss:.4f}')
        self.shadow_res["Vanilla test f1"].append(f'{test_f1:.4f}')

            
        thenthen = time.time()
        runtime = thenthen - nownow
        self.shadow_res["Vanilla runtime per"].append(f'{runtime/self.vanilla_model.last_epoch:.4f}')
        self.shadow_res["Epochs"].append(self.vanilla_model.last_epoch)
        with open(os.path.join(ss.root_dir, 'adam_hyperparams.csv'), 'a') as f: 
            f.write(f"{ss.args.dataset},{ss.args.noise_scale},{ss.args.learning_rate},{test_f1:.4f}\n")

    def train_vanilla(self,ss,seed=None):
        nownow = time.time()
        if seed:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True
        dataset = ss.args.dataset
        self.data = self.get_dataloader(dataset,num_test=ss.args.num_test,num_val=ss.args.num_val,shadow_set=False,mia_subsample_rate=ss.args.mia_subsample_rate)

        if ss.args.private:
            raise TypeError("vanilla train not properly used, check your setting.")
        else:
            self.shadow_res['Dp'].append('False')
            self.shadow_res['Rdp'].append('False')
            self.shadow_res['Ldp'].append('False')
            self.shadow_res['Epsilon'].append(f'0')
            self.shadow_res['Delta'].append(f'0')
            shadow = 'vanilla'
            self.vanilla_model = vanilla_GCN_node(ss,
                                                data=self.data,
                                                shadow=shadow)
            
            best_score = self.vanilla_model.train_vanilla()
        
        test_loss, test_acc, test_prec, test_rec, test_f1 = self.vanilla_model.evaluate_on_test()
        
        self.vanilla_model.output_results(best_score,shadow=shadow)
        print(f"Test score: {test_loss:.4f} with accuracy {test_acc:.4f} and f1 {test_f1:.4f}")
        self.shadow_res["Vanilla train acc"].append(f'{self.vanilla_model.train_accs[-1]:.4f}')
        self.shadow_res["Vanilla train loss"].append(f'{self.vanilla_model.train_losses[-1]:.4f}')
        self.shadow_res["Vanilla train f1"].append(f'{self.vanilla_model.train_f1s[-1]:.4f}')
        
        self.shadow_res["Vanilla test acc"].append(f'{test_acc:.4f}')
        self.shadow_res["Vanilla test loss"].append(f'{test_loss:.4f}')
        self.shadow_res["Vanilla test f1"].append(f'{test_f1:.4f}')

        thenthen = time.time()
        runtime = thenthen - nownow
        self.shadow_res["Vanilla runtime per"].append(f'{runtime/self.vanilla_model.last_epoch:.4f}')
        self.shadow_res["Epochs"].append(self.vanilla_model.last_epoch)
        with open(os.path.join(ss.root_dir, 'adam_hyperparams.csv'), 'a') as f: 
            f.write(f"{ss.args.dataset},{ss.args.noise_scale},{ss.args.learning_rate},{test_f1:.4f}\n")

    def train_shadow(self,ss,seed=None):
        nownow = time.time()
        if seed:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True

        dataset = ss.args.dataset
        self.shadow_data = self.get_dataloader(dataset,num_test=ss.args.num_test,num_val=ss.args.num_val,shadow_set=True,mia_subsample_rate=ss.args.mia_subsample_rate)
        self.shadow_model = vanilla_GCN_node(ss,data=self.shadow_data,shadow='shadow')
        
        best_score = self.shadow_model.train_vanilla()
        test_loss, test_acc, test_prec, test_rec, test_f1 = self.shadow_model.evaluate_on_test()
        
        self.shadow_model.output_results(best_score,shadow='shadow')
        print(f"Shadow Model Test score: {test_loss:.4f} with accuracy {test_acc:.4f} and f1 {test_f1:.4f}")
        self.shadow_res["Shadow train acc"].append(f'{self.shadow_model.train_accs[-1]:.4f}')
        self.shadow_res["Shadow train loss"].append(f'{self.shadow_model.train_losses[-1]:.4f}')
        self.shadow_res["Shadow train f1"].append(f'{self.shadow_model.train_f1s[-1]:.4f}')

        self.shadow_res["Shadow test acc"].append(f'{test_acc:.4f}')
        self.shadow_res["Shadow test loss"].append(f'{test_loss:.4f}')
        self.shadow_res["Shadow test f1"].append(f'{test_f1:.4f}')

        thenthen = time.time()
        runtime = thenthen - nownow
        self.shadow_res["Shadow runtime per"].append(f'{runtime/self.shadow_model.last_epoch:.4f}')
        self.shadow_res["Shadow epochs"].append(self.shadow_model.last_epoch)
        with open(os.path.join(ss.root_dir, 'adam_hyperparams.csv'), 'a') as f: 
            f.write(f"{ss.args.dataset},{ss.args.noise_scale},{ss.args.learning_rate},{test_f1:.4f}\n")
    
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

        elif dataset == 'reddit':
            data = get_dataset(cls="Reddit")
        
        elif dataset == 'flickr':
            data = get_dataset(cls="Flickr")
        elif dataset == 'github':
            data = get_dataset(cls="GitHub")
        elif dataset == 'lastfmasia':
            data = get_dataset(cls="LastFMAsia")
        elif dataset in ['RU','PT','DE','FR','ES','EN']:
            data = get_dataset(cls="Twitch",name=dataset)
        else:
            raise Exception("Incorrect dataset specified.")
        raw_data = data[0]
        # data = node_split(data[0],num_val=num_val,num_test=num_test)
        vanilla,shadow = subsample_graph_both_pyg(data=raw_data,rate=mia_subsample_rate)
        # vanilla,shadow = subsample_mask_graph_full(data,mia_subsample_rate)
        print(f"\nGet dataset {dataset}\n")
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
            data = vanilla
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
        self.shadow_mia_data_list = [shadow_train_x.detach().cpu().numpy(),
                                     shadow_train_y.detach().cpu().numpy(),
                                     shadow_test_x.detach().cpu().numpy(),
                                     shadow_test_y.detach().cpu().numpy(),
                                     shadow_test_x_label.detach().cpu().numpy()]     

    def shadow_MIA_mlp(self):
        
        [shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,_ ] = self.shadow_mia_data_list

        ss_dict = {
            "max_iter":800,
            "random_state":self.seed
        }
        mia_mlp = Shadow_MIA_mlp(shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,ss_dict)   
        mia_mlp_score = mia_mlp.evaluate(shadow_test_x,shadow_test_y) 
        self.shadow_res['MIA mlp'].append(f'{mia_mlp_score:.4f}')

    def shadow_MIA_svm(self):
        [shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,_] = self.shadow_mia_data_list

        ss_dict = {
            "kernel":"rbf",#linear rbf poly sigmoid
            "random_state":self.seed
        }
        mia_svm = Shadow_MIA_svm(shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,ss_dict)
        mia_svm_score = mia_svm.evaluate(shadow_test_x,shadow_test_y)
        self.shadow_res['MIA svm'].append(f'{mia_svm_score:.4f}')

    def shadow_MIA_ranfor(self):
        [shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,_] = self.shadow_mia_data_list

        ss_dict = {
            "criterion":"gini",#'gini', 'entropy', 'log_loss'
            "random_state":self.seed,
            "n_estimators":400
        }
        mia_ranfor = Shadow_MIA_ranfor(shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,ss_dict)
        mia_ranfor_score = mia_ranfor.evaluate(shadow_test_x,shadow_test_y)
        self.shadow_res['MIA ranfor'].append(f'{mia_ranfor_score:.4f}')

    def shadow_MIA_logi(self):
        [shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,_] = self.shadow_mia_data_list

        ss_dict = {
            "penalty":"elasticnet",#'elasticnet', 'l1', 'l2'
            "random_state":self.seed,
            "max_iter":100,
            'solver':'saga',#‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’
            'l1_ratio':0.05
        }
        mia_logi = Shadow_MIA_logi(shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,ss_dict)
        mia_logi_score = mia_logi.evaluate(shadow_test_x,shadow_test_y)
        self.shadow_res['MIA logi'].append(f'{mia_logi_score:.4f}')

    def shadow_MIA_ada(self):
        [shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,_] = self.shadow_mia_data_list

        ss_dict = {
            "algorithm":"SAMME.R",#'SAMME', 'SAMME.R'
            "random_state":self.seed,
            "n_estimators":50
        }
        mia_ada = Shadow_MIA_ada(shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,ss_dict)
        mia_ada_score = mia_ada.evaluate(shadow_test_x,shadow_test_y)
        self.shadow_res['MIA ada'].append(f'{mia_ada_score:.4f}')

    def confidence_MIA(self):
        # confidences = trange(start=0.001,stop=1,step=0.001,desc='Loss confidence range',leave=True)
        res = {}
        j = 0
        best_score = 0 
        confidences = []
        for i in range(100):
            j+=0.01
            res[j] = []
            confidences.append(j)
        [_,_,shadow_test_x,shadow_test_y,shadow_test_x_label] = self.shadow_mia_data_list
        shadow_test_y = shadow_test_y.reshape(-1)

        losses = []
        for i in range(shadow_test_x.shape[0]):
            item = shadow_test_x[i]
            label = shadow_test_x_label[i]
            loss = metrics.mean_squared_error(label,item,squared=False)
            losses.append(1-loss)
            for j in confidences:
                res[j].append(1. if (loss<j) else 0.)
        # print(metrics.roc_auc_score(shadow_test_y,losses))
        for j in confidences:
            temp = metrics.accuracy_score(shadow_test_y,res[j])
            print(metrics.roc_auc_score(shadow_test_y,res[j]))
            if temp>best_score:
                best_score = temp
                best_j = j
        print(f"\nBest Confidence MIA attack with threshold {best_j:.4f}:\n")
        print(metrics.classification_report(shadow_test_y,res[best_j],labels=range(2)))
        self.shadow_res['MIA confidence mse'].append(f'{best_score:.4f}')
        self.shadow_res['MIA confidence thr'].append(f'{best_j:.4f}')

    def begin_log(self):
        
        self.shadow_res['Sampler'].append(self.ss.args.sampler)
        self.shadow_res['Sampler batchsize'].append(self.ss.args.sampler_batchsize)
        self.shadow_res["Occurance k"].append(self.ss.args.occurance_k)
        self.shadow_res["Cluster numparts"].append(self.ss.args.cluster_numparts)
        self.shadow_res["Saint rootnodes"].append(self.ss.args.saint_rootnodes)
        self.shadow_res["Saint samplecoverage"].append(self.ss.args.saint_samplecoverage)
        self.shadow_res["Saint walklenth"].append(self.ss.args.saint_walklenth)
        self.shadow_res["Dataset"].append(self.ss.args.dataset)
        self.shadow_res["MIA subsample rate"].append(self.ss.args.mia_subsample_rate)
        self.shadow_res["Num val"].append(self.ss.args.num_val)
        self.shadow_res["Num test"].append(self.ss.args.num_test)
        self.shadow_res["Layers"].append(self.ss.args.k_layers)
        self.shadow_res["Hidden dims"].append(self.ss.args.hidden_dim)
        self.shadow_res["Learning rate"].append(self.ss.args.learning_rate)
        self.shadow_res["Shadow learning rate"].append(self.ss.args.shadow_learning_rate)
        self.shadow_res["Dropout"].append(self.ss.args.dropout)
        self.shadow_res["Activation"].append(self.ss.args.activation)
        self.shadow_res["Early stopping"].append(self.ss.args.early_stopping)
        self.shadow_res["Patience"].append(self.ss.args.patience)
        self.shadow_res["Optim type"].append(self.ss.args.optim_type)

    def output_total_res(self):
        df = pd.DataFrame(self.shadow_res)
        if os.path.exists(self.path_total_params):
            old_df = pd.read_csv(self.path_total_params,index_col=0)
            df = pd.concat([old_df,df],ignore_index=True)
            df.to_csv(self.path_total_params,mode='w')
        else:
            df.to_csv(self.path_total_params)

if __name__ == '__main__':
    model = node_GCN()
