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

        self.train_vanilla(ss)
        
        if ss.args.mia == 'shadow':
            self.shadow_model, self.shadow_data = self.train_shadow(ss)
            
            self.shadow_MIA_mlp(shadow_model=self.shadow_model,shadow_data=self.shadow_data)
            self.shadow_MIA_svm(shadow_model=self.shadow_model,shadow_data=self.shadow_data)
            self.shadow_MIA_ranfor(shadow_model=self.shadow_model,shadow_data=self.shadow_data)
            self.confidence_MIA(shadow_model=self.shadow_model,shadow_data=self.shadow_data)
        
        then = time.time()
        runtime = then - now
        print(f"\n--- Script completed in {runtime} seconds ---\n")

    def train_vanilla(self,ss):
        dataset = ss.args.dataset
        self.data = self.get_dataloader(dataset,num_test=ss.args.num_test,num_val=ss.args.num_val)
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
        shadow_data = self.get_dataloader(dataset,num_test=ss.args.num_test,num_val=ss.args.num_val,mia_subsample_rate=ss.args.mia_subsample_rate)
        model = vanilla_GCN_node(ss,data=shadow_data,shadow='shadow')
        
        best_score = model.train()
        test_loss, test_acc, test_prec, test_rec, test_f1 = model.evaluate_on_test()
        
        model.output_results(best_score,shadow='shadow')
        print(f"Shadow Model Test score: {test_loss:.4f} with accuracy {test_acc:.4f} and f1 {test_f1:.4f}")
    
        with open(os.path.join(ss.root_dir, 'adam_hyperparams.csv'), 'a') as f: 
            f.write(f"{ss.args.dataset},{ss.args.noise_scale},{ss.args.learning_rate},{test_f1:.4f}\n")
        return model,shadow_data
    
    def get_dataloader(self,dataset='cora', num_val=None, num_test=None,get_edge_counts=False,mia_subsample_rate=1):
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

        if mia_subsample_rate != 1.:
            print("Shadow dataset Subsampling graph...")
            # subsample_graph(shadow_data, rate=self.mia_subsample_rate,
                            # maintain_class_dists=True)
            data = subsample_graph_pyg(data,rate=mia_subsample_rate)
            data = node_split(data,num_val=0.1,num_test=0.45)
            
            print(f"Shadow: Total number of nodes: {data.x.shape[0]}")
            print(f"Shadow: Total number of edges: {data.edge_index.shape[1]}")
            print(f"Shadow: Number of train nodes: {data.train_mask.sum().item()}")
            print(f"Shadow: Number of validation nodes: {data.val_mask.sum().item()}")
            print(f"Shadow: Number of test nodes: {data.test_mask.sum().item()}")
        else:
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

    def shadow_MIA_svm(self,shadow_model,shadow_data):
        shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,_ = self.get_shadow_data(shadow_model,shadow_data)

        ss_dict = {
            "kernel":"rbf",#linear rbf poly sigmoid
            "random_state":self.seed
        }
        mia_svm = Shadow_MIA_svm(shadow_train_x.detach().cpu().numpy(),shadow_train_y.detach().cpu().numpy(),shadow_test_x.detach().cpu().numpy(),shadow_test_y.detach().cpu().numpy(),ss_dict)

    def shadow_MIA_ranfor(self,shadow_model,shadow_data):
        shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,_ = self.get_shadow_data(shadow_model,shadow_data)

        ss_dict = {
            "criterion":"gini",#'gini', 'entropy', 'log_loss'
            "random_state":self.seed,
            "n_estimators":100
        }
        mia_ranfor = Shadow_MIA_ranfor(shadow_train_x.detach().cpu().numpy(),shadow_train_y.detach().cpu().numpy(),shadow_test_x.detach().cpu().numpy(),shadow_test_y.detach().cpu().numpy(),ss_dict)

    def confidence_MIA(self,shadow_model,shadow_data):
        confidences = [0.01,0.02,0.03,0.04,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]
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
                temp = metrics.mean_squared_error(label,item)
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

if __name__ == '__main__':
    model = node_GCN()
