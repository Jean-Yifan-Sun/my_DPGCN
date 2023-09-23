import torch,sys,os,torch_geometric,time,random,math
from torch.nn import Linear
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from dataset import get_dataset,node_split
from sklearn.metrics import precision_recall_fscore_support
from model import *
from settings import Settings
from utils import *
from sklearn import metrics

class GCNModel(object):
    def __init__(self, ss):
        self.dataset = ss.args.dataset
        self.root_dir = ss.root_dir
        self.log_dir = ss.log_dir
        self.time_dir = ss.time_dir
        self.model_name = ss.model_name
        self.epochs = ss.args.epochs
        self.hidden_dim = ss.args.hidden_dim
        self.learning_rate = ss.args.learning_rate
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

        self.learning_rate_decay = False
        self.scheduler = None
        self.max_epochs_lr_decay = 200
        self.scheduler_gamma = 1

        self.split_graph = ss.args.split_graph
        self.split_n_subgraphs = ss.args.split_n_subgraphs

        self.mia = ss.args.mia
        self.mia_subsample_rate = ss.args.mia_subsample_rate

        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'cpu')

        self.seed = ss.args.seed

        self.total_params = 0
        self.trainable_params = 0

        # Privacy parameters
        self.private = ss.args.private
        self.delta = ss.args.delta
        self.noise_scale = ss.args.noise_scale
        self.gradient_norm_bound = ss.args.gradient_norm_bound
        self.lot_size = ss.args.lot_size
            # Number of subgraphs in a lot, if no graph splitting then 1
        self.sample_size = ss.args.sample_size
        self.alpha = None
        self.total_samples = self.split_n_subgraphs

        # Results
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

        temp = self.get_dataloader(dataset=self.dataset,
                                        num_test=ss.args.num_test,
                                        num_val=ss.args.num_val,
                                        get_edge_counts=False)
        self.data = temp[0]
        self.shadow_data = None

        self._init_model()

        if self.mia == 'shadow':
            self.shadow_train_accs = []
            self.shadow_train_losses = []
            self.shadow_train_f1s = []
            self.shadow_valid_accs = []
            self.shadow_valid_losses = []
            self.shadow_valid_f1s = []
            self.shadow_test_loss = None
            self.shadow_test_acc = None
            self.shadow_test_f1 = None
            self.shadow_random_baseline = True
            self.shadow_majority_baseline = True
            self.shadow_model_name = ss.model_name
            self.shadow_scheduler = None
            self.shadow_data = temp[1]

            self._init_shadow_model()

        
        

    def get_dataloader(self, dataset='cora', num_val=None, num_test=None,
                       get_edge_counts=False):
        '''
        Prepares the dataloader for a particular split of the data.
        '''
        if dataset == 'cora':
            data = get_dataset(cls="Planetoid",name="Cora",num_test=num_test,num_val=num_val)
            # data = Planetoid(self.root_dir, "Cora")[0] 
            # data.train_mask.fill_(True)
            # data.train_mask[data.val_mask | data.test_mask] = False

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
        #     data = Reddit(os.path.join(self.root_dir, 'Reddit'))[0]
        # elif dataset == 'reddit-small':
        #     try:
        #         data = torch.load(os.path.join(self.root_dir, 'RedditS',
        #                                        'processed', 'data.pt'))
        #     except FileNotFoundError:
        #         print("Small reddit data not found, preparing...")
        #         data = make_small_reddit(rate=0.1)
        # elif dataset == 'pokec-pets':
        #     try:
        #         data = torch.load(
        #                 os.path.join(self.root_dir, 'Pokec', 'processed',
        #                              f'pokec-pets_{pokec_feat_type}_cased.pt')
        #                 )
        #     except FileNotFoundError:
        #         print("Pokec dataset not found, preparing...")
        #         data = prepare_pokec_main(feat_type=pokec_feat_type)
        else:
            raise Exception("Incorrect dataset specified.")

        ###
        # Place code here for mini-batching/graph splitting
        ###

        shadow_data = data.copy()
        data = data[0]
        shadow_data = shadow_data[0]
        if self.split_graph:
            batch_masks = random_graph_split(data, n_subgraphs=self.split_n_subgraphs)
            batch_masks = [mask.to(self.device) for mask in batch_masks]
            data.batch_masks = batch_masks
            num_sample_nodes = data.x[batch_masks[0]].shape[0]
            print(f"Split graph into {self.split_n_subgraphs} subgraphs of "
                  f"{num_sample_nodes} nodes.")
    
        print(f"Total number of nodes: {data.x.shape[0]}")
        print(f"Total number of edges: {data.edge_index.shape[1]}")
        print(f"Number of train nodes: {data.train_mask.sum().item()}")
        print(f"Number of validation nodes: {data.val_mask.sum().item()}")
        print(f"Number of test nodes: {data.test_mask.sum().item()}")

        if self.mia == 'shadow':
            if self.mia_subsample_rate != 1.:
                print("Shadow dataset Subsampling graph...")
                # subsample_graph(shadow_data, rate=self.mia_subsample_rate,
                                # maintain_class_dists=True)
                subsample_graph_all(shadow_data,rate=self.mia_subsample_rate)
                shadow_data = node_split(shadow_data,num_val=0.2,num_test=0.4)
                shadow_data = shadow_data.to(self.device)
                self.shadow_num_nodes = shadow_data.x.shape[0]
                self.shadow_num_edges = shadow_data.edge_index.shape[1]
                print(f"Shadow: Total number of nodes: {shadow_data.x.shape[0]}")
                print(f"Shadow: Total number of edges: {shadow_data.edge_index.shape[1]}")
                print(f"Shadow: Number of train nodes: {shadow_data.train_mask.sum().item()}")
                print(f"Shadow: Number of validation nodes: {shadow_data.val_mask.sum().item()}")
                print(f"Shadow: Number of test nodes: {shadow_data.test_mask.sum().item()}")


        data = data.to(self.device)
        if get_edge_counts:
            if self.split_graph:
                print("Graph split: Showing edge count for first subgraph.")
            num_train_edges, num_test_edges = get_train_edge_count(data, split_graph=self.split_graph)
            print(f"Number of train edges: {num_train_edges}")
            print(f"Number of test edges: {num_test_edges}")

        self.num_nodes = data.x.shape[0]
        self.num_edges = data.edge_index.shape[1]

        return data,shadow_data
    
    def _init_shadow_model(self):
        self.num_classes = len(torch.unique(self.shadow_data.y))
        print("Using {}".format(self.device))
        
        ss_dict = {
            "k_layer":self.k_layers,
            "chanels":self.hidden_dim,
            "dropout":self.dropout,
            "activation":self.activation,
            "optimizer":self.optim_type,
            "num_features":self.shadow_data.num_node_features,
            "num_classes":self.num_classes
        }
        
        if self.k_layers == 2:
            model = two_layer_GCN(ss_dict).to(self.device)
        elif self.k_layers == 3:
            model = three_layer_GCN(ss_dict).to(self.device)
        else:
            model = one_layer_GCN(ss_dict).to(self.device)
        if self.parallel:
            pass

        # for param in model.parameters():
        #     print(param.shape)
        
        total_params = 0
        for param in list(model.parameters()):
            nn = 1
            for sp in list(param.size()):
                nn = nn * sp
            total_params += nn
        self.shadow_total_params = total_params
        print("Total parameters", self.shadow_total_params)

        model_params = filter(lambda param: param.requires_grad,
                              model.parameters())
        trainable_params = sum([np.prod(param.size())
                                for param in model_params])
        self.shadow_trainable_params = trainable_params
        print("Trainable parameters", self.shadow_trainable_params)

        if self.optim_type == 'sgd':
            self.shadow_optimizer = torch.optim.SGD(model.parameters(),
                                                lr=self.learning_rate,
                                                weight_decay=self.weight_decay)

        elif self.optim_type == 'adam':
            self.shadow_optimizer = torch.optim.Adam(model.parameters(),
                                                lr=self.learning_rate,
                                                weight_decay=self.weight_decay)
        else:
            raise Exception(f"{self.optim_type} not a valid optimizer (adam or sgd).")

        if self.learning_rate_decay:
            self.shadow_scheduler = torch.optim.lr_scheduler.StepLR(self.shadow_optimizer, step_size=1,
                                                             gamma=self.scheduler_gamma)
        self.shadow_loss = torch.nn.CrossEntropyLoss()

        self.shadow_model = model

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
            # model = GCNNet(self.data.num_node_features, self.num_classes,
            #                self.hidden_dim, self.device, self.activation,
            #                self.dropout).double().to(self.device)
        elif self.k_layers == 3:
            model = three_layer_GCN(ss_dict).to(self.device)
        else:
            model = one_layer_GCN(ss_dict).to(self.device)
        if self.parallel:
            # model = DataParallel(model)
            pass

        # for param in model.parameters():
        #     print(param.shape)
        
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

        if self.private:
            # if self.optim_type == 'sgd':
            #     self.optimizer = DPSGD(model.parameters(), self.noise_scale,
            #                            self.gradient_norm_bound, self.lot_size,
            #                            self.sample_size, lr=self.learning_rate)
            # elif self.optim_type == 'adam':
            #     self.optimizer = DPAdam(model.parameters(), self.noise_scale,
            #                             self.gradient_norm_bound,
            #                             self.lot_size, self.sample_size,
            #                             lr=self.learning_rate)
            # else:
            #     raise Exception(f"{self.optim_type} not a valid optimizer (adam or sgd).")
            pass
        else:
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
        # self.loss = torch.nn.NLLLoss()

        self.model = model

    
    def shadow_train(self):
        model = self.shadow_model
        optimizer = self.shadow_optimizer
        early_stopping = EarlyStopping(self.patience)

        model.train()

        parameters = []
        q = self.lot_size / self.total_samples
            # 'Sampling ratio'
            # Number of subgraphs in a lot divided by total number of subgraphs
            # If no graph splitting, both values are 1
        max_range = self.epochs / q  # max number of Ts
        max_parameters = [(q, self.noise_scale, max_range)]

        print('Shadow Training...')
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred_prob_node = model(self.shadow_data)
            loss = self.shadow_loss(pred_prob_node[self.shadow_data.train_mask],
                                    self.shadow_data.y[self.shadow_data.train_mask])
            loss.backward()
            optimizer.step()

            if self.shadow_scheduler != None and epoch < self.max_epochs_lr_decay:
                print("Old LR:", self.shadow_optimizer.param_groups[0]['lr'])
                self.shadow_scheduler.step()
                print("New LR:", self.shadow_optimizer.param_groups[0]['lr'])

            accuracy, prec, rec, f1 = self.calculate_accuracy(
                    pred_prob_node[self.shadow_data.train_mask],
                    self.shadow_data.y[self.shadow_data.train_mask])
            loss = loss.item()

            self.log(epoch, loss, accuracy, prec, rec, f1,
                     split='shadow train')
            self.shadow_train_losses.append(loss)
            self.shadow_train_accs.append(accuracy)
            self.shadow_train_f1s.append(f1)

            val_loss = self.evaluate_on_valid(model, epoch,shadow=True)

            self.plot_learning_curve(shadow=True)
            if self.early_stopping:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    break
            print('\n')

        if self.early_stopping:
            return -early_stopping.best_score
        else:
            return val_loss


    def train(self):
        model = self.model
        optimizer = self.optimizer
        early_stopping = EarlyStopping(self.patience)

        model.train()

        parameters = []
        q = self.lot_size / self.total_samples
            # 'Sampling ratio'
            # Number of subgraphs in a lot divided by total number of subgraphs
            # If no graph splitting, both values are 1
        max_range = self.epochs / q  # max number of Ts
        max_parameters = [(q, self.noise_scale, max_range)]

        print('Training...')
        for epoch in range(self.epochs):
            if self.split_graph:
                random.shuffle(self.data.batch_masks)

            if self.private:
                break # 调试先 
                if self.split_graph:
                    batch_losses = []
                    batch_accs = []
                    batch_precs = []
                    batch_recs = []
                    batch_f1s = []
                    lot_t = 0
                    for idx in range(self.split_n_subgraphs):
                        T_k = (lot_t + 1) + ((1 / q) * epoch)
                        optimizer.zero_accum_grad()
                        optimizer.zero_sample_grad()
                        pred_prob_node = model(self.data)
                        loss = self.loss(pred_prob_node[self.data.batch_masks[idx]],
                                         self.data.y[self.data.batch_masks[idx]])
                        loss.backward()
                        optimizer.per_sample_step()
                        optimizer.step(self.device)

                        parameters = [(q, self.noise_scale, T_k)]
                        eps, delta = get_priv(parameters, delta=self.delta,
                                              max_lmbd=32)
                        maxeps, maxdelta = get_priv(max_parameters,
                                                    delta=self.delta, max_lmbd=32)
                        b_acc, b_prec, b_rec, b_f1 = self.calculate_accuracy(
                                pred_prob_node[self.data.batch_masks[idx]],
                                self.data.y[self.data.batch_masks[idx]])

                        batch_losses.append(loss.item())
                        batch_accs.append(b_acc)
                        batch_precs.append(b_prec)
                        batch_recs.append(b_rec)
                        batch_f1s.append(b_f1)

                        lot_t += 1
                        print("Spent privacy (function accountant): \n", eps)
                        print("Spent MAX privacy (function accountant): \n", maxeps)

                else:
                    lot_t = 0  # For 1-graph datasets, always batch of 1
                    T_k = (lot_t + 1) + ((1 / q) * epoch)

                    optimizer.zero_accum_grad()
                    optimizer.zero_sample_grad()
                    pred_prob_node = model(self.data)
                    loss = self.loss(pred_prob_node[self.data.train_mask],
                                     self.data.y[self.data.train_mask])
                    loss.backward()
                    optimizer.per_sample_step()
                    optimizer.step(self.device)

                    parameters = [(q, self.noise_scale, T_k)]
                    eps, delta = get_priv(parameters, delta=self.delta,
                                          max_lmbd=32)
                    maxeps, maxdelta = get_priv(max_parameters,
                                                delta=self.delta, max_lmbd=32)
                    print("Spent privacy (function accountant): \n", eps)
                    print("Spent MAX privacy (function accountant): \n", maxeps)
            else:
                if self.split_graph:
                    batch_losses = []
                    batch_accs = []
                    batch_precs = []
                    batch_recs = []
                    batch_f1s = []
                    for idx in range(self.split_n_subgraphs):
                        optimizer.zero_grad()
                        pred_prob_node = model(self.data)
                        loss = self.loss(pred_prob_node[self.data.batch_masks[idx]],
                                         self.data.y[self.data.batch_masks[idx]])
                        loss.backward()
                        optimizer.step()

                        b_acc, b_prec, b_rec, b_f1 = self.calculate_accuracy(
                                pred_prob_node[self.data.batch_masks[idx]],
                                self.data.y[self.data.batch_masks[idx]])

                        batch_losses.append(loss.item())
                        batch_accs.append(b_acc)
                        batch_precs.append(b_prec)
                        batch_recs.append(b_rec)
                        batch_f1s.append(b_f1)
                else:
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

            if self.split_graph:
                loss = np.mean(batch_losses)
                accuracy = np.mean(batch_accs)
                prec = np.mean(batch_precs)
                rec = np.mean(batch_recs)
                f1 = np.mean(batch_f1s)
            else:
                accuracy, prec, rec, f1 = self.calculate_accuracy(
                        pred_prob_node[self.data.train_mask],
                        self.data.y[self.data.train_mask])
                loss = loss.item()

            self.log(epoch, loss, accuracy, prec, rec, f1,
                     split='train')
            self.train_losses.append(loss)
            self.train_accs.append(accuracy)
            self.train_f1s.append(f1)

            val_loss = self.evaluate_on_valid(model, epoch)

            self.plot_learning_curve()
            if self.early_stopping:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    break
            print('\n')

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

        results = precision_recall_fscore_support(
                target.cpu().numpy(), pred_node.cpu().numpy(), average='micro')
        prec, rec, f1, _ = results

        return acc, prec, rec, f1

    def log(self, epoch, loss, accuracy, prec, rec, f1, split='val'):

        if self.verbose:
            print("Epoch {} ({})\tLoss: {:.4f}\tA: {:.4f}\tP: {:.4f}\t"
                  "R: {:.4f}\tF1: {:.4f}".format(epoch, split, loss, accuracy,
                                                 prec, rec, f1), flush=True)

    def plot_learning_curve(self,shadow=False):
        '''
        Result png figures are saved in the log directory.
        '''
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        privacy = 'None'

        if shadow:
            fig.suptitle('Shadow Model Learning Curve ({}, % data {}, epsilon {})'.format(
            self.dataset, self.mia_subsample_rate, privacy))
            epochs = list(range(len(self.shadow_train_losses)))
            ax1.plot(epochs, self.shadow_train_losses, 'o-', markersize=2, color='b',
                    label='Train')
            ax1.plot(epochs, self.shadow_valid_losses, 'o-', markersize=2, color='c',
                    label='Validation')
            ax1.set(ylabel='Loss')

            ax2.plot(epochs, self.shadow_train_accs, 'o-', markersize=2, color='b',
                    label='Train')
            ax2.plot(epochs, self.shadow_valid_accs, 'o-', markersize=2, color='c',
                    label='Validation')
            ax2.set(xlabel='Epoch', ylabel='Accuracy')
            ax1.legend()
            plt.savefig(os.path.join(self.time_dir, 'shadow_learning_curve.png'))
            plt.close()
        else:
            fig.suptitle('Model Learning Curve ({}, % data {}, epsilon {})'.format(
                self.dataset, 1, privacy))
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
            plt.savefig(os.path.join(self.time_dir, 'learning_curve.png'))
            plt.close()

    def evaluate_on_valid(self, model, epoch, shadow=None):

        model.eval()

        with torch.no_grad():
            if shadow:
                pred_prob_node = model(self.shadow_data)
                loss = self.shadow_loss(pred_prob_node[self.shadow_data.val_mask],
                                self.shadow_data.y[self.shadow_data.val_mask])

                accuracy, prec, rec, f1 = self.calculate_accuracy(
                        pred_prob_node[self.shadow_data.val_mask],
                        self.shadow_data.y[self.shadow_data.val_mask])

                self.log(epoch, loss, accuracy, prec, rec, f1, split='shadow val')
                self.shadow_valid_losses.append(loss.item())
                self.shadow_valid_accs.append(accuracy)
                self.shadow_valid_f1s.append(f1)
            else:
                pred_prob_node = model(self.data)
                loss = self.loss(pred_prob_node[self.data.val_mask],
                                self.data.y[self.data.val_mask])
    
                accuracy, prec, rec, f1 = self.calculate_accuracy(
                        pred_prob_node[self.data.val_mask],
                        self.data.y[self.data.val_mask])

                self.log(epoch, loss, accuracy, prec, rec, f1, split='val')
                self.valid_losses.append(loss.item())
                self.valid_accs.append(accuracy)
                self.valid_f1s.append(f1)
        return loss.item()

    def evaluate_on_test(self,shadow=None):
        if shadow:
            model = self.shadow_model
            model.eval()
            confusion_matrix = torch.zeros(self.num_classes, self.num_classes,
                                        dtype=torch.long)

            test_size = self.shadow_data.y[self.shadow_data.test_mask].shape
            with torch.no_grad():
                pred_prob_node = model(self.shadow_data)
                loss = self.shadow_loss(pred_prob_node[self.shadow_data.test_mask],
                                self.shadow_data.y[self.shadow_data.test_mask])
                preds = pred_prob_node.max(dim=1)[1]

                accuracy, prec, rec, f1 = self.calculate_accuracy(
                        pred_prob_node[self.shadow_data.test_mask],
                        self.shadow_data.y[self.shadow_data.test_mask])

                for t, p in zip(self.shadow_data.y[self.shadow_data.test_mask],
                                preds[self.shadow_data.test_mask]):
                    if p.long() in range(self.num_classes):
                        confusion_matrix[t.long(), p.long()] += 1
                    else:
                        confusion_matrix[t.long(), -1] += 1

                confusion_out = confusion_matrix.data.cpu().numpy()
                np.savetxt(os.path.join(self.time_dir, 'shadow_confusion_matrix.csv'),
                        confusion_out, delimiter=',', fmt='% 4d')

                # Output predictions
                print("Preparing predictions file...\n")
                pred_filename = os.path.join(self.time_dir,
                                            f'shadow_preds_seed{self.seed}.csv')
                with open(pred_filename, 'w') as pred_f:
                    pred_f.write("Pred,Y\n")
                    for idx in range(preds[self.shadow_data.test_mask].shape[0]):
                        pred_f.write(f"{preds[self.shadow_data.test_mask][idx]},{self.shadow_data.y[self.shadow_data.test_mask][idx]}\n")

                self.shadow_test_loss = loss.item()
                self.shadow_test_acc = accuracy
                self.shadow_test_f1 = f1

                print("Test set results\tLoss: {:.4f}\tNode Accuracy: {:.4f}\t"
                    "Precision: {:.4f}\tRecall: {:.4f}\tF1: {:.4f}".format(
                        loss.item(), accuracy, prec, rec, f1))
            return loss.item(), accuracy, prec, rec, f1
        else: 
            model = self.model
            model.eval()
            confusion_matrix = torch.zeros(self.num_classes, self.num_classes,
                                        dtype=torch.long)

            test_size = self.data.y[self.data.test_mask].shape
            if self.random_baseline:
                rand_preds = torch.randint(0, self.data.y.unique().max().item(), (test_size[0],)).to(self.device)
                accuracy_rand, prec_rand, rec_rand, f1_rand = self.calculate_accuracy(
                        rand_preds,
                        self.data.y[self.data.test_mask], rand_maj_baseline=True)
                print(f"Random baseline results (test F1, test acc): {f1_rand}, {accuracy_rand}")
            if self.majority_baseline:
                majority = self.data.y[self.data.train_mask].bincount().argmax().item()
                majority_preds = torch.ones(test_size, device=self.device) * majority
                accuracy_maj, prec_maj, rec_maj, f1_maj = self.calculate_accuracy(
                        majority_preds,
                        self.data.y[self.data.test_mask], rand_maj_baseline=True)
                print(f"Majority baseline results (test F1, test acc): {f1_maj}, {accuracy_maj}")

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
        if shadow:
            filepath = os.path.join(self.time_dir, 'shadow_results.csv')
            best_val_loss = best_score
            epoch = self.shadow_valid_losses.index(best_val_loss)
            best_val_acc = self.shadow_valid_accs[epoch]
            best_val_f1 = self.shadow_valid_f1s[epoch]

            with open(filepath, 'w') as out_f:
                out_f.write('BestValidLoss,BestValidAcc,BestValidF1,'
                            'BestValidEpoch,TestLoss,TestAcc,TestF1,'
                            'NumTrainableParams,NumNodes(per_sg),NumEdges(per_sg),ModelConfig\n')
                out_f.write(f'{best_val_loss:.4f},{best_val_acc:.4f},'
                            f'{best_val_f1:.4f},{epoch},{self.shadow_test_loss:.4f},'
                            f'{self.shadow_test_acc:.4f},{self.shadow_test_f1:.4f},'
                            f'{self.shadow_trainable_params},{self.shadow_num_nodes},'
                            f'{self.shadow_num_edges},'
                            f'{self.shadow_model_name}\n')
        else:
            filepath = os.path.join(self.time_dir, 'results.csv')
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

    def get_shadow_data(self):
        model = self.shadow_model
        model.eval()
        with torch.no_grad():
            shadow_train_neg = model(self.shadow_data)[self.shadow_data.test_mask]
            shadow_train_pos = model(self.shadow_data)[self.shadow_data.train_mask]
            shadow_train_y = [1]*shadow_train_pos.shape[0]+[0]*shadow_train_neg.shape[0]
            shadow_train_y = torch.tensor(shadow_train_y,dtype=torch.float).to(self.device)
            shadow_train_x = torch.cat([shadow_train_pos,shadow_train_neg],dim=0)
            shadow_train_x = torch.softmax(shadow_train_x,dim=1)
            indices = torch.randperm(shadow_train_x.size(0))
            shadow_train_x = shadow_train_x[indices]
            shadow_train_y = shadow_train_y[indices]
            # shadow_train_y2 = (shadow_train_y-.5)*2
        
        model = self.model
        model.eval()
        with torch.no_grad():
            shadow_test_neg = model(self.data)[self.data.test_mask]
            shadow_test_pos = model(self.data)[self.data.train_mask]
            shadow_test_y = [1]*shadow_test_pos.shape[0]+[0]*shadow_test_neg.shape[0]
            shadow_test_y = torch.tensor(shadow_test_y,dtype=torch.float).to(self.device)
            shadow_test_x = torch.cat([shadow_test_pos,shadow_test_neg],dim=0)
            shadow_test_x = torch.softmax(shadow_test_x,dim=1)
            indices = torch.randperm(shadow_test_x.size(0))
            shadow_test_x = shadow_test_x[indices]
            shadow_test_y = shadow_test_y[indices]
            # shadow_test_y2 = (shadow_test_y-.5)*2
        return shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y

    def shadow_MIA_mlp(self):
        shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y = self.get_shadow_data()

        ss_dict = {
            "chanels":self.hidden_dim,
            "dropout":self.dropout,
            "optimizer":self.optim_type,
            "num_features":self.num_classes,
            "num_classes":1,
            "num_epochs":100
        }
        mia_mlp = Shadow_MIA_mlp(shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y,ss_dict)
        
    def shadow_MIA_svm(self):
        shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y = self.get_shadow_data()

        ss_dict = {
            "kernel":"rbf",#linear rbf poly sigmoid
            "random_state":self.seed
        }
        mia_svm = Shadow_MIA_svm(shadow_train_x.detach().cpu().numpy(),shadow_train_y.detach().cpu().numpy(),shadow_test_x.detach().cpu().numpy(),shadow_test_y.detach().cpu().numpy(),ss_dict)

    def shadow_MIA_ranfor(self):
        shadow_train_x,shadow_train_y,shadow_test_x,shadow_test_y = self.get_shadow_data()

        ss_dict = {
            "criterion":"gini",#'gini', 'entropy', 'log_loss'
            "random_state":self.seed,
            "n_estimators":100
        }
        mia_ranfor = Shadow_MIA_ranfor(shadow_train_x.detach().cpu().numpy(),shadow_train_y.detach().cpu().numpy(),shadow_test_x.detach().cpu().numpy(),shadow_test_y.detach().cpu().numpy(),ss_dict)

    def confidence_MIA(self):
        confidences = [0.50,0.55,0.60,0.65,0.70,0.75]
        res = {0.50:[],0.55:[],0.60:[],0.65:[],0.70:[],0.75:[]}
        _,_,shadow_test_x,shadow_test_y = self.get_shadow_data()
        shadow_test_x = shadow_test_x.detach().cpu().numpy()
        shadow_test_y = shadow_test_y.detach().cpu().numpy().reshape(-1)
        for i in range(shadow_test_x.shape[0]):
            item = shadow_test_x[i]
            for j in confidences:
                # temp = math.log(j)
                if np.count_nonzero(item>j) > 0 :
                    res[j].append(1.)
                else:
                    res[j].append(0.)
        for j in confidences:
            print(f"\nConfidence MIA attack with threshold {j}:\n")
            print(metrics.classification_report(shadow_test_y,res[j],labels=range(2)))
            # print(shadow_test_y,res[j])

def main():
    now = time.time()

    # Setting the seed
    ss = Settings()
    ss.make_dirs()
    torch.manual_seed(ss.args.seed)
    torch.cuda.manual_seed(ss.args.seed)
    np.random.seed(ss.args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

    model = GCNModel(ss)
    best_score = model.train()
    test_loss, test_acc, test_prec, test_rec, test_f1 = model.evaluate_on_test()
    model.output_results(best_score)
    print(f"Test score: {test_loss:.4f} with accuracy {test_acc:.4f} and f1 {test_f1:.4f}")
    with open(os.path.join(ss.root_dir, 'adam_hyperparams.csv'), 'a') as f: 
        f.write(f"{ss.args.dataset},{ss.args.noise_scale},{ss.args.learning_rate},{test_f1:.4f}\n")
    
    if ss.args.mia == 'shadow':
        best_score = model.shadow_train()
        test_loss, test_acc, test_prec, test_rec, test_f1 = model.evaluate_on_test(shadow=True)
        model.output_results(best_score,shadow=True)
        print(f"Shadow Model Test score: {test_loss:.4f} with accuracy {test_acc:.4f} and f1 {test_f1:.4f}")
        model.shadow_MIA_mlp()
        model.shadow_MIA_svm()
        model.shadow_MIA_ranfor()
        model.confidence_MIA()
        with open(os.path.join(ss.root_dir, 'adam_hyperparams.csv'), 'a') as f: 
            f.write(f"{ss.args.dataset},{ss.args.noise_scale},{ss.args.learning_rate},{test_f1:.4f}\n")

    then = time.time()
    runtime = then - now
    print(f"\n--- Script completed in {runtime} seconds ---\n")

if __name__ == '__main__':
    main()
