import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from torch_geometric.utils import *

'''
Early stopping for the main network.
'''

class EarlyStopping(object):

    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            
            if self.counter >= self.patience:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def get_train_edge_count(data, split_graph=False):
    '''
    Counts the number of edges used only in the training subset of the graph.
    '''
    if split_graph:
        train_nodes = data.batch_masks[0].nonzero().squeeze()
    else:
        train_nodes = data.train_mask.nonzero().squeeze()
    test_nodes = data.test_mask.nonzero().squeeze()
    edges = data.edge_index

    num_train_edges = 0
    num_test_edges = 0
    for idx in range(edges.shape[1]):
        edge = edges[:, idx]
        if edge[0] in train_nodes and edge[1] in train_nodes:
            num_train_edges += 1
        elif edge[0] in test_nodes and edge[1] in test_nodes:
            num_test_edges += 1

    return num_train_edges, num_test_edges

def random_graph_split(data, n_subgraphs=10,device='cpu'):
    '''
    Divide a graph into subgraphs using a random split:
        For n subsets, place nodes into subsets then for each node pair in
        the subgraph, check whether an edge exists in the original graph
    Note: Only the training portion of the graph is considered, val/test
          portions can be used as before with the original 'data' object
          with data.val_mask and data.test_mask
    '''
    full_len = data.x.shape[0]
    sample_tensor = torch.arange(full_len).to(device)
    sample_tensor = sample_tensor[data.train_mask]
    sample_tensor = sample_tensor[torch.randperm(sample_tensor.size()[0])]

    batch_indexes = np.array_split(sample_tensor, n_subgraphs)

    batch_masks = []
    for idx_list in batch_indexes:
        batch_mask = torch.zeros(full_len, dtype=torch.bool)
        batch_mask[idx_list] = True
        batch_masks.append(batch_mask)

    return batch_masks

'''
Graph subsampling.
'''
def subsample_graph(data, rate=0.1, maintain_class_dists=True,
                    every_class_present=True):
    '''
    Given a data object, sample the graph based on the provided rate
    (as a percent)
    every_class_present: making sure that all classes are present in the
                         subsample (only if class distributions are not
                         maintained)
    '''
    if not 1 > rate > 0:
        raise Exception("Rate of subsampling graph must be in interval (0,1).")

    if maintain_class_dists:
        # class_counts = torch.bincount(data.y[data.train_mask])
        # new_class_counts = torch.floor_divide(class_counts, 1/rate).long()
        # all_new_class_indexes = []
        # for cls_val in range(class_counts.shape[0]):
        #     full_class_indexes = (data.y == cls_val).nonzero().squeeze()
        #     train_class_indexes = torch.tensor(np.intersect1d(full_class_indexes.numpy(), data.train_mask.nonzero().squeeze().numpy()))
        #     sample_idx_tensor = torch.randperm(
        #             train_class_indexes.shape[0])[:new_class_counts[cls_val]]
        #     new_class_indexes = train_class_indexes[sample_idx_tensor]
        #     all_new_class_indexes.append(new_class_indexes)
        # sample_tensor = torch.cat(all_new_class_indexes)
        sample_tensor = subsample_mask(data=data,mask=data.train_mask,rate=rate)
        # val_idxs = subsample_mask(data=data,mask=data.val_mask,rate=rate)
        # test_idxs = subsample_mask(data=data,mask=data.test_mask,rate=rate)
    else:
        if every_class_present:
            class_counts = torch.bincount(data.y[data.train_mask])
            new_class_counts = torch.floor_divide(class_counts, 1/rate).long()
            idx_from_every_class = []
            for cls_val in range(class_counts.shape[0]):
                full_class_indexes = (data.y == cls_val).nonzero().squeeze()
                train_class_indexes = torch.tensor(np.intersect1d(full_class_indexes.numpy(), data.train_mask.nonzero().squeeze().numpy()))
                sample_idx_tensor = torch.randperm(
                        train_class_indexes.shape[0]
                        )[:new_class_counts[cls_val]]
                new_class_indexes = train_class_indexes[sample_idx_tensor]
                idx_from_every_class.append(new_class_indexes[0].item())

            full_len = data.x[data.train_mask].shape[0]
            sample_len = int(full_len * rate)
            sample_tensor = torch.randperm(full_len)[:sample_len]

            # Adding indexes from each class to the sample tensor:
            sample_tensor = torch.cat(
                    (sample_tensor,
                     torch.tensor(idx_from_every_class))
                    ).unique()
            val_sample = torch.randperm(data.x[data.val_mask].shape[0])[:int(data.x[data.val_mask].shape[0]*rate)]
            test_sample = torch.randperm(data.x[data.test_mask].shape[0])[:int(data.x[data.test_mask].shape[0]*rate)]
            val_idxs = data.val_mask.nonzero().squeeze()[val_sample]
            test_idxs = data.test_mask.nonzero().squeeze()[test_sample]
        else:
            full_len = data.x[data.train_mask].shape[0]
            sample_len = int(full_len * rate)
            sample_tensor = torch.randperm(full_len)[:sample_len]
            val_sample = torch.randperm(data.x[data.val_mask].shape[0])[:int(data.x[data.val_mask].shape[0]*rate)]
            test_sample = torch.randperm(data.x[data.test_mask].shape[0])[:int(data.x[data.test_mask].shape[0]*rate)]
            val_idxs = data.val_mask.nonzero().squeeze()[val_sample]
            test_idxs = data.test_mask.nonzero().squeeze()[test_sample]
            

    val_sample = torch.randperm(data.x[data.val_mask].shape[0])[:int(data.x[data.val_mask].shape[0]*rate)]
    test_sample = torch.randperm(data.x[data.test_mask].shape[0])[:int(data.x[data.test_mask].shape[0]*rate)]
    
    val_idxs = data.val_mask.nonzero().squeeze()[val_sample]
    test_idxs = data.test_mask.nonzero().squeeze()[test_sample]
    sample_tensor = torch.cat((sample_tensor, val_idxs, test_idxs))

    data.x = data.x[sample_tensor]
    data.train_mask = data.train_mask[sample_tensor]
    data.val_mask = data.val_mask[sample_tensor]
    data.test_mask = data.test_mask[sample_tensor]
    data.y = data.y[sample_tensor]

    old_to_new_node_idx = {old_idx.item(): new_idx
                           for new_idx, old_idx in enumerate(sample_tensor)}

    # Updating adjacency matrix
    new_edge_index_indexes = []
    for idx in tqdm(range(data.edge_index.shape[1])):
        if (data.edge_index[0][idx] in sample_tensor) and \
           (data.edge_index[1][idx] in sample_tensor):
            new_edge_index_indexes.append(idx)

    new_edge_idx_temp = torch.index_select(
            data.edge_index, 1, torch.tensor(new_edge_index_indexes)
            )
    new_edge_idx_0 = [old_to_new_node_idx[new_edge_idx_temp[0][a].item()]
                      for a in range(new_edge_idx_temp.shape[1])]
    new_edge_idx_1 = [old_to_new_node_idx[new_edge_idx_temp[1][a].item()]
                      for a in range(new_edge_idx_temp.shape[1])]
    data.edge_index = torch.stack((torch.tensor(new_edge_idx_0),
                                   torch.tensor(new_edge_idx_1)))

def subsample_mask(data,mask,rate):
    class_counts = torch.bincount(data.y[mask])
    new_class_counts = torch.floor_divide(class_counts, 1/rate).long()
    all_new_class_indexes = []
    for cls_val in range(class_counts.shape[0]):
        full_class_indexes = (data.y == cls_val).nonzero().squeeze()
        train_class_indexes = torch.tensor(np.intersect1d(full_class_indexes.numpy(), mask.nonzero().squeeze().numpy()))
        sample_idx_tensor = torch.randperm(
                train_class_indexes.shape[0])[:new_class_counts[cls_val]]
        new_class_indexes = train_class_indexes[sample_idx_tensor]
        all_new_class_indexes.append(new_class_indexes)
    sample_tensor = torch.cat(all_new_class_indexes)
    return sample_tensor

    
def subsample_graph_all(data,rate):    
    class_counts = torch.bincount(data.y)
    new_class_counts = torch.floor_divide(class_counts, 1/rate).long()
    all_new_class_indexes = []
    for cls_val in range(class_counts.shape[0]):
        full_class_indexes = (data.y == cls_val).nonzero().squeeze()
        train_class_indexes = full_class_indexes
        sample_idx_tensor = torch.randperm(
                train_class_indexes.shape[0])[:new_class_counts[cls_val]]
        new_class_indexes = train_class_indexes[sample_idx_tensor]
        all_new_class_indexes.append(new_class_indexes)
    sample_tensor = torch.cat(all_new_class_indexes)
    data.x = data.x[sample_tensor]
    data.train_mask = data.train_mask[sample_tensor]
    data.val_mask = data.val_mask[sample_tensor]
    data.test_mask = data.test_mask[sample_tensor]
    data.y = data.y[sample_tensor]
    old_to_new_node_idx = {old_idx.item(): new_idx
                           for new_idx, old_idx in enumerate(sample_tensor)}
    

    # Updating adjacency matrix
    new_edge_index_indexes = []
    for idx in tqdm(range(data.edge_index.shape[1])):
        if (data.edge_index[0][idx] in sample_tensor) and \
           (data.edge_index[1][idx] in sample_tensor):
            new_edge_index_indexes.append(idx)

    new_edge_idx_temp = torch.index_select(
            data.edge_index, 1, torch.tensor(new_edge_index_indexes)
            )
    new_edge_idx_0 = [old_to_new_node_idx[new_edge_idx_temp[0][a].item()]
                      for a in range(new_edge_idx_temp.shape[1])]
    new_edge_idx_1 = [old_to_new_node_idx[new_edge_idx_temp[1][a].item()]
                      for a in range(new_edge_idx_temp.shape[1])]
    data.edge_index = torch.stack((torch.tensor(new_edge_idx_0),
                                   torch.tensor(new_edge_idx_1)))
    
def subsample_graph_pyg(data,rate):
    class_counts = torch.bincount(data.y)
    new_class_counts = torch.floor_divide(class_counts, 1/rate).long()
    all_new_class_indexes = []
    for cls_val in range(class_counts.shape[0]):
        full_class_indexes = (data.y == cls_val).nonzero().squeeze()
        train_class_indexes = full_class_indexes
        sample_idx_tensor = torch.randperm(train_class_indexes.shape[0])[:new_class_counts[cls_val]]
        new_class_indexes = train_class_indexes[sample_idx_tensor]
        all_new_class_indexes.append(new_class_indexes)
    sample_tensor = torch.cat(all_new_class_indexes)
    return data.subgraph(sample_tensor) 

def subsample_graph_both_pyg(data,rate):
    """
    divide two subest of graph from the original graph that matches the dist in original graph and make sure they dont share any common node
    """
    assert rate<=.5 
    class_counts = torch.bincount(data.y)
    new_class_counts = torch.floor_divide(class_counts, 1/rate).long()
    all_new_class_indexes1 = []
    all_new_class_indexes2 = []
    for cls_val in range(class_counts.shape[0]):
        full_class_indexes = (data.y == cls_val).nonzero().squeeze()
        train_class_indexes = full_class_indexes
        rand_idx = torch.randperm(train_class_indexes.shape[0])

        sample_idx_tensor1 = rand_idx[:new_class_counts[cls_val]]
        new_class_indexes1 = train_class_indexes[sample_idx_tensor1]
        all_new_class_indexes1.append(new_class_indexes1)

        sample_idx_tensor2 = rand_idx[-new_class_counts[cls_val]:]
        new_class_indexes2 = train_class_indexes[sample_idx_tensor2]
        all_new_class_indexes2.append(new_class_indexes2)

    sample_tensor1 = torch.cat(all_new_class_indexes1)
    sample_tensor2 = torch.cat(all_new_class_indexes2)
    return data.subgraph(sample_tensor1),data.subgraph(sample_tensor2)

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

def subsample_mask_graph(data,mask,rate):
    class_counts = torch.bincount(data.y[mask])
    new_class_counts = torch.floor_divide(class_counts, 1/rate).long()
    all_new_class_indexes = []
    all_out_class_indexes = []
    for cls_val in range(class_counts.shape[0]):
        full_class_indexes = (data.y == cls_val).nonzero().squeeze()
        train_class_indexes = torch.tensor(np.intersect1d(full_class_indexes.numpy(), mask.nonzero().squeeze().numpy()))
        sample_in_out_tensor = torch.randperm(train_class_indexes.shape[0])
        sample_idx_tensor = sample_in_out_tensor[:new_class_counts[cls_val]]
        sample_out_tensor = sample_in_out_tensor[-new_class_counts[cls_val]:]
        new_class_indexes = train_class_indexes[sample_idx_tensor]
        out_class_indexes = train_class_indexes[sample_out_tensor]
        all_new_class_indexes.append(new_class_indexes)
        all_out_class_indexes.append(out_class_indexes)
    sample_tensor = torch.cat(all_new_class_indexes)
    out_tensor = torch.cat(all_out_class_indexes)
    res = []
    for i in [sample_tensor,out_tensor]:
        new_mask = torch.zeros_like(mask,dtype=bool)
        for j in i:
            new_mask[j] = True
        res.append(new_mask)
    
    return res

def subsample_mask_graph_full(data,rate):
    [train_mask_1,train_mask_2] = subsample_mask_graph(data=data,mask=data.train_mask,rate=rate)
    [test_mask_1,test_mask_2] = subsample_mask_graph(data=data,mask=data.test_mask,rate=rate)
    [val_mask_1,val_mask_2] = subsample_mask_graph(data=data,mask=data.val_mask,rate=rate)
    new_data = data.clone()
    data.train_mask = train_mask_1
    data.test_mask = test_mask_1
    data.val_mask = val_mask_1
    new_data.train_mask = train_mask_2
    new_data.test_mask = test_mask_2
    new_data.val_mask = val_mask_2
    return data,new_data

def matain_same_split_dist(data1,data2):
    class_counts = torch.bincount(data1.y[data1.train_mask])
    all_new_class_indexes = []
    for cls_val in range(class_counts.shape[0]):
        full_class_indexes = (data2.y == cls_val).nonzero().squeeze()
        sample_tensor = torch.randperm(full_class_indexes.shape[0])
        new_class_indexes = full_class_indexes[sample_tensor]
        new_class_indexes = new_class_indexes[:class_counts[cls_val]]
        all_new_class_indexes.append(new_class_indexes)
    sample_tensor = torch.cat(all_new_class_indexes)
    new_mask = torch.zeros_like(data1.train_mask,dtype=bool)
    for i in sample_tensor:
        new_mask[i] = True
    data2.train_mask = new_mask

    class_counts = torch.bincount(data1.y[data1.val_mask])
    all_new_class_indexes = []
    for cls_val in range(class_counts.shape[0]):
        full_class_indexes = (data2.y == cls_val).nonzero().squeeze()
        sample_tensor = torch.randperm(full_class_indexes.shape[0])
        new_class_indexes = full_class_indexes[sample_tensor]
        new_class_indexes = new_class_indexes[:class_counts[cls_val]]
        all_new_class_indexes.append(new_class_indexes)
    sample_tensor = torch.cat(all_new_class_indexes)
    new_mask = torch.zeros_like(data1.trainval_mask_mask,dtype=bool)
    for i in sample_tensor:
        new_mask[i] = True
    data2.val_mask = new_mask    

def get_neighbors(edge_index,node_idx):
    "给出给定节点为起点的所有邻居节点的idx"
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx=node_idx,
                                                            edge_index=edge_index,
                                                            num_hops=1)
    subset = subset.tolist()
    subset.remove(node_idx)
    
    if len(subset) == 0:
        return False
    return torch.Tensor(subset)
    
def sample_neighbors_with_p(neighbors,p):
    "用p为概率对neighbors里的节点进行抽样"
    p_tensor = torch.ones_like(neighbors) * p
    mask = torch.bernoulli(p_tensor).bool()
    return neighbors[mask]

def dfs_with_depth(node,neighbors_dict,depth):
    "从node开始按照depth使用dfs搜寻"
    all_nodes = [node]
    for i in range(depth):
        for j in all_nodes:
            all_nodes.extend(neighbors_dict[j])