import torch
import numpy as np

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
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
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

def random_graph_split(data, n_subgraphs=10):
    '''
    Divide a graph into subgraphs using a random split:
        For n subsets, place nodes into subsets then for each node pair in
        the subgraph, check whether an edge exists in the original graph
    Note: Only the training portion of the graph is considered, val/test
          portions can be used as before with the original 'data' object
          with data.val_mask and data.test_mask
    '''
    full_len = data.x.shape[0]
    sample_tensor = torch.arange(full_len)[data.train_mask]
    sample_tensor = sample_tensor[torch.randperm(sample_tensor.size()[0])]

    batch_indexes = np.array_split(sample_tensor, n_subgraphs)

    batch_masks = []
    for idx_list in batch_indexes:
        batch_mask = torch.zeros(full_len, dtype=torch.bool)
        batch_mask[idx_list] = True
        batch_masks.append(batch_mask)

    return batch_masks

