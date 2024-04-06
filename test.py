import torch_geometric,torch_sparse,torch_cluster
from torch.nn import functional as F
from model import two_layer_GCN
from torch_geometric import data,datasets
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
path = '/mnt/ssd1/sunyifan/WorkStation/dpuf/data'
dataset = Planetoid(path, name='Cora')
datas = dataset[0]
from dataset import *
# loader = OccuranceSampler(data=datas,
#                           k=3,
#                           depth=2,
#                           device='cpu',
#                           sampler_batchsize=.7)
# loader = ClusterSampler(data=datas,
#                         num_parts=100,
#                         batch_size=1)
# loader = SaintSampler(data=datas,
#                       type='saint_rw',
#                       batch_size=4,
#                       num_steps=100,
#                       sample_coverage=100,
#                       walk_length=4)
loader = NeighborReplaceSampler(data=datas,
                                batch_size=1,
                                layers=2)
sampled_data = next(iter(loader))
print(sampled_data)
# print(sampled_data.batch_size)
print(len(loader))
ss_dict = {
        "k_layer":2,
        "chanels":256,
        "dropout":.2,
        "activation":'relu',
        "optimizer":'adam',
        "num_features":dataset.num_node_features,
        "num_classes":dataset.num_classes
    }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn = two_layer_GCN(ss=ss_dict).to(device)
optimizer = torch.optim.Adam(gcn.parameters(), lr=0.001)

def train():
    gcn.train()
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = gcn(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return total_loss / total_examples


for epoch in range(1, 5):
    loss = train()
    # val_acc = test(val_loader)
    # test_acc = test(test_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

