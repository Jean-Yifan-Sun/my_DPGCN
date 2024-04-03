import torch_geometric,torch_sparse,torch_cluster
from torch_geometric import data,datasets
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
path = '/mnt/ssd1/sunyifan/WorkStation/dpuf/data'
datas = Planetoid(path, name='Cora')[0]
from dataset import *
loader = OccuranceSampler(data=datas,
                          k=3,
                          depth=2,
                          device='cpu',
                          sampler_batchsize=.7)
sampled_data = next(iter(loader))
print(sampled_data)
print(sampled_data.batch_size)
print(len(loader))