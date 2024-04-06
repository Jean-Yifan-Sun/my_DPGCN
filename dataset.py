from torch_geometric import datasets,data
import torch,random,os,torch_sparse
from torch_geometric import transforms as T
from torch_geometric.sampler import *
from torch_geometric.loader import *
from utils import *

current_path = os.path.dirname(__file__)
parent_directory = os.path.abspath(os.path.join(current_path, os.pardir))
ROOT = os.path.join(parent_directory,'data')

dataset_dic_full = {"GNNBenchmark":["PATTERN", "CLUSTER", "MNIST", "CIFAR10", "TSP", "CSL"],
                   "TU":["MUTAG","ENZYMES","PROTEINS","COLLAB","IMDB-BINARY","REDDIT-BINARY"],
                   "Planetoid":["Cora","CiteSeer","PubMed"],
                   "KarateClub":[],
                   "NELL":[],
                   "CitationFull":["Cora", "Cora_ML", "CiteSeer", "DBLP", "PubMed"],
                   "Coauthor":["CS", "Physics"],
                   "Amazon":["Computers","Photo"],
                   "PPI":[],
                   "Reddit":[],
                   "Reddit2":[],
                   "Flickr":[],
                   "Yelp":[],
                   "AmazonProducts":[],
                   "QM7b":[],
                   "QM9":[],
                   "MD17":[],
                   "ZINC":[],
                   "AQSOL":[],
                   "MoleculeNet":["ESOL", "FreeSolv", "Lipo", "PCBA", "MUV", "HIV", "BACE", "BBBP", "Tox21", "ToxCast", "SIDER", "ClinTox"],
                   "Entities":["AIFB", "MUTAG", "BGS", "AM"],
                   "RelLinkPredDataset":[],
                   "GEDDataset":["AIDS700nef", "LINUX", "ALKANE", "IMDBMulti"],
                   "AttributedGraphDataset":["Wiki", "Cora" "CiteSeer", "PubMed", "BlogCatalog", "PPI", "Flickr", "Facebook", "Twitter", "TWeibo", "MAG"],
                   "MNISTSuperpixels":[],
                   "FAUST":[],
                   "DynamicFAUST":[],
                   "ShapeNet":[],
                   "ModelNet":["10","40"],
                   "CoMA":[],
                   "SHREC2016":[],
                   "TOSCA":[],
                   "PCPNetDataset":[],
                   "S3DIS":[],
                   "GeometricShapes":[],
                   "BitcoinOTC":[],
                   "GDELTLite":[],
                   "ICEWS18":[],
                   "GDELT":[],
                   "WILLOWObjectClass":[],
                   "PascalVOCKeypoints":[],
                   "PascalPF":[],
                   "SNAPDataset":[],
                   "SuiteSparseMatrixCollection":[],
                   "WordNet18":[],
                   "WordNet18RR":[],
                   "FB15k_237":[],
                   "WikiCS":[],
                   "WebKB":["Cornell", "Texas", "Wisconsin"],
                   "WikipediaNetwork":[],
                   "HeterophilousGraphDataset":["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"],
                   "Actor":[],
                   "UPFD":[],
                   "GitHub":[],
                   "FacebookPagePage":[],
                   "LastFMAsia":[],
                   "DeezerEurope":[],
                   "GemsecDeezer":[],
                   "Twitch":["DE", "EN", "ES", "FR", "PT", "RU"],
                   "Airports":[],
                   "LRGBDataset":["PascalVOC-SP", "COCO-SP", "PCQM-Contact", "Peptides-func", "Peptides-struct"],
                   "MalNetTiny":[],
                   "OMDB":[],
                   "PolBlogs":[],
                   "EmailEUCore":[],
                   "LINKXDataset":[],
                   "EllipticBitcoinDataset":[],
                   "EllipticBitcoinTemporalDataset":[],
                   "DGraphFin":[],
                   "HydroNet":[],
                   "AirfRANS":[],
                   "JODIEDataset":["Reddit", "Wikipedia", "MOOC", "LastFM"]
                   }

def get_dataset(cls,name=None,num_val=None,num_test=None):
    dataset_dic_demo = {"GNNBenchmark":["PATTERN", "CLUSTER", "MNIST", "CIFAR10", "TSP", "CSL"],
                        "TU":["MUTAG","ENZYMES","PROTEINS","COLLAB","IMDB-BINARY","REDDIT-BINARY"],
                        "Planetoid":["Cora","CiteSeer","PubMed"],
                        "KarateClub":[],
                        "NELL":[],
                        "Reddit":[],
                        "Flickr":[],
                        "GitHub":[],
                        "LastFMAsia":[],
                        "Twitch":["RU", "PT","DE","FR","ES","EN"],
                        "CitationFull":["Cora", "Cora_ML", "CiteSeer", "DBLP", "PubMed"],
                        "Coauthor":["CS", "Physics"],
                        "Amazon":["Computers","Photo"]
                        }    
    dataset_keys = list(dataset_dic_demo.keys())

    # transform = T.RandomNodeSplit(split="train_rest",num_val=num_val,num_test=num_test)
    transform = None
    if not cls in dataset_keys:
        raise NameError(f"No matching name for dataset class {cls}")
    if cls == "GNNBenchmark":
        if name in ["PATTERN", "CLUSTER", "MNIST", "CIFAR10", "TSP", "CSL"]:
            temp = datasets.GNNBenchmarkDataset(root=ROOT, name=name,transform=transform)
        else:
            raise NameError(f"No matching name for dataset {name}")
    elif cls == "TU":
        if name in ["MUTAG","ENZYMES","PROTEINS","COLLAB","IMDB-BINARY","REDDIT-BINARY"]:
            temp = datasets.TUDataset(root=ROOT,name=name,transform=transform)
        else:
            raise NameError(f"No matching name for dataset {name}")
    elif cls == "Planetoid":
        if name in ["Cora","CiteSeer","PubMed"]:
            temp = datasets.Planetoid(root=ROOT,name=name,transform=transform)
        else:
            raise NameError(f"No matching name for dataset {name}")
    elif cls == "KarateClub":
        temp = datasets.KarateClub()
    elif cls == "NELL":
        temp = datasets.NELL(root=ROOT,transform=transform)
    elif cls == "CitationFull":
        if name in ["Cora", "Cora_ML", "CiteSeer", "DBLP", "PubMed"]:
            temp = datasets.CitationFull(root=ROOT,name=name,transform=transform)
        else:
            raise NameError(f"No matching name for dataset {name}")
    elif cls == "Coauthor":
        if name in ["CS", "Physics"]:
            temp = datasets.Coauthor(root=ROOT,name=name,transform=transform)
        else:
            raise NameError(f"No matching name for dataset {name}")
    elif cls == "Amazon":
        if name in ["Computers","Photo"]:
            temp = datasets.Amazon(root=ROOT,name=name,transform=transform)
    elif cls == "Reddit":
        temp = datasets.Reddit(root=os.path.join(ROOT,'Reddit'))
    elif cls == "Flickr":
        temp = datasets.Flickr(root=os.path.join(ROOT,'Flickr'))
    elif cls == "GitHub":
        temp = datasets.GitHub(root=os.path.join(ROOT,'GitHub'))
    elif cls == "LastFMAsia":
        temp = datasets.LastFMAsia(root=os.path.join(ROOT,'LastFMAsia'))
    elif cls == "Twitch":
        if name in ["RU", "PT","DE","FR","ES","EN"]:
            temp = datasets.Twitch(root=os.path.join(ROOT,'Twitch'),name=name)
        else:
            raise NameError(f"No matching name for dataset {name}")
    else:
        raise NameError(f"No matching name for dataset {name}")
        
    return temp

def node_split(data,num_val=100,num_test=100):
    """
    performe a train test valid mask split according the ratio given 
    """
    transform = T.RandomNodeSplit(split="train_rest",num_val=num_val,num_test=num_test)
    return transform(data)

def ClusterSampler(data:data,num_parts:int,batch_size:int):
    cluster = ClusterData(data=data.cpu(),
                          num_parts=num_parts)
    loader = ClusterLoader(cluster_data=cluster,
                           batch_size=batch_size,
                           shuffle=True)
    return loader

def SaintSampler(data:data,type:str,batch_size:int,num_steps:int,sample_coverage:int,walk_length:int):
    assert type in ['saint_rw','saint_node']
    if type == 'saint_node':
        loader = GraphSAINTNodeSampler(data=data.cpu(),
                                       batch_size=batch_size,#用多少根节点开始采样
                                       num_steps=num_steps,
                                       sample_coverage=sample_coverage,
                                       shuffle=True)
    elif type == 'saint_rw':
        loader = GraphSAINTRandomWalkSampler(data=data.cpu(),
                                             batch_size=batch_size,
                                             walk_length=walk_length,
                                             num_steps=num_steps,
                                             sample_coverage=sample_coverage,
                                             shuffle=True)
    return loader

def ShadowKSampler(data:data,depth:int,num_neighbors:int,node_idx:torch.Tensor,batch_size:int,replace:bool):
    loader = ShaDowKHopSampler(data=data.cpu(),
                               depth=depth,
                               num_neighbors=num_neighbors,
                               node_idx=node_idx,
                               replace=replace,
                               batch_size=batch_size,
                               shuffle=True)
    return loader

def NeighborReplaceSampler(data:data,batch_size:int,layers:int):
    loader = NeighborLoader(data=data,
                            num_neighbors=[-1]*layers,
                            input_nodes=None,
                            replace=False,
                            subgraph_type='induced',
                            disjoint=False,
                            batch_size=batch_size,
                            shuffle=True)
    return loader

def OccuranceSampler(data:data,k:int,depth:int,device:str,sampler_batchsize:float):
    sampled_dict = sample_subgraph_with_occurance_constr(data=data,
                                                            k=k,
                                                            depth=depth,
                                                            device=device)
    
    batch_idx = list(sampled_dict.keys())
    train_nodes = len(batch_idx)
    if sampler_batchsize<1:    
        sampler_batchsize = int(sampler_batchsize * train_nodes)
    datalist = list(sampled_dict.values())
    return sampled_dict
    # return DataLoader(datalist,batch_size=sampler_batchsize,shuffle=True,drop_last=True)


if __name__ == "__main__":
    dataset_dic_demo = {"GNNBenchmark":["PATTERN", "CLUSTER", "MNIST", "CIFAR10", "TSP", "CSL"],
                        "TU":["MUTAG","ENZYMES","IMDB-BINARY","REDDIT-BINARY"],
                        "Planetoid":["Cora","CiteSeer","PubMed"],
                        "KarateClub":[],
                        #"NELL":[],
                        "CitationFull":["Cora", "Cora_ML", "CiteSeer", "PubMed"],
                        "Coauthor":["CS", "Physics"],
                        "Amazon":["Computers","Photo"]
                        }    
    dataset_keys = list(dataset_dic_demo.keys())
    for i in dataset_keys:
        names = dataset_dic_demo[i]
        if names:
            for j in names:
                temp = get_dataset(i,j)
        else:
            temp = get_dataset(i)



