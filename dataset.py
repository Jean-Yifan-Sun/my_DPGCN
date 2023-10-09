from torch_geometric import datasets,data
import torch,random,os
from torch_geometric import transforms as T

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
                        "CitationFull":["Cora", "Cora_ML", "CiteSeer", "DBLP", "PubMed"],
                        "Coauthor":["CS", "Physics"],
                        "Amazon":["Computers","Photo"]
                        }    
    dataset_keys = list(dataset_dic_demo.keys())

    transform = T.RandomNodeSplit(split="train_rest",num_val=num_val,num_test=num_test)

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
        else:
            raise NameError(f"No matching name for dataset {name}")
        
    return temp

def node_split(data,num_val=100,num_test=100):
    """
    performe a train test valid mask split according the ratio given 
    """
    transform = T.RandomNodeSplit(split="train_rest",num_val=num_val,num_test=num_test)
    return transform(data)

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



