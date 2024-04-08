import os,time,socket,argparse

local = False

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected boolean value.')


def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--epochs", type=int, default=301)
    argparser.add_argument("--shadow_epochs", type=int, default=301)
    argparser.add_argument("--num_val", type=int, default=.1 )
    argparser.add_argument("--num_test", type=int, default=.45)
    argparser.add_argument("--k_layers", type=int, default=2)
    argparser.add_argument("--hidden_dim", type=int, default=256, nargs="+")
    argparser.add_argument("--learning_rate", type=float, default=0.0001)
    argparser.add_argument("--shadow_learning_rate", type=float, default=0.0001)
    argparser.add_argument("--weight_decay", type=float, default=0.01)
    argparser.add_argument("--momentum", type=float, default=0.9)
    argparser.add_argument("--amsgrad", type=str2bool, nargs='?',
                           const=True, default=False)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--activation", type=str, default='relu')
    argparser.add_argument("--early_stopping", type=str2bool, nargs='?',
                           const=True, default=True)
    argparser.add_argument("--patience", type=int, default=200)
    argparser.add_argument("--optim_type", type=str, default='adam',
                           help='sgd or adam')
    argparser.add_argument("--parallel", type=str2bool, nargs='?',
                           const=True, default=False)
    argparser.add_argument("--seed", type=int, default=123454321)
    
    argparser.add_argument("--split_graph", type=str2bool, nargs='?',
                           const=True, default=False)
    argparser.add_argument("--split_n_subgraphs", type=int, default=1)

    argparser.add_argument("--private", type=str2bool, nargs='?',
                           const=True, default=False)
    argparser.add_argument("--delta", type=float, default=1e-5)
    argparser.add_argument("--gradient_norm_bound", type=float, default=1.)
    argparser.add_argument("--noise_scale", type=float, default=10)
    argparser.add_argument("--lot_size", type=int, default=1)
    argparser.add_argument("--dp_subgraph_sample_size", type=int, default=1)

    argparser.add_argument("--verbose", type=str2bool, nargs='?',
                           const=True, default=True)
    argparser.add_argument("--dataset", type=str, default='cora',
                           help='cora, citeseer, pubmed, reddit,'
                                'reddit-small, or pokec-pets')
    argparser.add_argument("--mia", type=str, default='shadow',
                           help='shadow or more TBD')
    argparser.add_argument("--mia_subsample_rate", type=float, default=.5,
                           help='MIA shadow data, If 1. then no subsampling.')

    argparser.add_argument("--mia_shadow_mode", type=str, default='tsts',
                           help='tsts or tstf')
    argparser.add_argument("--device_num", type=int, default='3',
                           help='cuda device number')
    
    argparser.add_argument("--rdp", type=str2bool, default=False,
                           help='use RDP or not')
    # argparser.add_argument("--rdp_k", type=int, default=3,
    #                        help='occurance constrain k')
    # argparser.add_argument("--rdp_batchsize", type=float, default=.3,
    #                        help='rdp batch size %')
    argparser.add_argument("--sampler", type=str, default='none',
                           help='none, occurance, saint_node, saint_rw, shadow_k, cluster, neighbor')
    argparser.add_argument("--sampler_batchsize", type=float, default=.5,
                           help='batch size for each sampler. either int or percent (一个batch里几个子图)')
    argparser.add_argument("--occurance_k", type=int, default=3,
                           help='occurance constrain k')
    argparser.add_argument("--cluster_numparts", type=int, default=100,
                           help='num parts of cluster sampler')
    argparser.add_argument("--saint_rootnodes", type=int, default=10,
                           help='num roots of saint sampler (用几个节点去做初始节点)')
    argparser.add_argument("--saint_samplecoverage", type=int, default=100,
                           help='num nodes for calculate coverage of saint sampler (无所谓)')
    argparser.add_argument("--saint_walklenth", type=int, default=3,
                           help='num random walks for saint sampler')
    argparser.add_argument("--shadowk_depth", type=int, default=3,
                           help='num depths for shadowk sampler')
    argparser.add_argument("--shadowk_neighbors", type=int, default=100,
                           help='num neighbors for shadowk sampler')
    argparser.add_argument("--shadowk_replace", type=str2bool, default=False,
                           help='node replacement for shadowk sampler (True就有放回)')
    argparser.add_argument("--dp_type", type=str, default='rdp',
                           help='dp ldp rdp')
    argparser.add_argument("--ldp_eps", type=float, default=10,
                           help='ldp eps')

    args = argparser.parse_args()
    print(args)
    return args


class Settings(object):
    '''
    Configuration for the project.
    '''
    def __init__(self):
        self.args = parse_arguments()

        if not self.args.private:
            self.model_name = f'K{self.args.k_layers}_'\
                              f'E{self.args.epochs}_'\
                              f'SubSampl{self.args.mia_subsample_rate}_'\
                              f'ACT{self.args.activation}_'\
                              f'Hd{self.args.hidden_dim}_'\
                              f'Lr{self.args.learning_rate}_'\
                              f'Wd{self.args.weight_decay}_'\
                              f'M{self.args.momentum}_'\
                              f'D{self.args.dropout}_'\
                              f'Es{self.args.early_stopping}_'\
                              f'Pat{self.args.patience}_'\
                              f'Op{self.args.optim_type}_'\
                              f'Split{self.args.split_graph}'\
                            #   f'Subgraphs{self.args.split_n_subgraphs}'

        else:
            self.epsilon = 0
            self.model_name = f'K{self.args.k_layers}_'\
                              f'E{self.args.epochs}_'\
                              f'SubSampl{self.args.mia_subsample_rate}_'\
                              f'Hd{self.args.hidden_dim}_'\
                              f'Lr{self.args.learning_rate}_'\
                              f'Wd{self.args.weight_decay}_'\
                              f'M{self.args.momentum}_'\
                              f'D{self.args.dropout}_'\
                              f'Es{self.args.early_stopping}_'\
                              f'Pat{self.args.patience}_'\
                              f'Op{self.args.optim_type}_'\
                              f'Eps{self.epsilon}_'\
                              f'Gnb{self.args.gradient_norm_bound}_'\
                              f'Ns_{self.args.noise_scale}_'\
                              f'LotS_{self.args.lot_size}_'\
                              f'Split{self.args.split_graph}'\
                              f'Subgraphs{self.args.split_n_subgraphs}'

        # if self.args.dataset == 'pokec-pets':
        #     self.model_name += f'PokecType_{self.args.pokec_feat_type}'

        # Setting up directory structure
        current_directory = os.path.dirname(__file__)
        parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
        self.root_dir = current_directory
        self.out_dir = os.path.join(parent_directory,"out/")
        self.data_dir = os.path.join(self.out_dir, f'{self.args.dataset}')
        self.privacy_dir = os.path.join(self.data_dir,f'Privacy_{self.args.private}_{self.args.mia}_RDP_{self.args.rdp}_sampler_{self.args.sampler}')
        self.log_dir = os.path.join(self.privacy_dir, f'log_{self.model_name}')
        self.seed_dir = os.path.join(self.log_dir, f'Seed_{self.args.seed}')
        now = time.localtime()
        self.time_dir = os.path.join(
                self.seed_dir,
                f'{now[0]}_{now[1]}_{now[2]}_{now[3]:02d}:{now[4]:02d}:{now[5]:02d}'
                )

    def make_dirs(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.privacy_dir):
            os.makedirs(self.privacy_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.seed_dir):
            os.makedirs(self.seed_dir)
        if not os.path.exists(self.time_dir):
            os.makedirs(self.time_dir)
