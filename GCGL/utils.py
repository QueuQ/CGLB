import dgl
from dgl.data.utils import Subset
import numpy as np
import random
import torch
from itertools import accumulate
from dgllife.utils.splitters import RandomSplitter
from functools import partial

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.random.seed(seed)

class GraphLevelDataset():
    def __init__(self, args):
        if args['dataset'] in ['Tox21','Tox21-tIL']:
            from dgllife.data import Tox21
            dataset = Tox21(smiles_to_graph=partial(args['smiles_to_graph'], add_self_loop=True),
                            node_featurizer=args.get('node_featurizer', None),
                            edge_featurizer=args.get('edge_featurizer', None),
                            # load=True,
                            cache_file_path=f"{args['result_path']}/" + args['exp'])
            train_set, val_set, test_set = RandomSplitter.train_val_test_split(
                dataset, frac_train=args['frac_train'], frac_val=args['frac_val'],
                frac_test=args['frac_test'], random_state=args['random_seed'])
            args['n_train_examples'] = len(train_set)

        elif args['dataset'] in ['SIDER','SIDER-tIL']:
            from dgllife.data import SIDER
            dataset = SIDER(smiles_to_graph=partial(args['smiles_to_graph'], add_self_loop=True),
                            node_featurizer=args.get('node_featurizer', None),
                            edge_featurizer=args.get('edge_featurizer', None),
                            # load=True,
                            cache_file_path=f"{args['result_path']}/" + args['exp'])
            train_set, val_set, test_set = RandomSplitter.train_val_test_split(
                dataset, frac_train=args['frac_train'], frac_val=args['frac_val'],
                frac_test=args['frac_test'], random_state=args['random_seed'])
            args['n_train_examples'] = len(train_set)
        elif args['dataset'] in ['PubChemBioAssayAromaticity','Aromaticity-CL','Aromaticity']:
            from dgllife.data import PubChemBioAssayAromaticity
            dataset = PubChemBioAssayAromaticity(smiles_to_graph=partial(args['smiles_to_graph'], add_self_loop=True),
                                                 node_featurizer=args.get('node_featurizer', None),
                                                 edge_featurizer=args.get('edge_featurizer', None),
                                                 # load=True,
                                                 )
            dataset.labels = dataset.labels.view(-1)
            n_per_cls = np.array([(dataset.labels == i).sum() for i in range((dataset.labels.max().int().item() + 1))])
            selected_clss = (n_per_cls > args['threshold_pubchem']).nonzero()[
                0].tolist()  # at least 3 examples for train val test splittings
            ids_per_cls = {i: (dataset.labels == i).nonzero().view(-1).tolist() for i in selected_clss}
            args['n_per_cls'] = {i: len(ids_per_cls[i]) for i in ids_per_cls}
            args['tasks'] = []
            while len(selected_clss) >= args['n_cls_per_task']:
                clss = [selected_clss.pop(0) for i in range(args['n_cls_per_task'])]
                args['tasks'].append(clss)
            # args['tasks'].reverse()
            frac_list = np.asarray([args['frac_train'], args['frac_val'], args['frac_test']])
            train_set, val_set, test_set = [], [], []
            if args['method'] not in ['jointtrain', 'jointreplay'] or not args['clsIL']:
                for clss in args['tasks']:
                    train_ids_task, val_ids_task, test_ids_task = [], [], []
                    for cls in clss:
                        ids = ids_per_cls[cls]
                        assert np.allclose(np.sum(frac_list), 1.), \
                            'Expect frac_list sum to 1, got {:.4f}'.format(np.sum(frac_list))
                        num_data = len(ids)
                        lengths = (num_data * frac_list).astype(int)
                        for i in range(len(lengths) - 1, 0, -1):
                            lengths[i] = max(1, lengths[i])  # ensure at least one example for test and val
                        lengths[0] = num_data - np.sum(lengths[1:])
                        split = [ids[offset - length:offset] for offset, length in zip(accumulate(lengths), lengths)]
                        train_ids_task.extend(split[0])
                        val_ids_task.extend(split[1])
                        test_ids_task.extend(split[2])
                    train_set.append(Subset(dataset, train_ids_task))
                    val_set.append(Subset(dataset, val_ids_task))
                    test_set.append(Subset(dataset, test_ids_task))
                args['n_train_examples'] = len(train_ids_task)

            elif args['method'] is 'none':
                train_ids_task, val_ids_task, test_ids_task = [], [], []
                for clss in args['tasks']:
                    for cls in clss:
                        ids = ids_per_cls[cls]
                        random.shuffle(ids)
                        assert np.allclose(np.sum(frac_list), 1.), \
                            'Expect frac_list sum to 1, got {:.4f}'.format(np.sum(frac_list))
                        num_data = len(ids)
                        lengths = (num_data * frac_list).astype(int)
                        for i in range(len(lengths) - 1, 0, -1):
                            lengths[i] = max(1, lengths[i])  # ensure at least one example for test and val
                        lengths[0] = num_data - np.sum(lengths[1:])
                        split = [ids[offset - length:offset] for offset, length in zip(accumulate(lengths), lengths)]
                        train_ids_task.extend(split[0])
                        val_ids_task.extend(split[1])
                        test_ids_task.extend(split[2])
                train_set.append(Subset(dataset, train_ids_task))
                val_set.append(Subset(dataset, val_ids_task))
                test_set.append(Subset(dataset, test_ids_task))
                args['n_train_examples'] = len(train_ids_task)

            elif args['method'] in ['jointreplay','jointtrain']:
                # train_ids_task, val_ids_task, test_ids_task = [], [], []
                for cid, _ in enumerate(args['tasks']):
                    clss = []
                    train_ids_task, val_ids_task, test_ids_task = [], [], []
                    for c in args['tasks'][0:cid + 1]:
                        clss.extend(c)
                    for cls in clss:
                        ids = ids_per_cls[cls]
                        random.shuffle(ids)
                        assert np.allclose(np.sum(frac_list), 1.), \
                            'Expect frac_list sum to 1, got {:.4f}'.format(np.sum(frac_list))
                        num_data = len(ids)
                        lengths = (num_data * frac_list).astype(int)
                        for i in range(len(lengths) - 1, 0, -1):
                            lengths[i] = max(1, lengths[i])  # ensure at least one example for test and val
                        lengths[0] = num_data - np.sum(lengths[1:])
                        split = [ids[offset - length:offset] for offset, length in zip(accumulate(lengths), lengths)]
                        train_ids_task.extend(split[0])
                        val_ids_task.extend(split[1])
                        test_ids_task.extend(split[2])
                    train_set.append(Subset(dataset, train_ids_task))
                    val_set.append(Subset(dataset, val_ids_task))
                    test_set.append(Subset(dataset, test_ids_task))
                args['n_train_examples'] = len(train_ids_task)

        self.dataset, self.train_set, self.val_set, self.test_set = dataset, train_set, val_set, test_set


def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)  
    return smiles, bg, labels, masks

def collate_minigs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    assert len(data[0]) in [2], \
        'Expect the tuple to be of length 2, got {:d}'.format(len(data[0]))
    graphs, labels = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    masks = torch.ones(labels.shape)
    return masks, bg, labels, masks

def load_model(args):
    if args['method'] == 'twp':
        from Backbones.gcn_predictor import GCNPredictor
        model = GCNPredictor(in_feats=args['d_data'],
                             hidden_feats=args['gcn_hidden_feats'],
                             predictor_hidden_feats=args['predictor_hidden_feats'],
                             n_tasks=args['n_outheads'])
    else:
        from dgllife.model import GCNPredictor
        model = GCNPredictor(in_feats=args['d_data'],
                             hidden_feats=args['gcn_hidden_feats'],
                             predictor_hidden_feats=args['predictor_hidden_feats'],
                             n_tasks=args['n_outheads'])

    return model

def load_model_old(args):
    if args['backbone'] == 'CusGCN':
        from Backbones.gcn_predictor import CusGCNPredictor
        model = CusGCNPredictor(in_feats=args['node_featurizer'].feat_size(),
                              hidden_feats=args['gcn_hidden_feats'],
                              predictor_hidden_feats=args['predictor_hidden_feats'],
                              n_tasks=args['n_tasks'], args=args)

    if args['backbone'] == 'GCN':
        from dgllife.model import GCNPredictor
        model = GCNPredictor(in_feats=args['d_data'],
                             hidden_feats=args['gcn_hidden_feats'],
                             predictor_hidden_feats=args['predictor_hidden_feats'],
                             n_tasks=args['n_outheads'])

    if args['backbone'] == 'GAT':
        from dgllife.model import GATPredictor
        model = GATPredictor(in_feats=args['node_featurizer'].feat_size(),
                             hidden_feats=args['gat_hidden_feats'],
                             num_heads=args['num_heads'],
                             predictor_hidden_feats=args['predictor_hidden_feats'],
                             n_tasks=args['n_outheads'])

    if args['backbone'] == 'Weave':
        from dgllife.model import WeavePredictor
        model = WeavePredictor(node_in_feats=args['node_featurizer'].feat_size(),
                               edge_in_feats=args['edge_featurizer'].feat_size(),
                               num_gnn_layers=args['num_gnn_layers'],
                               gnn_hidden_feats=args['gnn_hidden_feats'],
                               graph_feats=args['graph_feats'],
                               n_tasks=args['n_outheads'])

    return model

def load_twpmodel(args):
    if args['backbone'] == 'GCN':
        from Backbones.gcn_predictor import GCNPredictor
        model = GCNPredictor(in_feats=args['d_data'],
                             hidden_feats=args['gcn_hidden_feats'],
                             predictor_hidden_feats=args['predictor_hidden_feats'],
                             n_tasks=args['n_outheads'])
    return model