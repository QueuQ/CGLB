import random
import pickle
import numpy as np
import torch
from torch import Tensor, device, dtype
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from dgl.data import CoraGraphDataset, CoraFullDataset, register_data_args, RedditDataset
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl, Evaluator
import copy
from sklearn.metrics import roc_auc_score, average_precision_score

class Linear_IL(nn.Linear):
    def forward(self, input: Tensor, n_cls=10000, normalize = True) -> Tensor:
        if normalize:
            return F.linear(F.normalize(input,dim=-1), F.normalize(self.weight[0:n_cls],dim=-1), bias=None)
        else:
            return F.linear(input, self.weight[0:n_cls], bias=None)

def accuracy(logits, labels, cls_balance=True, ids_per_cls=None):
    if cls_balance:
        logi = logits.cpu().numpy()
        _, indices = torch.max(logits, dim=1)
        ids = _.cpu().numpy()
        acc_per_cls = [torch.sum((indices == labels)[ids])/len(ids) for ids in ids_per_cls]
        return sum(acc_per_cls).item()/len(acc_per_cls)
    else:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def mean_AP(args,logits, labels, cls_balance=True, ids_per_cls=None):
    eval_ogb = Evaluator(args.dataset)
    pos = (F.sigmoid(logits)>0.5)
    APs = 0
    if cls_balance:
        _, indices = torch.max(logits, dim=1)
        ids = _.cpu().numpy()
        acc_per_cls = [torch.sum((indices == labels)[ids])/len(ids) for ids in ids_per_cls]
        return sum(acc_per_cls).item()/len(acc_per_cls)
    else:
        input_dict = {"y_true": labels, "y_pred": logits}

        eval_result_ogb = eval_ogb.eval(input_dict)
        for c,ids in enumerate(ids_per_cls):
            TP_ = (pos[ids,c]*labels[ids,c]).sum()
            FP_ = (pos[ids,c]*(labels[ids, c]==False)).sum()
            med0 = TP_ + FP_ + 0.0001
            med1 = TP_ / med0
            APs += med1
        med2 = APs/labels.shape[1]

            #mAP_per_cls.append((TP / (TP+FP)).mean().item())
        #return (TP / (TP+FP)).mean().item()

        return med2.item()

def evaluate_batch(args,model, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None):
    model.eval()
    with torch.no_grad():
        dataloader = dgl.dataloading.NodeDataLoader(g.cpu(), list(range(labels.shape[0])), args.nb_sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)
        output = torch.tensor([]).cuda(args.gpu)
        output_l = torch.tensor([]).cuda(args.gpu)
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            output_predictions, _ = model.forward_batch(blocks, input_features)
            output = torch.cat((output,output_predictions),dim=0)
            output_l = torch.cat((output_l, output_labels), dim=0)

        #output, _ = model(g, features)
        #judget = (labels==output_l).sum()
        logits = output[:, label_offset1:label_offset2]
        if cls_balance:
            return accuracy(logits, labels.cuda(args.gpu), cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask].cuda(args.gpu), cls_balance=cls_balance, ids_per_cls=ids_per_cls)

def evaluate(model, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None, save_logits_name=None):
    model.eval()
    with torch.no_grad():
        output, _ = model(g, features)
        logits = output[:, label_offset1:label_offset2]
        if save_logits_name is not None:
            with open(
                    '/store/continual_graph_learning/baselines_by_TWP/NCGL/results/logits_for_tsne/{}.pkl'.format(
                        save_logits_name), 'wb') as f:
                pickle.dump({'logits':logits,'ids_per_cls':ids_per_cls}, f)

        if cls_balance:
            return accuracy(logits, labels, cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask], cls_balance=cls_balance, ids_per_cls=ids_per_cls)

class incremental_graph_trans_(nn.Module):
    def __init__(self, dataset, n_cls):
        super().__init__()
        # transductive setting
        self.graph, self.labels = dataset[0]
        #self.graph = dgl.add_reverse_edges(self.graph)
        #self.graph = dgl.add_self_loop(self.graph)
        self.graph.ndata['label'] = self.labels
        self.d_data = self.graph.ndata['feat'].shape[1]
        self.n_cls = n_cls
        self.d_data = self.graph.ndata['feat'].shape[1]
        self.n_nodes = self.labels.shape[0]
        self.tr_va_te_split = dataset[1]

    def get_graph(self, tasks_to_retain=[], node_ids = None, remove_edges=True):
        # get the partial graph
        # tasks-to-retain: classes retained in the partial graph
        # tasks-to-infer: classes to predict on the partial graph
        node_ids_ = copy.deepcopy(node_ids)
        node_ids_retained = []
        ids_train_old, ids_valid_old, ids_test_old = [], [], []
        if len(tasks_to_retain) > 0:
            # retain nodes according to classes
            for t in tasks_to_retain:
                ids_train_old.extend(self.tr_va_te_split[t][0])
                ids_valid_old.extend(self.tr_va_te_split[t][1])
                ids_test_old.extend(self.tr_va_te_split[t][2])
                node_ids_retained.extend(self.tr_va_te_split[t][0] + self.tr_va_te_split[t][1] + self.tr_va_te_split[t][2])
            subgraph_0 = dgl.node_subgraph(self.graph, node_ids_retained, store_ids=True)
            if node_ids_ is None:
                subgraph = subgraph_0
        if node_ids_ is not None:
            # retrain the given nodes
            if not isinstance(node_ids_[0],list):
                # if nodes are not divided into different tasks
                subgraph_1 = dgl.node_subgraph(self.graph, node_ids_, store_ids=True)
                if remove_edges:
                    # to facilitate the methods like ER-GNN to only retrieve nodes
                    n_edges = subgraph_1.edges()[0].shape[0]
                    subgraph_1.remove_edges(list(range(n_edges)))
            elif isinstance(node_ids_[0],list):
                # if nodes are diveded into different tasks
                subgraph_1 = dgl.node_subgraph(self.graph, node_ids_[0], store_ids=True) # load the subgraph containing nodes of the first task
                node_ids_.pop(0)
                for ids in node_ids_:
                    # merge the remaining nodes
                    subgraph_1 = dgl.batch([subgraph_1,dgl.node_subgraph(self.graph, ids, store_ids=True)])

            if len(tasks_to_retain)==0:
                subgraph = subgraph_1

        if len(tasks_to_retain)>0 and node_ids is not None:
            subgraph = dgl.batch([subgraph_0,subgraph_1])

        old_ids = subgraph.ndata['_ID'].cpu()
        ids_train = [(old_ids == i).nonzero()[0][0].item() for i in ids_train_old]
        ids_val = [(old_ids == i).nonzero()[0][0].item() for i in ids_valid_old]
        ids_test = [(old_ids == i).nonzero()[0][0].item() for i in ids_test_old]
        node_ids_per_task_reordered = []
        for c in tasks_to_retain:
            ids = (subgraph.ndata['label'] == c).nonzero()[:, 0].view(-1).tolist()
            node_ids_per_task_reordered.append(ids)
        subgraph = dgl.add_self_loop(subgraph)

        return subgraph, node_ids_per_task_reordered, [ids_train, ids_val, ids_test]

def train_valid_test_split(ids,ratio_valid_test):
    va_te_ratio = sum(ratio_valid_test)
    train_ids, va_te_ids = train_test_split(ids, test_size=va_te_ratio)
    return [train_ids] + train_test_split(va_te_ids, test_size=ratio_valid_test[1]/va_te_ratio)


class NodeLevelDataset(incremental_graph_trans_):
    def __init__(self,name='ogbn-arxiv',IL='class',default_split=False,ratio_valid_test=None,args=None):
        r""""
        name: name of the dataset
        IL: use task- or class-incremental setting
        default_split: if True, each class is split according to the splitting of the original dataset, which may cause the train-val-test ratio of different classes greatly different
        ratio_valid_test: in form of [r_val,r_test] ratio of validation and test set, train set ratio is directly calculated by 1-r_val-r_test
        """

        # return an incremental graph instance that can return required subgraph upon request
        if name[0:4] == 'ogbn':
            data = DglNodePropPredDataset(name, root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        elif name in ['CoraFullDataset', 'CoraFull','corafull', 'CoraFull-CL','Corafull-CL']:
            data = CoraFullDataset()
            graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
        elif name in ['reddit','Reddit','Reddit-CL']:
            data = RedditDataset(self_loop=False)
            graph, label = data.graph, data.labels.view(-1, 1)
        elif name == 'Arxiv-CL':
            data = DglNodePropPredDataset('ogbn-arxiv', root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        elif name == 'Products-CL':
            data = DglNodePropPredDataset('ogbn-products', root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        else:
            print('invalid data name')
        n_cls = data.num_classes
        cls = [i for i in range(n_cls)]
        cls_id_map = {i: list((label.squeeze() == i).nonzero().squeeze().view(-1, ).numpy()) for i in cls}
        cls_sizes = {c: len(cls_id_map[c]) for c in cls_id_map}
        for c in cls_sizes:
            if cls_sizes[c] < 2:
                cls.remove(c) # remove classes with less than 2 examples, which cannot be split into train, val, test sets
        cls_id_map = {i: list((label.squeeze() == i).nonzero().squeeze().view(-1, ).numpy()) for i in cls}
        n_cls = len(cls)
        if default_split:
            split_idx = data.get_idx_split()
            train_idx, valid_idx, test_idx = split_idx["train"].tolist(), split_idx["valid"].tolist(), split_idx[
                "test"].tolist()
            tr_va_te_split = {c: [list(set(cls_id_map[c]).intersection(set(train_idx))),
                                  list(set(cls_id_map[c]).intersection(set(valid_idx))),
                                  list(set(cls_id_map[c]).intersection(set(test_idx)))] for c in cls}

        elif not default_split:
            split_name = f'{args.data_path}/tr{round(1-ratio_valid_test[0]-ratio_valid_test[1],2)}_va{ratio_valid_test[0]}_te{ratio_valid_test[1]}_split_{name}.pkl'
            try:
                tr_va_te_split = pickle.load(open(split_name, 'rb')) # could use same split across different experiments for consistency
            except:
                if ratio_valid_test[1] > 0:
                    tr_va_te_split = {c: train_valid_test_split(cls_id_map[c], ratio_valid_test=ratio_valid_test)
                                      for c in
                                      cls}
                    print(f'splitting is {ratio_valid_test}')
                elif ratio_valid_test[1] == 0:
                    tr_va_te_split = {c: [cls_id_map[c], [], []] for c in
                                      cls}
                with open(split_name, 'wb') as f:
                    pickle.dump(tr_va_te_split, f)
        super().__init__([[graph, label], tr_va_te_split], n_cls)