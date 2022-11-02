# most of the gem code are modifed from facebook's official repo
import pickle
import gc
import random
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
#import quadprog
import dgl
from .gem_utils import store_grad, overwrite_grad, project2cone2


class NET(nn.Module):
    """
    GEM baseline for NCGL tasks

    :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
    :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
    :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

    """
    def __init__(self,
                model,
                task_manager,
                args):
        super(NET, self).__init__()        
        self.net = model
        self.task_manager = task_manager

        self.ce = torch.nn.functional.cross_entropy
        self.opt = optim.Adam(self.net.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        
        self.margin = args.gem_args['memory_strength']
        self.n_memories = int(args.gem_args['n_memories'])

        # allocate episodic memory
        # for semi-supervised data, it will store the training mask for every old tasks
        self.memory_data = []
        
        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.net.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), len(args.task_seq)).cuda()
        
        # allocate counters
        self.observed_tasks = []
        self.current_task = -1
        self.mem_cnt = 0
        self.old_dataloaders = {}
    
    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                The method for learning the given tasks under the class-IL setting.

                :param args: Same as the args in __init__().
                :param g: The graph of the current task.
                :param features: Node features of the current task.
                :param labels: Labels of the nodes in the current task.
                :param t: Index of the current task.
                :param train_ids: The indices of the nodes participating in the training.
                :param ids_per_cls: Indices of the nodes in each class (not in use in the current baseline).
                :param dataset: The entire dataset (not in use in the current baseline).

                """
        # update memory
        if t != self.current_task:
            self.observed_tasks.append(t)
            self.current_task = t
            
        # Update ring buffer storing examples from current task        
        if t >= len(self.memory_data):
            tmask = train_ids[0:self.n_memories]
            self.memory_data.append(tmask)
        
        # compute gradient on previous tasks
        for old_task_i in self.observed_tasks[:-1]:
            if args.inter_task_edges:
                subgraph, ids_per_cls, [train_ids_, valid_ids, test_ids] = pickle.load(open(f'{args.data_path}/inter_tsk_edge/{args.dataset}_{args.task_seq[old_task_i]}.pkl','rb'))
            else:
                subgraph, ids_per_cls, [train_ids_, valid_ids, test_ids] = pickle.load(open(
                    f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[old_task_i]}.pkl', 'rb'))

            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features_, labels_ = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            self.net.zero_grad()

            offset1, offset2 = self.task_manager.get_label_offset(old_task_i)
            output, _ = self.net(subgraph, features_)

            old_task_loss = self.ce(
                output[self.memory_data[old_task_i], offset1: offset2],
                labels_[self.memory_data[old_task_i]])# - offset1)

            old_task_loss.backward()
            store_grad(self.net.parameters, self.grads, self.grad_dims,
                            old_task_i)

        # now compute the grad on the current minibatch
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        output, _ = self.net(g, features)
        output_labels = labels[train_ids]
        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        if args.classifier_increase:
            loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
        else:
            loss = self.ce(output[train_ids], output_labels, weight=loss_w_)

        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.net.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.net.parameters, self.grads[:, t],
                               self.grad_dims)
        
        self.opt.step()

    def observe_task_IL(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                        The method for learning the given tasks under the task-IL setting.

                        :param args: Same as the args in __init__().
                        :param g: The graph of the current task.
                        :param features: Node features of the current task.
                        :param labels: Labels of the nodes in the current task.
                        :param t: Index of the current task.
                        :param train_ids: The indices of the nodes participating in the training.
                        :param ids_per_cls: Indices of the nodes in each class (not in use in the current baseline).
                        :param dataset: The entire dataset (not in use in the current baseline).

                        """
        # update memory
        if t != self.current_task:
            self.observed_tasks.append(t)
            self.current_task = t

        # Update ring buffer storing examples from current task
        if t >= len(self.memory_data):
            tmask = random.choices(train_ids, k=self.n_memories) #train_ids[0:self.n_memories]
            self.memory_data.append(tmask)

        # compute gradient on previous tasks
        for old_task_i in self.observed_tasks[:-1]:
            if args.inter_task_edges:
                subgraph, ids_per_cls, [train_ids_, valid_ids, test_ids] = pickle.load(open(
                    f'{args.data_path}/inter_tsk_edge/{args.dataset}_{args.task_seq[old_task_i]}.pkl', 'rb'))
            else:
                subgraph, ids_per_cls, [train_ids_, valid_ids, test_ids] = pickle.load(open(
                    f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[old_task_i]}.pkl','rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features_, labels_ = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            self.net.zero_grad()

            offset1, offset2 = self.task_manager.get_label_offset(old_task_i-1)[1], self.task_manager.get_label_offset(old_task_i)[1]
            output, _ = self.net(subgraph, features_)
            output_labels_ = labels_[self.memory_data[old_task_i]]

            # balance the loss of data from different classes
            if args.cls_balance:
                n_per_cls = [(output_labels_ == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))

            old_task_loss = self.ce(output[self.memory_data[old_task_i], offset1: offset2], output_labels_-offset1, weight=loss_w_[offset1: offset2])

            old_task_loss.backward()
            store_grad(self.net.parameters, self.grads, self.grad_dims, old_task_i)

        # now compute the grad on the current minibatch
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t-1)[1], self.task_manager.get_label_offset(t)[1]
        output, _ = self.net(g, features)
        output_labels = labels[train_ids]-offset1
        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.net.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1), self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.net.parameters, self.grads[:, t], self.grad_dims)
        self.opt.step()

    def observe_task_IL_batch(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                        The method for learning the given tasks under the task-IL setting with mini-batch training.

                        :param args: Same as the args in __init__().
                        :param g: The graph of the current task.
                        :param dataloader: The data loader for mini-batch training
                        :param features: Node features of the current task.
                        :param labels: Labels of the nodes in the current task.
                        :param t: Index of the current task.
                        :param train_ids: The indices of the nodes participating in the training.
                        :param ids_per_cls: Indices of the nodes in each class (currently not in use).
                        :param dataset: The entire dataset (currently not in use).

                        """
        # Update ring buffer storing examples from current task
        if t >= len(self.memory_data):
            tmask = random.sample(train_ids, k=min(self.n_memories, len(train_ids)))  # train_ids[0:self.n_memories]
            self.memory_data.append(tmask)

        # update memory
        if t != self.current_task:
            self.observed_tasks.append(t)
            self.current_task = t

        # compute gradient on previous tasks
        for old_task_i in self.observed_tasks[:-1]:
            old_task_loss = 0
            offset1, offset2 = self.task_manager.get_label_offset(old_task_i - 1)[1], \
                               self.task_manager.get_label_offset(old_task_i)[1]
            if args.inter_task_edges:
                subgraph, ids_per_cls, [train_ids_, valid_ids, test_ids] = pickle.load(open(
                    f'{args.data_path}/inter_tsk_edge/{args.dataset}_{args.task_seq[old_task_i]}.pkl', 'rb'))
            else:
                subgraph, ids_per_cls, [train_ids_, valid_ids, test_ids] = pickle.load(open(
                    f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[old_task_i]}.pkl', 'rb'))
            # subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            old_dataloader = dgl.dataloading.NodeDataLoader(subgraph.cpu(),
                                                            self.memory_data[old_task_i],
                                                            args.nb_sampler,
                                                            batch_size=args.batch_size,
                                                            shuffle=args.batch_shuffle,
                                                            drop_last=False)
            for input_nodes, output_nodes, blocks in old_dataloader:
                self.net.zero_grad()
                blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label'].squeeze()
                if args.cls_balance:
                    n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                    loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                else:
                    loss_w_ = [1. for i in range(args.n_cls)]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                output_labels = output_labels - offset1
                output_predictions, _ = self.net.forward_batch(blocks, input_features)
                if args.classifier_increase:
                    old_task_loss_ = self.ce(output_predictions[:, offset1:offset2], output_labels,
                                   weight=loss_w_[offset1: offset2])
                else:
                    old_task_loss_ = self.ce(output_predictions, output_labels, weight=self.aux_loss_w_)
                old_task_loss = old_task_loss + old_task_loss_
            old_task_loss.backward()
            store_grad(self.net.parameters, self.grads, self.grad_dims, old_task_i)
            gc.collect()

        # now compute the grad on the current task
        offset1, offset2 = self.task_manager.get_label_offset(t-1)[1], self.task_manager.get_label_offset(t)[1]

        no_solution_ind = 0
        for input_nodes, output_nodes, blocks in dataloader:
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            #output_labels = output_labels - offset1
            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            output_labels = output_labels - offset1

            output_predictions,_ = self.net.forward_batch(blocks, input_features)
            if args.classifier_increase:
                loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output_predictions, output_labels, weight=self.aux_loss_w_)
            loss.backward()
            # check if gradient violates constraints
            if len(self.observed_tasks) > 1:
                # copy gradient
                store_grad(self.net.parameters, self.grads, self.grad_dims, t)
                indx = torch.cuda.LongTensor(self.observed_tasks[:-1])
                dotp = torch.mm(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    try:
                        project2cone2(self.grads[:, t].unsqueeze(1), self.grads.index_select(1, indx), self.margin)
                        # copy gradients back
                        overwrite_grad(self.net.parameters, self.grads[:, t], self.grad_dims)
                    except:
                        if no_solution_ind == 0:
                            # avoid output error for each batch
                            print('no solution situation observed')
                        no_solution_ind = 1

            self.opt.step()
            gc.collect()
            torch.cuda.empty_cache()


    def observe_class_IL_batch(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                                The method for learning the given tasks under the class-IL setting with mini-batch training.

                                :param args: Same as the args in __init__().
                                :param g: The graph of the current task.
                                :param dataloader: The data loader for mini-batch training
                                :param features: Node features of the current task.
                                :param labels: Labels of the nodes in the current task.
                                :param t: Index of the current task.
                                :param train_ids: The indices of the nodes participating in the training.
                                :param ids_per_cls: Indices of the nodes in each class (currently not in use).
                                :param dataset: The entire dataset (currently not in use).

                                """
        # update memory
        if t != self.current_task:
            self.observed_tasks.append(t)
            self.current_task = t

            # create data loaders for previous tasks
            self.old_dataloaders = {}
            for old_task_i in self.observed_tasks[:-1]:
                if args.inter_task_edges:
                    subgraph, ids_per_cls, [train_ids_, valid_ids, test_ids] = pickle.load(open(
                        f'{args.data_path}/inter_tsk_edge/{args.dataset}_{args.task_seq[old_task_i]}.pkl', 'rb'))
                else:
                    subgraph, ids_per_cls, [train_ids_, valid_ids, test_ids] = pickle.load(open(
                        f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[old_task_i]}.pkl', 'rb'))

                self.old_dataloaders[old_task_i] = dgl.dataloading.NodeDataLoader(subgraph, self.memory_data[old_task_i], args.nb_sampler,
                                                            batch_size=args.batch_size, shuffle=args.batch_shuffle,
                                                            drop_last=False)

        # Update ring buffer storing examples from current task
        if t >= len(self.memory_data):
            tmask = random.sample(train_ids, k=min(self.n_memories,len(train_ids))) #train_ids[0:self.n_memories]
            self.memory_data.append(tmask)

        # compute gradient on previous tasks
        for old_task_i in self.observed_tasks[:-1]:
            old_task_loss = 0
            offset1, offset2 = self.task_manager.get_label_offset(old_task_i)
            for input_nodes, output_nodes, blocks in self.old_dataloaders[old_task_i]:
                self.net.zero_grad()
                blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label'].squeeze()
                if args.cls_balance:
                    n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                    loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                else:
                    loss_w_ = [1. for i in range(args.n_cls)]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                output_predictions, _ = self.net.forward_batch(blocks, input_features)
                if args.classifier_increase:
                    old_task_loss_ = self.ce(output_predictions[:, offset1:offset2], output_labels,
                                   weight=loss_w_[offset1: offset2])
                else:
                    old_task_loss_ = self.ce(output_predictions, output_labels, weight=self.aux_loss_w_)
                old_task_loss = old_task_loss + old_task_loss_
            old_task_loss.backward()
            store_grad(self.net.parameters, self.grads, self.grad_dims, old_task_i)

        # now compute the grad on the current task
        offset1, offset2 = self.task_manager.get_label_offset(t)

        no_solution_ind = 0
        for input_nodes, output_nodes, blocks in dataloader:
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            output_predictions,_ = self.net.forward_batch(blocks, input_features)
            if args.classifier_increase:
                loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output_predictions, output_labels, weight=self.aux_loss_w_)
            loss.backward()
            # check if gradient violates constraints
            if len(self.observed_tasks) > 1:
                # copy gradient
                store_grad(self.net.parameters, self.grads, self.grad_dims, t)
                indx = torch.cuda.LongTensor(self.observed_tasks[:-1])
                dotp = torch.mm(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    try:
                        project2cone2(self.grads[:, t].unsqueeze(1), self.grads.index_select(1, indx), self.margin)
                        # copy gradients back
                        overwrite_grad(self.net.parameters, self.grads[:, t], self.grad_dims)
                    except:
                        if no_solution_ind == 0:
                            # avoid output error for each batch
                            print('no solution situation observed')
                        no_solution_ind = 1

            self.opt.step()