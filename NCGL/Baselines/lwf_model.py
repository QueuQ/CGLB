import torch
import copy
import os
from torch.autograd import Variable
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax, GATConv
import torch.nn as nn

def MultiClassCrossEntropy(logits, labels, T):
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return outputs


def kaiming_normal_init(m):
	if isinstance(m, torch.nn.Conv2d):
		torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, torch.nn.Linear):
		torch.nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
        
class NET(torch.nn.Module):
    """
        LwF baseline for NCGL tasks

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

        """
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager
        self.args = args
        self.activation = F.elu

        # setup network
        self.net = model
        self.net.apply(kaiming_normal_init)                

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        self.current_task = 0
        self.n_classes = 5
        self.n_known = 0
        self.prev_model = None
 
    def forward(self, g, features):
        
        h = features
        h = self.feature_extractor(g, h)[0]
        if len(h.shape)==3:
            h = h.flatten(1)
        h = self.activation(h)
        h = self.gat(g, h)[0]
        if len(h.shape)==3:
            h = h.mean(1)
        return h
                
    def observe(self, args, g, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        """
                The method for learning the given tasks under the class-IL setting.

                :param args: Same as the args in __init__().
                :param g: The graph of the current task.
                :param features: Node features of the current task.
                :param labels: Labels of the nodes in the current task.
                :param t: Index of the current task.
                :param prev_model: The model obtained after learning the previous task.
                :param train_ids: The indices of the nodes participating in the training.
                :param ids_per_cls: Indices of the nodes in each class (not in use in the current baseline).
                :param dataset: The entire dataset (not in use in the current baseline).

                """
        self.net.train()
        # if new task
        if t != self.current_task:            
            self.current_task = t        
            
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        logits = self.net(g, features)
        if isinstance(logits,tuple):
            logits = logits[0]
        output_labels = labels[train_ids]

        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        if args.classifier_increase:
            loss = self.ce(logits[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
        else:
            loss = self.ce(logits[train_ids], output_labels, weight=loss_w_)

        if t > 0:
            target_ = prev_model.forward(g, features)
            if isinstance(target_, tuple):
                target_ = target_[0]
            target = target_[train_ids]
            o1, o2 = self.task_manager.get_label_offset(t - 1)
            logits_dist = logits[train_ids,o1:o2]
            dist_target = target[:, o1:o2]
            dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, args.lwf_args['T'])
            loss = loss + args.lwf_args['lambda_dist']*dist_loss
        
        loss.backward()
        self.opt.step()

    def observe_task_IL(self, args, g, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        """
                        The method for learning the given tasks under the task-IL setting.

                        :param args: Same as the args in __init__().
                        :param g: The graph of the current task.
                        :param features: Node features of the current task.
                        :param labels: Labels of the nodes in the current task.
                        :param t: Index of the current task.
                        :param prev_model: The model obtained after learning the previous task.
                        :param train_ids: The indices of the nodes participating in the training.
                        :param ids_per_cls: Indices of the nodes in each class (not in use in the current baseline).
                        :param dataset: The entire dataset (not in use in the current baseline).

                        """
        self.net.train()
        # if new task
        if t != self.current_task:
            self.current_task = t

        self.net.zero_grad()
        self.cuda()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        logits = self.net(g, features)
        if isinstance(logits, tuple):
            logits = logits[0]
        output_labels = labels[train_ids]

        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        loss = self.ce(logits[train_ids, offset1:offset2], output_labels- offset1, weight=loss_w_[offset1: offset2])

        if t > 0:
            # print('here',prev_model)
            target = prev_model(g, features)
            if isinstance(target, tuple):
                target = target[0]
            for oldt in range(t):
                o1, o2 = self.task_manager.get_label_offset(oldt-1)[1], self.task_manager.get_label_offset(oldt)[1]
                logits_dist = logits[train_ids,o1:o2]
                dist_target = target[train_ids,o1:o2]
                dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, args.lwf_args['T'])
                loss = loss + args.lwf_args['lambda_dist'] * dist_loss

        loss.backward()
        self.opt.step()

    def observe_task_IL_batch(self, args, g, dataloader, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        """
                        The method for learning the given tasks under the task-IL setting with mini-batch training.

                        :param args: Same as the args in __init__().
                        :param g: The graph of the current task.
                        :param dataloader: The data loader for mini-batch training
                        :param features: Node features of the current task.
                        :param labels: Labels of the nodes in the current task.
                        :param t: Index of the current task.
                        :param prev_model: The model obtained after learning the previous task.
                        :param train_ids: The indices of the nodes participating in the training.
                        :param ids_per_cls: Indices of the nodes in each class (currently not in use).
                        :param dataset: The entire dataset (currently not in use).

                        """
        self.net.train()
        # if new task
        if t != self.current_task:
            self.current_task = t

        self.net.zero_grad()
        self.cuda()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
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
            output_labels = output_labels - offset1

            output_predictions,_ = self.net.forward_batch(blocks, input_features)
            if args.classifier_increase:
                loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output_predictions, output_labels, weight=self.aux_loss_w_)

            # knowledge distillation
            if t > 0:
                target = prev_model.forward_batch(blocks, input_features)
                if isinstance(target, tuple):
                    target = target[0]
                for oldt in range(t):
                    o1, o2 = self.task_manager.get_label_offset(oldt - 1)[1], self.task_manager.get_label_offset(oldt)[
                        1]
                    logits_dist = output_predictions[:, o1:o2]
                    dist_target = target[:, o1:o2]
                    dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, args.lwf_args['T'])
                    loss = loss + args.lwf_args['lambda_dist'] * dist_loss

            loss.backward()
            self.opt.step()

    def observe_class_IL_batch(self, args, g, dataloader, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        """
                                The method for learning the given tasks under the class-IL setting with mini-batch training.

                                :param args: Same as the args in __init__().
                                :param g: The graph of the current task.
                                :param dataloader: The data loader for mini-batch training
                                :param features: Node features of the current task.
                                :param labels: Labels of the nodes in the current task.
                                :param t: Index of the current task.
                                :param prev_model: The model obtained after learning the previous task.
                                :param train_ids: The indices of the nodes participating in the training.
                                :param ids_per_cls: Indices of the nodes in each class (currently not in use).
                                :param dataset: The entire dataset (currently not in use).

                                """
        self.net.train()
        # if new task
        if t != self.current_task:
            self.current_task = t

        self.net.zero_grad()
        self.cuda()
        offset1, offset2 = self.task_manager.get_label_offset(t)
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

            # knowledge distillation
            if t > 0:
                target = prev_model.forward_batch(blocks, input_features)
                if isinstance(target, tuple):
                    target = target[0]
                for oldt in range(t):
                    o1, o2 = self.task_manager.get_label_offset(oldt)
                    logits_dist = output_predictions[:, o1:o2]
                    dist_target = target[:, o1:o2]
                    dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, args.lwf_args['T'])
                    loss = loss + args.lwf_args['lambda_dist'] * dist_loss

            loss.backward()
            self.opt.step()