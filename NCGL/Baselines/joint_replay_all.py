import torch
import copy
from .ergnn_utils import *

class NET(torch.nn.Module):
    """
        Jointly trained model baseline for NCGL tasks

        In this baseline, when ever a new task comes, the model will be jointly trained on all existing tasks.

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

        # setup network
        self.net = model

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = -1

    def forward(self, features):
        output = self.net(features)
        return output


    def observe(self, args, gs, featuress, labelss, t, train_idss, ids_per_clss, dataset):
        """
                The method for learning the given tasks under the class-IL setting.

                :param args: Same as the args in __init__().
                :param gs: The graphs of all the existing tasks to be jointly trained on.
                :param featuress: Node features of the nodes in all the existing tasks.
                :param labelss: Labels of the nodes in all the existing tasks.
                :param t: Index of the newest task.
                :param train_idss: The indices of the nodes participating in the training in all the existing tasks.
                :param ids_per_clss: Indices of the nodes in each class of all the existing tasks.
                :param dataset: The entire dataset (not in use in the current baseline).

                """
        self.net.train()
        if t!=self.current_task:
            self.current_task = t
            self.net.reset_params()

        self.net.zero_grad()
        loss = 0
        offset1, offset2 = self.task_manager.get_label_offset(t)
        for g,features, labels_all, train_ids, ids_per_cls in zip(gs, featuress, labelss, train_idss, ids_per_clss):
            labels = labels_all[train_ids]
            output, _ = self.net(g, features)
            if args.cls_balance:
                n_per_cls = [(labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            if args.classifier_increase:
                loss += self.ce(output[train_ids, offset1:offset2], labels, weight=loss_w_[offset1: offset2])
            else:
                loss += self.ce(output[train_ids], labels, weight=loss_w_)
        loss.backward()
        self.opt.step()

    def observe_task_IL(self, args, gs, featuress, labelss, t, train_idss, ids_per_clss, dataset):
        """
                The method for learning the given tasks under the task-IL setting.

                :param args: Same as the args in __init__().
                :param gs: The graphs of all the existing tasks to be jointly trained on.
                :param featuress: Node features of the nodes in all the existing tasks.
                :param labelss: Labels of the nodes in all the existing tasks.
                :param t: Index of the newest task.
                :param train_idss: The indices of the nodes participating in the training in all the existing tasks.
                :param ids_per_clss: Indices of the nodes in each class of all the existing tasks.
                :param dataset: The entire dataset (not in use in the current baseline).

                """
        self.net.train()
        if t!=self.current_task:
            self.current_task = t
            self.net.reset_params()

        self.net.zero_grad()
        loss = 0

        for tt, (g,features, labels_all, train_ids, ids_per_cls) in enumerate(zip(gs, featuress, labelss, train_idss, ids_per_clss)):
            labels = labels_all[train_ids]
            offset1, offset2 = self.task_manager.get_label_offset(tt - 1)[1], self.task_manager.get_label_offset(tt)[1]
            output, _ = self.net(g, features)
            if args.cls_balance:
                n_per_cls = [(labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            loss += self.ce(output[train_ids, offset1:offset2], labels-offset1, weight=loss_w_[offset1: offset2])
        loss.backward()
        self.opt.step()

    def observe_task_IL_batch(self, args, gs, dataloader, featuress, labelss, t, train_idss, ids_per_clss, dataset):
        """
                        The method for learning the given tasks under the task-IL setting with mini-batch training.

                        :param args: Same as the args in __init__().
                        :param gs: The graphs of all the existing tasks to be jointly trained on.
                        :param dataloader: The data loader for mini-batch training
                        :param featuress: Node features of the nodes in all the existing tasks.
                        :param labelss: Labels of the nodes in all the existing tasks.
                        :param t: Index of the newest task.
                        :param train_idss: The indices of the nodes participating in the training in all the existing tasks.
                        :param ids_per_clss: Indices of the nodes in each class of all the existing tasks.
                        :param dataset: The entire dataset (not in use in the current baseline).

                        """
        self.net.train()
        if t!=self.current_task:
            self.current_task = t
            self.net.reset_params()

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
            loss = 0
            for tt in range(t+1):
                ids_current_task = []
                for cls_c in args.task_seq[tt]:
                    try:
                        ids_current_task.extend((output_labels==cls_c).nonzero()[:,0].tolist())
                    except:
                        a=1 # maybe not all classes will appear each batch
                offset1_c, offset2_c = self.task_manager.get_label_offset(tt - 1)[1], self.task_manager.get_label_offset(tt)[1]
                loss+=self.ce(output_predictions[ids_current_task,offset1_c:offset2_c], output_labels[ids_current_task]-offset1_c, weight=loss_w_[offset1_c:offset2_c])
            loss.backward()
            self.opt.step()

    def observe_class_IL_batch(self, args, gs, dataloader, featuress, labelss, t, train_idss, ids_per_clss, dataset):
        """
                                The method for learning the given tasks under the class-IL setting with mini-batch training.

                                :param args: Same as the args in __init__().
                                :param gs: The graphs of all the existing tasks to be jointly trained on.
                                :param dataloader: The data loader for mini-batch training
                                :param featuress: Node features of the nodes in all the existing tasks.
                                :param labelss: Labels of the nodes in all the existing tasks.
                                :param t: Index of the newest task.
                                :param train_idss: The indices of the nodes participating in the training in all the existing tasks.
                                :param ids_per_clss: Indices of the nodes in each class of all the existing tasks.
                                :param dataset: The entire dataset.

                                """
        self.net.train()
        if t!=self.current_task:
            self.current_task = t
            self.net.reset_params()
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
            loss = self.ce(output_predictions[:,offset1:offset2], output_labels, weight=loss_w_[offset1:offset2])
            loss.backward()
            self.opt.step()

    def observe_task_IL_crsedge(self, args, g, features, labels_all, t, train_ids, ids_per_cls_all, dataset):
        """
                        The method for learning the given tasks under the task-IL setting with inter-task edges.

                        :param args: Same as the args in __init__().
                        :param g: The graphs of all the existing tasks to be jointly trained on.
                        :param features: Node features of the nodes in all the existing tasks.
                        :param labels_all: Labels of the nodes in all the existing tasks.
                        :param t: Index of the newest task.
                        :param train_ids: The indices of the nodes participating in the training in all the existing tasks.
                        :param ids_per_cls_all: Indices of the nodes in each class of all the existing tasks.
                        :param dataset: The entire dataset (not in use in the current baseline).

                        """
        self.net.train()
        if t!=self.current_task:
            self.current_task = t
            self.net.reset_params()

        self.net.zero_grad()
        loss = 0
        output, _ = self.net(g, features)

        for tt,task in enumerate(args.task_seq[0:t+1]):
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in task]
            ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls_current_task]
            train_ids_current_task = []
            for ids in ids_per_cls_train:
                train_ids_current_task.extend(ids)
            labels = labels_all[train_ids_current_task]
            offset1, offset2 = self.task_manager.get_label_offset(tt - 1)[1], self.task_manager.get_label_offset(tt)[1]
            if args.cls_balance:
                n_per_cls = [(labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            loss += self.ce(output[train_ids_current_task, offset1:offset2], labels - offset1, weight=loss_w_[offset1: offset2])
        loss.backward()
        self.opt.step()

    def observe_class_IL_crsedge(self, args, g, features, labels_all, t, train_ids, ids_per_cls_all, dataset):
        """
                                The method for learning the given tasks under the class-IL setting with inter-task edges.

                                :param args: Same as the args in __init__().
                                :param g: The graphs of all the existing tasks to be jointly trained on.
                                :param features: Node features of the nodes in all the existing tasks.
                                :param labels_all: Labels of the nodes in all the existing tasks.
                                :param t: Index of the newest task.
                                :param train_ids: The indices of the nodes participating in the training in all the existing tasks.
                                :param ids_per_cls_all: Indices of the nodes in each class of all the existing tasks.
                                :param dataset: The entire dataset (not in use in the current baseline).

                                """
        self.net.train()
        if t!=self.current_task:
            self.current_task = t
            self.net.reset_params()

        self.net.zero_grad()
        loss = 0
        output, _ = self.net(g, features)
        offset1, offset2 = self.task_manager.get_label_offset(t)
        for tt,task in enumerate(args.task_seq[0:t+1]):
            ids_per_cls_current_task = [ids_per_cls_all[i] for i in task]
            ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls_current_task]
            train_ids_current_task = []
            for ids in ids_per_cls_train:
                train_ids_current_task.extend(ids)
            labels = labels_all[train_ids_current_task]

            if args.cls_balance:
                n_per_cls = [(labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            loss += self.ce(output[train_ids_current_task, offset1:offset2], labels, weight=loss_w_[offset1: offset2])
        loss.backward()
        self.opt.step()