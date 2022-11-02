import torch
from torch.autograd import Variable
import numpy as np

class NET(torch.nn.Module):
    """
        Bare model baseline for NCGL tasks

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
        self.net.twp=True
        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # setup losses
        self.ce = torch.nn.functional.cross_entropy
        # setup memories
        self.current_task = 0
        self.fisher_loss = {}
        self.fisher_att = {}
        self.optpar = {}
        self.mem_mask = None
        # hyper-parameters
        self.lambda_l = args.twp_args['lambda_l']
        self.lambda_t = args.twp_args['lambda_t']
        self.beta = args.twp_args['beta']
        self.epochs = 0

    def forward(self, features):
        output, elist = self.net(features)
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
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()

        # train the current task
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        output, elist = self.net(g, features)
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

        loss.backward(retain_graph=True)
        grad_norm = 0
        for p in self.net.parameters():
            pg = p.grad.data.clone() # this follows the official code of TWP, performance does not change obviously with pg = p.grad.clone()
            # pg = p.grad.clone()
            grad_norm += torch.norm(pg, p=1)

        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.lambda_l * self.fisher_loss[tt][i] + self.lambda_t * self.fisher_att[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()

        loss = loss + self.beta * grad_norm
        loss.backward()
        self.opt.step()

        # store gradients for future tasks
        if last_epoch == 0:
            self.net.zero_grad()
            self.fisher_loss[t] = []
            self.fisher_att[t] = []
            self.optpar[t] = []

            output, elist = self.net(g, features)
            if args.classifier_increase:
                loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output[train_ids], output_labels, weight=loss_w_)
            loss.backward(retain_graph=True)

            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar[t].append(pd)
                self.fisher_loss[t].append(pg)

            eloss = torch.norm(elist[0])
            eloss.backward()
            for p in self.net.parameters():
                pg = p.grad.data.clone().pow(2)
                self.fisher_att[t].append(pg)

            self.current_task = t

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
        ############### try to simulate the original setting of twp
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()

        # train the current task
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        output, elist = self.net(g, features)
        output_labels = labels[train_ids]
        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        output_labels = output_labels - offset1
        loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])

        loss.backward(retain_graph=True)
        grad_norm = 0

        # the following code is same as the official TWP code, performance does not change obviously with pg = p.grad.clone()
        for p in self.net.parameters():
            pg = p.grad.data.clone()
            grad_norm += torch.norm(pg, p=1)

        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.lambda_l * self.fisher_loss[tt][i] + self.lambda_t * self.fisher_att[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()

        loss = loss + self.beta * grad_norm
        loss.backward()
        self.opt.step()

        # store gradients for future tasks
        if last_epoch == 0:
            self.net.zero_grad()
            self.fisher_loss[t] = []
            self.fisher_att[t] = []
            self.optpar[t] = []

            output, elist = self.net(g, features)

            # loss = self.ce((output[self.mem_mask, offset1: offset2]), labels[self.mem_mask] - offset1)
            if args.classifier_increase:
                loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            loss.backward(retain_graph=True)

            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar[t].append(pd)
                self.fisher_loss[t].append(pg)

            eloss = torch.norm(elist[0])
            eloss.backward()
            for p in self.net.parameters():
                pg = p.grad.data.clone().pow(2)
                self.fisher_att[t].append(pg)

            self.current_task = t

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
        ############### try to simulate the original setting of twp
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()

        # train the current task
        self.net.zero_grad()
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
            output_predictions, elist = self.net.forward_batch(blocks, input_features)
            if args.classifier_increase:
                loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output_predictions, output_labels, weight=self.aux_loss_w_)
            loss.backward(retain_graph=True)
            grad_norm = 0
            for p in self.net.parameters():
                pg = p.grad.data.clone()
                grad_norm += torch.norm(pg, p=1)
            for tt in range(t):
                for i, p in enumerate(self.net.parameters()):
                    l = self.lambda_l * self.fisher_loss[tt][i] + self.lambda_t * self.fisher_att[tt][i]
                    l = l * (p - self.optpar[tt][i]).pow(2)
                    loss += l.sum()
            loss = loss + self.beta * grad_norm
            loss.backward()
            self.opt.step()

        # store gradients for future tasks
        if last_epoch == 0:
            self.net.zero_grad()
            self.fisher_loss[t] = []
            self.fisher_att[t] = []
            self.optpar[t] = []

            pgss_floss,pgss_fatt = [], []
            for input_nodes, output_nodes, blocks in dataloader:
                pgs_floss,pgs_fatt = [], []
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
                output_predictions, elist = self.net.forward_batch(blocks, input_features)
                if args.classifier_increase:
                    loss = self.ce(output_predictions[:, offset1:offset2], output_labels,weight=loss_w_[offset1: offset2])
                else:
                    loss = self.ce(output_predictions, output_labels, weight=self.aux_loss_w_)
                loss.backward(retain_graph=True)
                for p in self.net.parameters():
                    pg = p.grad.data.clone().pow(2)
                    pgs_floss.append(pg)
                pgss_floss.append(pgs_floss)
                eloss = torch.norm(elist[0])
                eloss.backward()
                for p in self.net.parameters():
                    pg = p.grad.data.clone().pow(2)
                    pgs_fatt.append(pg)
                pgss_fatt.append(pgs_floss)

            for i,p in enumerate(self.net.parameters()):
                pg_floss_,pgs_fatt_ = [],[]
                for pgs_ in pgss_floss:
                    pg_floss_.append(pgs_[i])
                for pgs_ in pgss_fatt:
                    pgs_fatt_.append(pgs_[i])
                pd = p.data.clone()
                self.optpar[t].append(pd)
                self.fisher_loss[t].append(sum(pg_floss_)/len(pg_floss_))
                self.fisher_att[t].append(sum(pgs_fatt_)/len(pgs_fatt_))

            self.current_task = t

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
        ############### try to simulate the original setting of twp
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        self.net.train()

        # train the current task
        self.net.zero_grad()
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
            output_predictions, elist = self.net.forward_batch(blocks, input_features)
            if args.classifier_increase:
                loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output_predictions, output_labels, weight=self.aux_loss_w_)
            loss.backward(retain_graph=True)
            grad_norm = 0
            for p in self.net.parameters():
                pg = p.grad.data.clone()
                grad_norm += torch.norm(pg, p=1)
            for tt in range(t):
                for i, p in enumerate(self.net.parameters()):
                    l = self.lambda_l * self.fisher_loss[tt][i] + self.lambda_t * self.fisher_att[tt][i]
                    l = l * (p - self.optpar[tt][i]).pow(2)
                    loss += l.sum()
            loss = loss + self.beta * grad_norm
            loss.backward()
            self.opt.step()

        # store gradients for future tasks
        if last_epoch == 0:
            self.net.zero_grad()
            self.fisher_loss[t] = []
            self.fisher_att[t] = []
            self.optpar[t] = []

            pgss_floss,pgss_fatt = [], []
            for input_nodes, output_nodes, blocks in dataloader:
                pgs_floss,pgs_fatt = [], []
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
                output_predictions, elist = self.net.forward_batch(blocks, input_features)
                if args.classifier_increase:
                    loss = self.ce(output_predictions[:, offset1:offset2], output_labels,weight=loss_w_[offset1: offset2])
                else:
                    loss = self.ce(output_predictions, output_labels, weight=self.aux_loss_w_)
                loss.backward(retain_graph=True)
                for p in self.net.parameters():
                    pg = p.grad.data.clone().pow(2)
                    pgs_floss.append(pg)
                pgss_floss.append(pgs_floss)
                eloss = torch.norm(elist[0])
                eloss.backward()
                for p in self.net.parameters():
                    pg = p.grad.data.clone().pow(2)
                    pgs_fatt.append(pg)
                pgss_fatt.append(pgs_floss)

            for i,p in enumerate(self.net.parameters()):
                pg_floss_,pgs_fatt_ = [],[]
                for pgs_ in pgss_floss:
                    pg_floss_.append(pgs_[i])
                for pgs_ in pgss_fatt:
                    pgs_fatt_.append(pgs_[i])
                pd = p.data.clone()
                self.optpar[t].append(pd)
                self.fisher_loss[t].append(sum(pg_floss_)/len(pg_floss_))
                self.fisher_att[t].append(sum(pgs_fatt_)/len(pgs_fatt_))

            self.current_task = t
