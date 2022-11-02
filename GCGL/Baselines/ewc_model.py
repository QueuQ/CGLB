import torch
from torch.optim import Adam
from dgllife.utils import Meter
import numpy as np

def predict(args, model, bg):
    node_feats = bg.ndata.pop(args['node_data_field']).cuda()
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata.pop(args['edge_data_field']).cuda()
        return model(bg, node_feats, edge_feats)
    else:
        return model(bg, node_feats)


class NET(torch.nn.Module):
    """
            EWC baseline for GCGL tasks

            :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
            :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

            """
    def __init__(self,
                 model,
                 args):
        super(NET, self).__init__()

        # setup network
        self.net = model
        self.optimizer = Adam(model.parameters(), lr=args['lr'])

        # mas
        self.reg = args['ewc_args']['memory_strength']
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.data_loader = None


    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, data_loader, loss_criterion, task_i, args):
        """
                        The method for learning the given tasks under the task-IL setting with multi-label classification datasets.

                        :param data_loader: The data loader for mini-batch training.
                        :param loss_criterion: The loss function.
                        :param task_i: Index of the current task.
                        :param args: Same as the args in __init__().

                        """
        # not real cls-IL, just task Il under multi-class setting
        self.net.train()
        if task_i != self.current_task:
            self.optimizer.zero_grad()

            for batch_id, batch_data in enumerate(self.data_loader):
                smiles, bg, labels, masks = batch_data
                bg = bg.to(f"cuda:{args['gpu']}")
                labels, masks = labels.cuda(), masks.cuda()
                logits = predict(args, self.net, bg)

                loss = loss_criterion(logits, labels) * (masks != 0).float()
                loss = loss[:, self.current_task].mean()
                loss.backward()

                self.fisher[self.current_task] = []
                self.optpar[self.current_task] = []
                for p in self.net.parameters():
                    pd = p.data.clone()
                    try:
                        pg = p.grad.data.clone().pow(2)
                        self.fisher[self.current_task].append(pg)
                        self.optpar[self.current_task].append(pd)
                    except:
                        1

                self.current_task = task_i

        train_meter = Meter()
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(f"cuda:{args['gpu']}")
            labels, masks = labels.cuda(), masks.cuda()
            logits = predict(args, self.net, bg)

            # Mask non-existing labels
            loss = loss_criterion(logits, labels) * (masks != 0).float()
            loss = loss[:, task_i].mean()

            for tt in range(task_i):
                i = 0
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2)
                        l = self.reg * self.fisher[tt][i]
                        l = l * (p - self.optpar[tt][i]).pow(2)
                        loss += l.sum()
                        i += 1
                    except:
                        1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_meter.update(logits, labels, masks)

        train_score = np.mean(train_meter.compute_metric(args['metric_name']))

    def observe_clsIL(self, data_loader, loss_criterion, task_i, args):
        """
                                        The method for learning the given tasks under the class-IL setting with multi-class classification datasets.

                                        :param data_loader: The data loader for mini-batch training.
                                        :param loss_criterion: The loss function.
                                        :param task_i: Index of the current task.
                                        :param args: Same as the args in __init__().

                                        """
        self.net.train()
        clss = []
        for tid in range(task_i + 1):
            clss.extend(args['tasks'][tid])
        if task_i != self.current_task:
            # store weights for regularization
            clss_old = [] # activated output heads in last task (self.current_task)
            for tid in range(task_i):
                clss_old.extend(args['tasks'][tid])
            self.optimizer.zero_grad()

            for batch_id, batch_data in enumerate(self.data_loader[self.current_task]):
                smiles, bg, labels, masks = batch_data
                bg = bg.to(f"cuda:{args['gpu']}")
                labels, masks = labels.cuda(), masks.cuda()
                logits = predict(args, self.net, bg)

                # class balance
                n_per_cls = [(labels == j).sum() for j in clss_old]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
                # labels= labels.long()
                for i, c in enumerate(clss_old):
                    labels[labels == c] = i

                loss = loss_criterion(logits[:, clss_old], labels.long(), weight=loss_w_).float()
                # loss = loss[:,self.current_task].mean()
                loss.backward()

                self.fisher[self.current_task] = []
                self.optpar[self.current_task] = []
                for p in self.net.parameters():
                    pd = p.data.clone()
                    try:
                        pg = p.grad.data.clone().pow(2)
                        self.fisher[self.current_task].append(pg)
                        self.optpar[self.current_task].append(pd)
                    except:
                        1

                self.current_task = task_i

        # train_meter = Meter()
        for batch_id, batch_data in enumerate(data_loader[task_i]):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(f"cuda:{args['gpu']}")
            labels, masks = labels.cuda(), masks.cuda()
            logits = predict(args, self.net, bg)

            # class balance
            n_per_cls = [(labels == j).sum() for j in clss]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
            for i, c in enumerate(clss):
                labels[labels == c] = i

            # Mask non-existing labels
            loss = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()
            # loss = loss[:,task_i].mean()

            for tt in range(task_i):
                i = 0
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2)
                        l = self.reg * self.fisher[tt][i]
                        l = l * (p - self.optpar[tt][i]).pow(2)
                        loss += l.sum()
                        i += 1
                    except:
                        1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # train_meter.update(logits, labels, masks)

        # train_score = np.mean(train_meter.compute_metric(args['metric_name']))

    def observe_tskIL_multicls(self, data_loader, loss_criterion, task_i, args):
        """
                                The method for learning the given tasks under the task-IL setting with multi-class classification datasets.

                                :param data_loader: The data loader for mini-batch training.
                                :param loss_criterion: The loss function.
                                :param task_i: Index of the current task.
                                :param args: Same as the args in __init__().

                                """
        self.net.train()
        if task_i != self.current_task:
            # store weights for regularization
            clss_old = args['tasks'][self.current_task]
            self.optimizer.zero_grad()

            for batch_id, batch_data in enumerate(self.data_loader[self.current_task]):
                smiles, bg, labels, masks = batch_data
                bg = bg.to(f"cuda:{args['gpu']}")
                labels, masks = labels.cuda(), masks.cuda()
                logits = predict(args, self.net, bg)

                # class balance
                n_per_cls = [(labels == j).sum() for j in clss_old]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
                # labels= labels.long()
                for i, c in enumerate(clss_old):
                    labels[labels == c] = i

                loss = loss_criterion(logits[:, clss_old], labels.long(), weight=loss_w_).float()
                # loss = loss[:,self.current_task].mean()
                loss.backward()

                self.fisher[self.current_task] = []
                self.optpar[self.current_task] = []
                for p in self.net.parameters():
                    pd = p.data.clone()
                    try:
                        pg = p.grad.data.clone().pow(2)
                        self.fisher[self.current_task].append(pg)
                        self.optpar[self.current_task].append(pd)
                    except:
                        1

                self.current_task = task_i

        # train_meter = Meter()
        clss = args['tasks'][task_i]
        for batch_id, batch_data in enumerate(data_loader[task_i]):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(f"cuda:{args['gpu']}")
            labels, masks = labels.cuda(), masks.cuda()
            logits = predict(args, self.net, bg)

            # class balance
            n_per_cls = [(labels == j).sum() for j in clss]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
            for i, c in enumerate(clss):
                labels[labels == c] = i

            # Mask non-existing labels
            loss = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()
            # loss = loss[:,task_i].mean()

            for tt in range(task_i):
                i = 0
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2)
                        l = self.reg * self.fisher[tt][i]
                        l = l * (p - self.optpar[tt][i]).pow(2)
                        loss += l.sum()
                        i += 1
                    except:
                        1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # train_meter.update(logits, labels, masks)

        # train_score = np.mean(train_meter.compute_metric(args['metric_name']))

