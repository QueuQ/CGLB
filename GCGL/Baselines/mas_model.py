import torch
from torch.optim import Adam
from dgllife.utils import EarlyStopping, Meter
import numpy as np
from torch.utils.data import DataLoader

def predict(args, model, bg):
    node_feats = bg.ndata.pop(args['node_data_field']).cuda()
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata.pop(args['edge_data_field']).cuda()
        return model(bg, node_feats, edge_feats)
    else:
        return model(bg, node_feats)


class NET(torch.nn.Module):
    """
            MAS baseline for GCGL tasks

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
        self.reg = args['mas_args']['memory_strength']
        self.current_task = 0
        self.optpar = []
        self.fisher = []
        self.n_observed_data = 0
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
        self.net.train()

        if task_i != self.current_task:
            self.optpar = []
            #self.fisher = []
            new_fisher_all_batches = []
            self.optimizer.zero_grad()

            n_new_data=0
            for batch_id, batch_data in enumerate(self.data_loader): # self.dataloader uses full batch by default
                smiles, bg, labels, masks = batch_data
                masks = masks.cuda(args['gpu'])
                n_new_data+=masks[:,self.current_task].sum().item()
                bg = bg.to(f"cuda:{args['gpu']}")
                #labels, masks = labels.cuda(), masks.cuda()
                output = predict(args, self.net, bg)[:,self.current_task] * (masks[:,self.current_task] != 0).float()

                output.pow_(2)
                loss = output.mean()
                self.net.zero_grad()
                loss.backward()

                #i = 0
                new_fisher_batch = []
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2)  # error
                        new_fisher_batch.append(pg)
                        #self.optpar.append(p.data.clone())
                    except:
                        print('here')
                new_fisher_all_batches.append(new_fisher_batch)
            self.current_task = task_i

            # update the new fishers into existing ones
            new_fisher = []
            for f in range(len(new_fisher_batch)):  # for each recorded parameter
                pg_all_batches = []
                for b in new_fisher_all_batches:
                    pg_all_batches.append(b[f])
                new_fisher.append(sum(pg_all_batches) / len(pg_all_batches))
            if self.n_observed_data==0:
                self.fisher=new_fisher
            else:
                for f in range(len(new_fisher)):
                    self.fisher[f] = (self.fisher[f] * self.n_observed_data + new_fisher[f] * n_new_data)/(self.n_observed_data + n_new_data)
            self.n_observed_data+=n_new_data

            # record parameters
            for p in self.net.parameters():
                try:
                    pg = p.grad.data.clone().pow(2)  # error
                    self.optpar.append(p.data.clone())
                except:
                    print('here')

        train_meter = Meter()
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(f"cuda:{args['gpu']}")
            labels, masks = labels.cuda(args['gpu']), masks.cuda(args['gpu'])
            logits = predict(args, self.net, bg)

            # Mask non-existing labels
            loss = loss_criterion(logits, labels) * (masks != 0).float()
            loss = loss[:,task_i].mean()

            if task_i > 0:
                i = 0
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2)
                        l = self.reg * self.fisher[i]
                        l = l * (p - self.optpar[i]).pow(2)
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
            # update fisher and parameters
            new_fisher_all_natches = []
            self.optimizer.zero_grad()

            n_new_data=0
            for batch_id, batch_data in enumerate(self.data_loader[task_i]):
                smiles, bg, labels, masks = batch_data
                bg = bg.to(f"cuda:{args['gpu']}")
                n_new_data+=labels.shape[0]

                output = predict(args, self.net, bg)[:, args['tasks'][self.current_task]]

                '''
                # no need to class balance due to the mean() also considers the number of examples per cls
                n_per_cls = [(labels == j).sum() for j in clss]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                for i, c in enumerate(clss):
                    labels[labels == c] = i
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
                loss_w = torch.tensor([loss_w_[i] for i in labels.long().tolist()]).to(device='cuda:{}'.format(args['gpu']))
                loss_w*output.pow_(2)
                '''

                output.pow_(2)
                loss = output.mean()
                self.net.zero_grad()
                loss.backward()

                fisher_batch=[]
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2)  # error
                        fisher_batch.append(pg)
                    except:
                        1
                new_fisher_all_natches.append(fisher_batch)
                self.current_task = task_i

            new_fisher=[]
            for f in range(len(fisher_batch)):
                pg_all_batches=[]
                for b in new_fisher_all_natches:
                    pg_all_batches.append(b[f])
                new_fisher.append(sum(pg_all_batches)/len(pg_all_batches))

            # update new fisher into existing ones
            if self.n_observed_data==0:
                self.fisher=new_fisher
            else:
                for f in range(len(new_fisher)):
                    self.fisher[f] = (self.fisher[f]*self.n_observed_data + new_fisher[f]+n_new_data)/(self.n_observed_data+n_new_data)
            self.n_observed_data+=n_new_data

            # store optimal parameters
            self.optpar = []
            for p in self.net.parameters():
                pd = p.data.clone()
                try:
                    pg = p.grad.data.clone().pow(2)  # error
                    self.optpar.append(pd)
                except:
                    1


        #train_meter = Meter()
        for batch_id, batch_data in enumerate(data_loader[task_i]):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(f"cuda:{args['gpu']}")
            labels, masks = labels.cuda(), masks.cuda()
            logits = predict(args, self.net, bg)

            # class balance
            n_per_cls = [(labels == j).sum() for j in clss]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
            # labels= labels.long()
            for i, c in enumerate(clss):
                labels[labels == c] = i

            # Mask non-existing labels
            loss = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()
            #loss = loss[:, task_i].mean()

            if task_i > 0:
                i = 0
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2)
                        l = self.reg * self.fisher[i]
                        l = l * (p - self.optpar[i]).pow(2)
                        loss += l.sum()
                        i += 1
                    except:
                        1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #train_meter.update(logits, labels, masks)

        #train_score = np.mean(train_meter.compute_metric(args['metric_name']))

    def observe_tskIL_multicls(self, data_loader, loss_criterion, task_i, args):
        """
                                The method for learning the given tasks under the task-IL setting with multi-class classification datasets.

                                :param data_loader: The data loader for mini-batch training.
                                :param loss_criterion: The loss function.
                                :param task_i: Index of the current task.
                                :param args: Same as the args in __init__().

                                """
        self.net.train()
        clss = args['tasks'][task_i]
        if task_i != self.current_task:
            #self.optpar = []
            #self.fisher = []
            new_fisher_all_batches = []
            self.optimizer.zero_grad()

            n_new_data=0
            for batch_id, batch_data in enumerate(self.data_loader[self.current_task]):
                smiles, bg, labels, masks = batch_data
                n_new_data += labels.shape[0]
                bg = bg.to(f"cuda:{args['gpu']}")
                output = predict(args, self.net, bg)[:, args['tasks'][self.current_task]]

                output.pow_(2)
                loss = output.mean()
                self.net.zero_grad()
                loss.backward()

                new_fisher_batch=[]
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2)  # error
                        new_fisher_batch.append(pg)
                    except:
                        1
                new_fisher_all_batches.append(new_fisher_batch)
                self.current_task = task_i

            # update new fisher into existing ones:
            new_fisher = []
            for f in range(len(new_fisher_batch)):
                pg_all_batches=[]
                for b in new_fisher_all_batches:
                    pg_all_batches.append(b[f])
                new_fisher.append(sum(pg_all_batches)/len(pg_all_batches))
            if self.n_observed_data==0:
                self.fisher=new_fisher
            else:
                for f in range(len(new_fisher)):
                    self.fisher[f]=(self.fisher[f]*self.n_observed_data + new_fisher[f]*n_new_data)/(self.n_observed_data + n_new_data)
            self.n_observed_data+=n_new_data

            # store parameters
            self.optpar=[]
            for p in self.net.parameters():
                try:
                    pg = p.grad.data.clone().pow(2)
                    pd = p.data.clone()
                    self.optpar.append(pd)
                except:
                    1
        #train_meter = Meter()
        for batch_id, batch_data in enumerate(data_loader[task_i]):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(f"cuda:{args['gpu']}")
            labels, masks = labels.cuda(args['gpu']), masks.cuda(args['gpu'])
            logits = predict(args, self.net, bg)

            # class balance
            n_per_cls = [(labels == j).sum() for j in clss]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
            # labels= labels.long()
            for i, c in enumerate(clss):
                labels[labels == c] = i

            # Mask non-existing labels
            loss = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()
            #loss = loss[:, task_i].mean()

            if task_i > 0:
                i = 0
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2)
                        l = self.reg * self.fisher[i]
                        l = l * (p - self.optpar[i]).pow(2)
                        loss += l.sum()
                        i += 1
                    except:
                        1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #train_meter.update(logits, labels, masks)

        #train_score = np.mean(train_meter.compute_metric(args['metric_name']))

    def observe_deprecated(self, data_loader, loss_criterion, task_i, args):
        """
                        The method for learning the given tasks under the task-IL setting with multi-label classification datasets.

                        :param data_loader: The data loader for mini-batch training.
                        :param loss_criterion: The loss function.
                        :param task_i: Index of the current task.
                        :param args: Same as the args in __init__().

                        """
        self.net.train()

        if task_i != self.current_task:
            self.optpar = []
            self.fisher = []
            self.optimizer.zero_grad()

            for batch_id, batch_data in enumerate(self.data_loader):
                smiles, bg, labels, masks = batch_data
                bg = bg.to(f"cuda:{args['gpu']}")
                # labels, masks = labels.cuda(), masks.cuda()
                output = predict(args, self.net, bg)[:, self.current_task]

                output.pow_(2)
                loss = output.mean()
                self.net.zero_grad()
                loss.backward()

                for p in self.net.parameters():
                    pd = p.data.clone()
                    try:
                        pg = p.grad.data.clone().pow(2)  # error
                        self.fisher.append(pg)
                        self.optpar.append(pd)
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

            if task_i > 0:
                i = 0
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2)
                        l = self.reg * self.fisher[i]
                        l = l * (p - self.optpar[i]).pow(2)
                        loss += l.sum()
                        i += 1
                    except:
                        1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_meter.update(logits, labels, masks)

        train_score = np.mean(train_meter.compute_metric(args['metric_name']))

    def observe_clsIL_deprecated(self, data_loader, loss_criterion, task_i, args):
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
            self.optpar = []
            self.fisher = []
            self.optimizer.zero_grad()

            for batch_id, batch_data in enumerate(self.data_loader[task_i]):
                smiles, bg, labels, masks = batch_data
                bg = bg.to(f"cuda:{args['gpu']}")
                labels, masks = labels.cuda(), masks.cuda()

                output = predict(args, self.net, bg)[:, self.current_task]

                '''
                # no need to class balance due to the mean() also considers the number of examples per cls
                n_per_cls = [(labels == j).sum() for j in clss]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                for i, c in enumerate(clss):
                    labels[labels == c] = i
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
                loss_w = torch.tensor([loss_w_[i] for i in labels.long().tolist()]).to(device='cuda:{}'.format(args['gpu']))
                loss_w*output.pow_(2)
                '''

                output.pow_(2)
                loss = output.mean()
                self.net.zero_grad()
                loss.backward()

                for p in self.net.parameters():
                    pd = p.data.clone()
                    try:
                        pg = p.grad.data.clone().pow(2)  # error
                        self.fisher.append(pg)
                        self.optpar.append(pd)
                    except:
                        1
                self.current_task = task_i

        #train_meter = Meter()
        for batch_id, batch_data in enumerate(data_loader[task_i]):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(f"cuda:{args['gpu']}")
            labels, masks = labels.cuda(), masks.cuda()
            logits = predict(args, self.net, bg)

            # class balance
            n_per_cls = [(labels == j).sum() for j in clss]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
            # labels= labels.long()
            for i, c in enumerate(clss):
                labels[labels == c] = i

            # Mask non-existing labels
            loss = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()
            #loss = loss[:, task_i].mean()

            if task_i > 0:
                i = 0
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2)
                        l = self.reg * self.fisher[i]
                        l = l * (p - self.optpar[i]).pow(2)
                        loss += l.sum()
                        i += 1
                    except:
                        1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #train_meter.update(logits, labels, masks)

        #train_score = np.mean(train_meter.compute_metric(args['metric_name']))

    def observe_tskIL_multicls_deprecated(self, data_loader, loss_criterion, task_i, args):
        """
                                The method for learning the given tasks under the task-IL setting with multi-class classification datasets.

                                :param data_loader: The data loader for mini-batch training.
                                :param loss_criterion: The loss function.
                                :param task_i: Index of the current task.
                                :param args: Same as the args in __init__().

                                """
        self.net.train()
        clss = args['tasks'][task_i]
        if task_i != self.current_task:
            self.optpar = []
            self.fisher = []
            self.optimizer.zero_grad()

            for batch_id, batch_data in enumerate(self.data_loader[task_i]):
                smiles, bg, labels, masks = batch_data
                bg = bg.to(f"cuda:{args['gpu']}")
                labels, masks = labels.cuda(), masks.cuda()
                output = predict(args, self.net, bg)[:, self.current_task]

                '''
                # class balance
                n_per_cls = [(labels == j).sum() for j in clss]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
                loss_w = torch.tensor([loss_w_[i] for i in labels.long().tolist()]).to(device='cuda:{}'.format(args['gpu']))
                loss_w*output.pow_(2)
                '''

                output.pow_(2)
                loss = output.mean()
                self.net.zero_grad()
                loss.backward()

                for p in self.net.parameters():
                    pd = p.data.clone()
                    try:
                        pg = p.grad.data.clone().pow(2)  # error
                        self.fisher.append(pg)
                        self.optpar.append(pd)
                    except:
                        1
                self.current_task = task_i

        #train_meter = Meter()
        for batch_id, batch_data in enumerate(data_loader[task_i]):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(f"cuda:{args['gpu']}")
            labels, masks = labels.cuda(), masks.cuda()
            logits = predict(args, self.net, bg)

            # class balance
            n_per_cls = [(labels == j).sum() for j in clss]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args['gpu']))
            # labels= labels.long()
            for i, c in enumerate(clss):
                labels[labels == c] = i

            # Mask non-existing labels
            loss = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()
            #loss = loss[:, task_i].mean()

            if task_i > 0:
                i = 0
                for p in self.net.parameters():
                    try:
                        pg = p.grad.data.clone().pow(2)
                        l = self.reg * self.fisher[i]
                        l = l * (p - self.optpar[i]).pow(2)
                        loss += l.sum()
                        i += 1
                    except:
                        1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #train_meter.update(logits, labels, masks)

        #train_score = np.mean(train_meter.compute_metric(args['metric_name']))