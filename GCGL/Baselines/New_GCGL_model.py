import torch
from torch.optim import Adam
from dgllife.utils import EarlyStopping, Meter
import numpy as np
from torch.autograd import Variable


def predict(args, model, bg):
    node_feats = bg.ndata[args['node_data_field']].cuda()
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata[args['edge_data_field']].cuda()
        return model(bg, node_feats, edge_feats)
    else:
        return model(bg, node_feats)


def MultiClassCrossEntropy(logits, labels, T):
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return Variable(outputs.data, requires_grad=True).cuda()

class NET(torch.nn.Module):
    """
            LwF baseline for GCGL tasks

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

    def forward(self, args, bg):
        logits = predict(args, self.net, bg)
        return logits

    def observe_clsIL(self, data_loader, loss_criterion, task_i, args, prev_model):
        """
                                        The method for learning the given tasks under the class-IL setting with multi-class classification datasets.

                                        :param data_loader: The data loader for mini-batch training.
                                        :param loss_criterion: The loss function.
                                        :param task_i: Index of the current task.
                                        :param args: Same as the args in __init__().
                                        :param prev_model: The model obtained after learning the previous task.

                                        """
        self.net.train()
        clss = []
        for tid in range(task_i + 1):
            clss.extend(args['tasks'][tid])

        for batch_id, batch_data in enumerate(data_loader[task_i]):
            clss = args['tasks'][task_i]
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

            if task_i > 0:
                target = prev_model.forward(args, bg)
                for oldt in range(task_i):
                    clss = args['tasks'][oldt]
                    logits_dist = torch.unsqueeze(logits[:,clss], 0)
                    dist_target = torch.unsqueeze(target[:,clss], 0)
                    dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2).mean()
                    loss = loss + dist_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def observe_tskIL_multicls(self, data_loader, loss_criterion, task_i, args, prev_model):
        """
                                The method for learning the given tasks under the task-IL setting with multi-class classification datasets.

                                :param data_loader: The data loader for mini-batch training.
                                :param loss_criterion: The loss function.
                                :param task_i: Index of the current task.
                                :param args: Same as the args in __init__().
                                :param prev_model: The model obtained after learning the previous task.

                                """
        self.net.train()
        #train_meter = Meter()

        for batch_id, batch_data in enumerate(data_loader[task_i]):
            clss = args['tasks'][task_i]
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
            #loss = loss[:,task_i].mean()

            if task_i > 0:
                target = prev_model.forward(args, bg)
                for oldt in range(task_i):
                    clss = args['tasks'][oldt]
                    logits_dist = torch.unsqueeze(logits[:,clss], 0)
                    dist_target = torch.unsqueeze(target[:,clss], 0)
                    dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2).mean()
                    loss = loss + dist_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def observe(self, data_loader, loss_criterion, task_i, args, prev_model):
        """
                        The method for learning the given tasks under the task-IL setting with multi-label classification datasets.

                        :param data_loader: The data loader for mini-batch training.
                        :param loss_criterion: The loss function.
                        :param task_i: Index of the current task.
                        :param args: Same as the args in __init__().
                        :param prev_model: The model obtained after learning the previous task.

                        """
        self.net.train()
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
                target = prev_model.forward(args, bg)
                for oldt in range(task_i):
                    logits_dist = torch.unsqueeze(logits[:, oldt], 0)
                    dist_target = torch.unsqueeze(target[:, oldt], 0)
                    dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2)
                    loss = loss + dist_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_meter.update(logits, labels, masks)

        train_score = np.mean(train_meter.compute_metric(args['metric_name']))
        